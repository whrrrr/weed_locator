#!/home/whr/miniconda3/envs/lerobot/bin/python
"""Interactive keyboard Cartesian jog for LeRobot SO101.

This keeps LeRobot's official kinematics loaded and sends one safe IK command
per key press through /lerobot/write_joints.
"""

import math
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import JointState

from weed_locator.srv import ReadJoints, WriteJoints


LEROBOT_SRC = Path('/home/whr/lerobot/src')
if LEROBOT_SRC.exists():
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.model.kinematics import RobotKinematics


ARM_JOINTS = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
]


KEY_DELTAS = {
    'w': np.array([1.0, 0.0, 0.0]),
    's': np.array([-1.0, 0.0, 0.0]),
    'a': np.array([0.0, 1.0, 0.0]),
    'd': np.array([0.0, -1.0, 0.0]),
    'r': np.array([0.0, 0.0, 1.0]),
    'f': np.array([0.0, 0.0, -1.0]),
}


class TerminalMode:
    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError('keyboard jog needs an interactive terminal')
        self.settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


class LeRobotIKKeyboardJog(Node):
    """Keyboard-controlled Cartesian jog using official LeRobot kinematics."""

    def __init__(self):
        super().__init__('lerobot_ik_keyboard_jog')

        self.declare_parameter('step_m', 0.02)
        self.declare_parameter('relative_frame', 'base')
        self.declare_parameter('execute', True)
        self.declare_parameter('target_frame', 'gripper_frame_link')
        self.declare_parameter('urdf_path', '')
        self.declare_parameter('arm_joint_names', ','.join(ARM_JOINTS))
        self.declare_parameter('position_weight', 1.0)
        self.declare_parameter('orientation_weight', 0.0)
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('read_joints_service', '/lerobot/read_joints')
        self.declare_parameter('write_joints_service', '/lerobot/write_joints')
        self.declare_parameter('max_joint_delta_deg', 15.0)
        self.declare_parameter('max_position_error_m', 0.003)
        self.declare_parameter('joint_delta_gain', 1.0)
        self.declare_parameter('z_up_joint_delta_gain', 0.0)
        self.declare_parameter('z_down_joint_delta_gain', 0.0)
        self.declare_parameter('min_command_delta_deg', 0.0)
        self.declare_parameter('z_up_min_command_delta_deg', -1.0)
        self.declare_parameter('z_down_min_command_delta_deg', -1.0)
        self.declare_parameter('min_command_joint_names', '')
        self.declare_parameter('z_up_min_command_joint_names', '')
        self.declare_parameter('z_down_min_command_joint_names', '')
        self.declare_parameter('z_up_x_bias_m', 0.0)
        self.declare_parameter('z_up_y_bias_m', 0.0)
        self.declare_parameter('z_down_x_bias_m', 0.0)
        self.declare_parameter('z_down_y_bias_m', 0.0)
        for key in KEY_DELTAS:
            self.declare_parameter(f'{key}_joint_delta_gain', 0.0)
            self.declare_parameter(f'{key}_min_command_delta_deg', -1.0)
            self.declare_parameter(f'{key}_min_command_joint_names', '')
        self.declare_parameter('verify_after_execute', False)
        self.declare_parameter('settle_sec', 0.2)
        self.declare_parameter('timeout_sec', 3.0)
        self.declare_parameter('min_command_interval_sec', 0.15)

        self.joint_state = None
        self.joint_state_count = 0
        self.step_m = float(self.get_parameter('step_m').value)
        self.last_command_time = 0.0
        self.arm_joints = self.parse_arm_joint_names()

        self.create_subscription(
            JointState,
            str(self.get_parameter('joint_state_topic').value),
            self.on_joint_state,
            10,
        )
        self.read_client = self.create_client(ReadJoints, str(self.get_parameter('read_joints_service').value))
        self.write_client = self.create_client(WriteJoints, str(self.get_parameter('write_joints_service').value))
        self.kinematics = self.make_kinematics()
        self.gripper_percent = None

    def on_joint_state(self, msg):
        self.joint_state = msg
        self.joint_state_count += 1

    def wait_for_joint_state(self):
        deadline = time.monotonic() + float(self.get_parameter('timeout_sec').value)
        while rclpy.ok() and self.joint_state is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        return self.joint_state

    def parse_arm_joint_names(self):
        names = [
            name.strip()
            for name in str(self.get_parameter('arm_joint_names').value).split(',')
            if name.strip()
        ]
        if not names:
            raise RuntimeError('arm_joint_names must not be empty')
        unknown = [name for name in names if name not in ARM_JOINTS]
        if unknown:
            raise RuntimeError(f'unknown arm_joint_names: {unknown}; expected subset of {ARM_JOINTS}')
        return names

    def spin_for(self, duration_sec):
        deadline = time.monotonic() + max(0.0, float(duration_sec))
        while rclpy.ok() and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            rclpy.spin_once(self, timeout_sec=min(0.03, max(0.0, remaining)))

    def current_arm_degrees(self, msg):
        positions_by_name = dict(zip(msg.name, msg.position, strict=False))
        missing = [name for name in self.arm_joints if name not in positions_by_name]
        if missing:
            raise RuntimeError(f'/joint_states missing joints: {missing}')
        return np.array([math.degrees(float(positions_by_name[name])) for name in self.arm_joints], dtype=float)

    def read_gripper_percent(self):
        if not self.read_client.wait_for_service(timeout_sec=float(self.get_parameter('timeout_sec').value)):
            raise RuntimeError('/lerobot/read_joints service is not available')

        future = self.read_client.call_async(ReadJoints.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=float(self.get_parameter('timeout_sec').value))
        result = future.result()
        if result is None or not result.success or len(result.positions) < 6:
            raise RuntimeError('failed to read LeRobot joints for gripper position')
        return float(result.positions[5])

    def make_kinematics(self):
        pkg_dir = get_package_share_directory('weed_locator')
        urdf_path = str(self.get_parameter('urdf_path').value).strip()
        if not urdf_path:
            urdf_name = 'so101_no_elbow.urdf' if 'elbow_flex' not in self.arm_joints else 'so101_new_calib.urdf'
            urdf_path = str(Path(pkg_dir) / 'config' / 'SO101' / urdf_name)
        self.get_logger().info(f'IK URDF: {urdf_path}')
        return RobotKinematics(
            urdf_path,
            target_frame_name=str(self.get_parameter('target_frame').value),
            joint_names=self.arm_joints,
        )

    def apply_delta(self, current_pose, delta):
        frame = str(self.get_parameter('relative_frame').value).lower().strip()
        target_pose = current_pose.copy()
        if frame == 'tool':
            target_pose[:3, 3] += current_pose[:3, :3] @ delta
        else:
            target_pose[:3, 3] += delta
        return target_pose

    def solve_target(self, delta, axis_key=None):
        msg = self.wait_for_joint_state()
        if msg is None:
            raise RuntimeError('timed out waiting for /joint_states')

        q_current = self.current_arm_degrees(msg)
        t_current = self.kinematics.forward_kinematics(q_current)
        t_target = self.apply_delta(t_current, delta)
        q_target = self.kinematics.inverse_kinematics(
            q_current,
            t_target,
            position_weight=float(self.get_parameter('position_weight').value),
            orientation_weight=float(self.get_parameter('orientation_weight').value),
        )
        q_command = self.apply_command_compensation(q_current, q_target, delta, axis_key=axis_key)
        t_result = self.kinematics.forward_kinematics(q_target)

        delta_deg = q_target - q_current
        command_delta_deg = q_command - q_current
        pos_error = float(np.linalg.norm(t_result[:3, 3] - t_target[:3, 3]))
        max_delta = float(np.max(np.abs(command_delta_deg)))
        return q_current, q_target, q_command, t_current, t_target, pos_error, max_delta

    def compensation_settings(self, cartesian_delta, axis_key=None):
        gain = float(self.get_parameter('joint_delta_gain').value)
        min_delta = abs(float(self.get_parameter('min_command_delta_deg').value))
        min_joint_names = str(self.get_parameter('min_command_joint_names').value).strip()
        dz = float(cartesian_delta[2])

        if dz > 1e-9:
            z_gain = float(self.get_parameter('z_up_joint_delta_gain').value)
            z_min_delta = float(self.get_parameter('z_up_min_command_delta_deg').value)
            z_min_joint_names = str(self.get_parameter('z_up_min_command_joint_names').value).strip()
            if z_gain > 0.0:
                gain = z_gain
            if z_min_delta >= 0.0:
                min_delta = z_min_delta
            if z_min_joint_names:
                min_joint_names = z_min_joint_names
        elif dz < -1e-9:
            z_gain = float(self.get_parameter('z_down_joint_delta_gain').value)
            z_min_delta = float(self.get_parameter('z_down_min_command_delta_deg').value)
            z_min_joint_names = str(self.get_parameter('z_down_min_command_joint_names').value).strip()
            if z_gain > 0.0:
                gain = z_gain
            if z_min_delta >= 0.0:
                min_delta = z_min_delta
            if z_min_joint_names:
                min_joint_names = z_min_joint_names

        if axis_key:
            key_gain = float(self.get_parameter(f'{axis_key}_joint_delta_gain').value)
            key_min_delta = float(self.get_parameter(f'{axis_key}_min_command_delta_deg').value)
            key_min_joint_names = str(self.get_parameter(f'{axis_key}_min_command_joint_names').value).strip()
            if key_gain > 0.0:
                gain = key_gain
            if key_min_delta >= 0.0:
                min_delta = key_min_delta
            if key_min_joint_names:
                min_joint_names = key_min_joint_names

        return gain, min_delta, min_joint_names

    def apply_command_compensation(self, q_current, q_target, cartesian_delta, axis_key=None):
        delta = q_target - q_current
        gain, min_delta, min_joint_names = self.compensation_settings(cartesian_delta, axis_key=axis_key)
        compensated_delta = delta * gain

        if min_delta > 1e-9:
            if min_joint_names:
                enabled = {name.strip() for name in min_joint_names.split(',') if name.strip()}
            else:
                enabled = set(self.arm_joints)
            for i, value in enumerate(compensated_delta):
                if self.arm_joints[i] not in enabled:
                    continue
                if abs(delta[i]) > 1e-9 and abs(value) < min_delta:
                    compensated_delta[i] = math.copysign(min_delta, value)

        return q_current + compensated_delta

    def verify_executed_pose(self, q_start, q_target, start_pose, target_pose):
        start_count = self.joint_state_count
        self.spin_for(float(self.get_parameter('settle_sec').value))
        if self.joint_state_count == start_count:
            self.get_logger().warning('no fresh /joint_states received after command')
            return

        q_actual = self.current_arm_degrees(self.joint_state)
        t_actual = self.kinematics.forward_kinematics(q_actual)
        actual_delta = t_actual[:3, 3] - start_pose[:3, 3]
        desired_delta = target_pose[:3, 3] - start_pose[:3, 3]
        target_error = float(np.linalg.norm(t_actual[:3, 3] - target_pose[:3, 3]))
        desired_norm_sq = float(np.dot(desired_delta, desired_delta))
        progress = 0.0
        if desired_norm_sq > 1e-12:
            progress = float(np.dot(actual_delta, desired_delta) / desired_norm_sq)

        self.get_logger().info(
            'actual delta xyz mm: '
            f'{np.round(actual_delta * 1000.0, 1).tolist()}, '
            f'target error: {target_error * 1000.0:.1f} mm, '
            f'progress: {progress * 100.0:.0f}%'
        )
        self.get_logger().info(
            f'target joint delta deg {self.arm_joints}: '
            f'{np.round(q_target - q_start, 2).tolist()}'
        )
        self.get_logger().info(
            f'actual joint delta deg {self.arm_joints}: '
            f'{np.round(q_actual - q_start, 2).tolist()}'
        )
        self.get_logger().info(
            f'remaining joint error deg {self.arm_joints}: '
            f'{np.round(q_target - q_actual, 2).tolist()}'
        )

    def send_command(self, q_target):
        if not self.write_client.wait_for_service(timeout_sec=float(self.get_parameter('timeout_sec').value)):
            raise RuntimeError('/lerobot/write_joints service is not available')

        targets = [float(value) for value in q_target]
        targets.append(float(self.gripper_percent))
        bad = [
            (index, value)
            for index, value in enumerate(targets)
            if not math.isfinite(value)
        ]
        if bad:
            raise RuntimeError(f'IK command contains non-finite target_positions: {bad}')

        request = WriteJoints.Request()
        request.target_positions = targets
        future = self.write_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=float(self.get_parameter('timeout_sec').value))
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError('failed to execute /lerobot/write_joints command')

    def jog(self, axis_key):
        now = time.monotonic()
        min_interval = float(self.get_parameter('min_command_interval_sec').value)
        if now - self.last_command_time < min_interval:
            return

        delta = KEY_DELTAS[axis_key] * self.step_m
        if axis_key == 'r':
            delta = delta.copy()
            delta[0] += float(self.get_parameter('z_up_x_bias_m').value)
            delta[1] += float(self.get_parameter('z_up_y_bias_m').value)
        elif axis_key == 'f':
            delta = delta.copy()
            delta[0] += float(self.get_parameter('z_down_x_bias_m').value)
            delta[1] += float(self.get_parameter('z_down_y_bias_m').value)
        try:
            q_current, q_target, q_command, t_current, t_target, pos_error, max_delta = self.solve_target(
                delta,
                axis_key=axis_key,
            )
        except Exception as exc:
            self.get_logger().error(str(exc))
            return

        max_allowed_delta = float(self.get_parameter('max_joint_delta_deg').value)
        max_allowed_error = float(self.get_parameter('max_position_error_m').value)
        if max_delta > max_allowed_delta or pos_error > max_allowed_error:
            self.get_logger().error(
                f'rejected {axis_key}: max_delta={max_delta:.2f} deg '
                f'(limit {max_allowed_delta:.2f}), ik_error={pos_error * 1000.0:.2f} mm '
                f'(limit {max_allowed_error * 1000.0:.2f})'
            )
            return

        if bool(self.get_parameter('execute').value):
            try:
                self.send_command(q_command)
            except Exception as exc:
                self.get_logger().error(str(exc))
                return
            status = 'sent'
        else:
            status = 'dry-run'

        self.last_command_time = time.monotonic()
        self.get_logger().info(
            f'{status} {axis_key}: target delta mm '
            f'{np.round(delta * 1000.0, 1).tolist()}, '
            f'ik_error={pos_error * 1000.0:.2f} mm, max_joint={max_delta:.2f} deg'
        )
        if not np.allclose(q_command, q_target):
            self.get_logger().info(
                f'command delta deg {self.arm_joints}: {np.round(q_command - q_current, 2).tolist()}'
            )
        if bool(self.get_parameter('verify_after_execute').value) and bool(self.get_parameter('execute').value):
            self.verify_executed_pose(q_current, q_command, t_current, t_target)

    def print_help(self):
        execute = bool(self.get_parameter('execute').value)
        frame = str(self.get_parameter('relative_frame').value)
        print('')
        print('LeRobot SO101 IK keyboard jog')
        print(f'  frame: {frame}, step: {self.step_m * 1000.0:.1f} mm, execute: {execute}')
        print(f'  joints: {self.arm_joints}')
        print('  w/s: +x/-x    a/d: +y/-y    r/f: +z/-z')
        print('  +/-: step up/down             q: quit')
        print('')

    def adjust_step(self, scale):
        self.step_m = max(0.001, min(0.05, self.step_m * scale))
        print(f'step: {self.step_m * 1000.0:.1f} mm')

    def initialize(self):
        if self.wait_for_joint_state() is None:
            raise RuntimeError('timed out waiting for /joint_states')
        self.gripper_percent = self.read_gripper_percent()
        if bool(self.get_parameter('execute').value):
            if not self.write_client.wait_for_service(timeout_sec=float(self.get_parameter('timeout_sec').value)):
                raise RuntimeError('/lerobot/write_joints service is not available')

    def run(self):
        self.initialize()
        self.print_help()
        with TerminalMode():
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.02)
                readable, _, _ = select.select([sys.stdin], [], [], 0.02)
                if not readable:
                    continue

                key = sys.stdin.read(1).lower()
                if key == 'q' or key == '\x03':
                    print('quit')
                    return
                if key in ('+', '='):
                    self.adjust_step(1.5)
                    continue
                if key in ('-', '_'):
                    self.adjust_step(1.0 / 1.5)
                    continue
                if key in KEY_DELTAS:
                    self.jog(key)


def main(args=None):
    rclpy.init(args=args)
    node = LeRobotIKKeyboardJog()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        node.get_logger().error(str(exc))
        raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
