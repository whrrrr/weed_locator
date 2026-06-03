#!/home/whr/miniconda3/envs/lerobot/bin/python
"""Interactive Z jog scanner for the modified LeRobot SO101 arm."""

import math
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
import rclpy
import yaml
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


class TerminalMode:
    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError('keyboard scan needs an interactive terminal')
        self.settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


class LeRobotZKeyboardScan(Node):
    """Press keys to move Z and append response records to YAML."""

    def __init__(self):
        super().__init__('lerobot_z_keyboard_scan')

        self.declare_parameter('execute', True)
        self.declare_parameter('step_m', 0.02)
        self.declare_parameter('settle_sec', 2.0)
        self.declare_parameter('timeout_sec', 10.0)
        self.declare_parameter('target_frame', 'gripper_frame_link')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('read_joints_service', '/lerobot/read_joints')
        self.declare_parameter('write_joints_service', '/lerobot/write_joints')
        self.declare_parameter('max_joint_delta_deg', 14.0)
        self.declare_parameter('position_weight', 1.0)
        self.declare_parameter('orientation_weight', 0.0)
        self.declare_parameter('z_min_m', 0.11)
        self.declare_parameter('z_max_m', 0.26)
        self.declare_parameter('z_up_joint_delta_gain', 1.6)
        self.declare_parameter('z_down_joint_delta_gain', 0.75)
        self.declare_parameter('z_up_min_command_delta_deg', 3.0)
        self.declare_parameter('z_down_min_command_delta_deg', 0.0)
        self.declare_parameter('z_up_preload_m', 0.0)
        self.declare_parameter('min_command_joint_names', 'elbow_flex,wrist_flex')
        self.declare_parameter('z_up_min_command_joint_names', 'elbow_flex')
        self.declare_parameter('z_down_min_command_joint_names', '')
        self.declare_parameter(
            'save_path',
            '/home/whr/cc_ws/tros_ws/calibration_targets/lerobot_z_keyboard_scan.yaml',
        )

        self.joint_state = None
        self.joint_state_count = 0
        self.records = []
        self.start_time = time.time()

        self.create_subscription(
            JointState,
            str(self.get_parameter('joint_state_topic').value),
            self.on_joint_state,
            10,
        )
        self.read_client = self.create_client(ReadJoints, str(self.get_parameter('read_joints_service').value))
        self.write_client = self.create_client(WriteJoints, str(self.get_parameter('write_joints_service').value))
        self.kinematics = self.make_kinematics()

    def on_joint_state(self, msg):
        self.joint_state = msg
        self.joint_state_count += 1

    def make_kinematics(self):
        pkg_dir = get_package_share_directory('weed_locator')
        urdf_path = str(Path(pkg_dir) / 'config' / 'SO101' / 'so101_new_calib.urdf')
        return RobotKinematics(
            urdf_path,
            target_frame_name=str(self.get_parameter('target_frame').value),
            joint_names=ARM_JOINTS,
        )

    def wait_for_joint_state(self):
        deadline = time.monotonic() + float(self.get_parameter('timeout_sec').value)
        while rclpy.ok() and self.joint_state is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.joint_state is None:
            raise RuntimeError('timed out waiting for /joint_states')
        return self.joint_state

    def spin_for(self, duration_sec):
        deadline = time.monotonic() + max(0.0, float(duration_sec))
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def current_arm_degrees(self):
        msg = self.wait_for_joint_state()
        positions_by_name = dict(zip(msg.name, msg.position, strict=False))
        missing = [name for name in ARM_JOINTS if name not in positions_by_name]
        if missing:
            raise RuntimeError(f'/joint_states missing joints: {missing}')
        return np.array([math.degrees(float(positions_by_name[name])) for name in ARM_JOINTS], dtype=float)

    def current_pose(self):
        q = self.current_arm_degrees()
        return q, self.kinematics.forward_kinematics(q)

    def read_gripper_percent(self):
        if not self.read_client.wait_for_service(timeout_sec=float(self.get_parameter('timeout_sec').value)):
            raise RuntimeError('/lerobot/read_joints service is not available')
        future = self.read_client.call_async(ReadJoints.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=float(self.get_parameter('timeout_sec').value))
        result = future.result()
        if result is None or not result.success or len(result.positions) < 6:
            raise RuntimeError('failed to read LeRobot joints for gripper position')
        return float(result.positions[5])

    def compensation_settings(self, dz):
        if dz >= 0.0:
            return (
                float(self.get_parameter('z_up_joint_delta_gain').value),
                abs(float(self.get_parameter('z_up_min_command_delta_deg').value)),
                str(self.get_parameter('z_up_min_command_joint_names').value).strip()
                or str(self.get_parameter('min_command_joint_names').value).strip(),
            )
        return (
            float(self.get_parameter('z_down_joint_delta_gain').value),
            abs(float(self.get_parameter('z_down_min_command_delta_deg').value)),
            str(self.get_parameter('z_down_min_command_joint_names').value).strip()
            or str(self.get_parameter('min_command_joint_names').value).strip(),
        )

    def compensated_command(self, q_current, q_target, dz):
        delta = q_target - q_current
        gain, min_delta, min_joint_names = self.compensation_settings(dz)
        command_delta = delta * gain
        enabled = {
            name.strip()
            for name in min_joint_names.split(',')
            if name.strip()
        }
        if min_delta > 1e-9:
            for i, value in enumerate(command_delta):
                if ARM_JOINTS[i] not in enabled:
                    continue
                if abs(delta[i]) > 1e-9 and abs(value) < min_delta:
                    command_delta[i] = math.copysign(min_delta, value)
        return q_current + command_delta

    def send_joints(self, q_command, gripper_percent):
        if not bool(self.get_parameter('execute').value):
            return
        if not self.write_client.wait_for_service(timeout_sec=float(self.get_parameter('timeout_sec').value)):
            raise RuntimeError('/lerobot/write_joints service is not available')
        request = WriteJoints.Request()
        request.target_positions = q_command.tolist() + [gripper_percent]
        future = self.write_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=float(self.get_parameter('timeout_sec').value))
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError('failed to execute /lerobot/write_joints command')

    def save_records(self):
        save_path = Path(str(self.get_parameter('save_path').value)).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'created_unix_sec': self.start_time,
            'updated_unix_sec': time.time(),
            'execute': bool(self.get_parameter('execute').value),
            'parameters': {
                'step_m': float(self.get_parameter('step_m').value),
                'z_min_m': float(self.get_parameter('z_min_m').value),
                'z_max_m': float(self.get_parameter('z_max_m').value),
                'z_up_joint_delta_gain': float(self.get_parameter('z_up_joint_delta_gain').value),
                'z_down_joint_delta_gain': float(self.get_parameter('z_down_joint_delta_gain').value),
                'z_up_min_command_delta_deg': float(self.get_parameter('z_up_min_command_delta_deg').value),
                'z_down_min_command_delta_deg': float(self.get_parameter('z_down_min_command_delta_deg').value),
                'z_up_preload_m': float(self.get_parameter('z_up_preload_m').value),
                'min_command_joint_names': str(self.get_parameter('min_command_joint_names').value),
                'z_up_min_command_joint_names': str(self.get_parameter('z_up_min_command_joint_names').value),
                'z_down_min_command_joint_names': str(self.get_parameter('z_down_min_command_joint_names').value),
            },
            'records': self.records,
        }
        save_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding='utf-8')
        self.get_logger().info(f'saved {len(self.records)} records: {save_path}')

    def solve_z_command(self, q_start, t_start, dz):
        t_target = t_start.copy()
        t_target[2, 3] += dz
        z_min = float(self.get_parameter('z_min_m').value)
        z_max = float(self.get_parameter('z_max_m').value)
        if t_target[2, 3] < z_min or t_target[2, 3] > z_max:
            raise RuntimeError(f'target z={t_target[2, 3]:.4f} outside [{z_min:.4f}, {z_max:.4f}]')

        q_target = self.kinematics.inverse_kinematics(
            q_start,
            t_target,
            position_weight=float(self.get_parameter('position_weight').value),
            orientation_weight=float(self.get_parameter('orientation_weight').value),
        )
        q_command = self.compensated_command(q_start, q_target, dz)
        t_result = self.kinematics.forward_kinematics(q_target)
        target_delta = q_target - q_start
        command_delta = q_command - q_start
        max_delta = float(np.max(np.abs(command_delta)))
        max_allowed = float(self.get_parameter('max_joint_delta_deg').value)
        if max_delta > max_allowed:
            raise RuntimeError(f'max joint delta {max_delta:.2f} > {max_allowed:.2f} deg')
        return q_target, q_command, t_target, t_result

    def jog_z(self, dz, key_name):
        q_start, t_start = self.current_pose()
        preload_record = None
        command_start_q = q_start
        command_start_t = t_start
        command_dz = dz

        preload = max(0.0, float(self.get_parameter('z_up_preload_m').value))
        if dz > 0.0 and preload > 1e-9:
            try:
                _, q_preload_command, t_preload_target, _ = self.solve_z_command(q_start, t_start, -preload)
            except Exception as exc:
                self.get_logger().warning(f'rejected {key_name} preload: {exc}')
                return

            gripper_percent = self.read_gripper_percent()
            self.get_logger().info(
                f'{key_name}: preload {-preload * 1000.0:.1f} mm, '
                f'z {t_start[2, 3]:.4f}->{t_preload_target[2, 3]:.4f}'
            )
            self.send_joints(q_preload_command, gripper_percent)
            self.spin_for(float(self.get_parameter('settle_sec').value))
            q_preload_after, t_preload_after = self.current_pose()
            preload_record = {
                'preload_requested_dz_m': -preload,
                'preload_target_xyz_m': t_preload_target[:3, 3].round(6).tolist(),
                'preload_after_xyz_m': t_preload_after[:3, 3].round(6).tolist(),
                'preload_actual_delta_xyz_m': (t_preload_after[:3, 3] - t_start[:3, 3]).round(6).tolist(),
                'preload_command_delta_joints_deg': (q_preload_command - q_start).round(4).tolist(),
                'preload_actual_delta_joints_deg': (q_preload_after - q_start).round(4).tolist(),
            }
            command_start_q = q_preload_after
            command_start_t = t_preload_after
            command_dz = dz + preload

        try:
            q_target, q_command, t_target, t_result = self.solve_z_command(command_start_q, command_start_t, command_dz)
        except Exception as exc:
            self.get_logger().warning(f'rejected {key_name}: {exc}')
            return
        target_delta = q_target - command_start_q
        command_delta = q_command - command_start_q

        gripper_percent = self.read_gripper_percent()
        self.get_logger().info(
            f'{key_name}: target net dz={dz * 1000.0:.1f} mm, '
            f'z {t_start[2, 3]:.4f}->{t_target[2, 3]:.4f}, '
            f'cmd_delta={np.round(command_delta, 2).tolist()}'
        )
        self.send_joints(q_command, gripper_percent)
        self.spin_for(float(self.get_parameter('settle_sec').value))
        q_after, t_after = self.current_pose()

        actual_delta = t_after[:3, 3] - t_start[:3, 3]
        desired_delta = t_target[:3, 3] - t_start[:3, 3]
        target_error = float(np.linalg.norm(t_after[:3, 3] - t_target[:3, 3]))
        record = {
            'index': len(self.records) + 1,
            'key': key_name,
            'requested_dz_m': float(dz),
            'command_dz_m': float(command_dz),
            'start_xyz_m': t_start[:3, 3].round(6).tolist(),
            'target_xyz_m': t_target[:3, 3].round(6).tolist(),
            'after_xyz_m': t_after[:3, 3].round(6).tolist(),
            'desired_delta_xyz_m': desired_delta.round(6).tolist(),
            'actual_delta_xyz_m': actual_delta.round(6).tolist(),
            'target_error_m': target_error,
            'ik_error_m': float(np.linalg.norm(t_result[:3, 3] - t_target[:3, 3])),
            'start_joints_deg': q_start.round(4).tolist(),
            'target_joints_deg': q_target.round(4).tolist(),
            'command_joints_deg': q_command.round(4).tolist(),
            'after_joints_deg': q_after.round(4).tolist(),
            'target_delta_joints_deg': target_delta.round(4).tolist(),
            'command_delta_joints_deg': command_delta.round(4).tolist(),
            'actual_delta_joints_deg': (q_after - q_start).round(4).tolist(),
        }
        if preload_record is not None:
            record.update(preload_record)
        self.records.append(record)
        self.get_logger().info(
            f'{key_name}: actual_delta_mm={np.round(actual_delta * 1000.0, 1).tolist()}, '
            f'error={target_error * 1000.0:.1f} mm'
        )
        self.save_records()

    def read_key(self):
        if not select.select([sys.stdin], [], [], 0.05)[0]:
            rclpy.spin_once(self, timeout_sec=0.01)
            return None
        ch = sys.stdin.read(1)
        if ch == '\x1b' and select.select([sys.stdin], [], [], 0.01)[0]:
            rest = sys.stdin.read(2)
            if rest == '[A':
                return 'up'
            if rest == '[B':
                return 'down'
        return ch

    def run(self):
        step = abs(float(self.get_parameter('step_m').value))
        self.wait_for_joint_state()
        print('\nLeRobot Z keyboard scan')
        print(f'  step: {step * 1000.0:.1f} mm, execute: {bool(self.get_parameter("execute").value)}')
        print('  Up/r: +Z    Down/f: -Z    s: save    q: quit\n')
        with TerminalMode():
            while rclpy.ok():
                key = self.read_key()
                if key is None:
                    continue
                if key in ('q', '\x03'):
                    break
                if key == 's':
                    self.save_records()
                elif key in ('up', 'r'):
                    self.jog_z(step, 'up')
                elif key in ('down', 'f'):
                    self.jog_z(-step, 'down')
        self.save_records()


def main(args=None):
    rclpy.init(args=args)
    node = LeRobotZKeyboardScan()
    try:
        node.run()
    except Exception as exc:
        node.get_logger().error(str(exc))
        raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
