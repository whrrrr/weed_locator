#!/home/whr/miniconda3/envs/lerobot/bin/python
"""Empirical keyboard jog for the modified LeRobot SO101 arm.

This node is intentionally not a strict Cartesian controller. Horizontal keys
send tuned joint-space increments, then optionally correct Z drift using the
current kinematic model. It is meant for demo teleoperation on a modified,
heavier SO101 arm with backlash and gravity sag.
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

DEFAULT_MAP = {
    # w/s are approximate +X/-X. Tune on the real arm with +/-.
    'w': [0.0, 2.0, -1.5, -1.0, 0.0],
    's': [0.0, -2.0, 1.5, 1.0, 0.0],
    # a/d are approximate +Y/-Y from previous dy tests.
    'a': [-2.0, 0.0, 0.0, 0.0, 0.0],
    'd': [2.0, 0.0, 0.0, 0.0, 0.0],
}


class TerminalMode:
    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError('empirical keyboard jog needs an interactive terminal')
        self.settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


class LeRobotEmpiricalKeyboardJog(Node):
    """Keyboard teleop with empirical joint deltas and Z hold."""

    def __init__(self):
        super().__init__('lerobot_empirical_keyboard_jog')

        self.declare_parameter('execute', True)
        self.declare_parameter('settle_sec', 1.2)
        self.declare_parameter('timeout_sec', 10.0)
        self.declare_parameter('continuous_mode', False)
        self.declare_parameter('command_period_sec', 0.18)
        self.declare_parameter('key_timeout_sec', 0.28)
        self.declare_parameter('continuous_step_scale', 1.0)
        self.declare_parameter('continuous_z_step_m', 0.006)
        self.declare_parameter('continuous_z_use_empirical', True)
        self.declare_parameter('continuous_z_up_delta_deg', '0.0,0.10,-0.44,-0.38,0.02')
        self.declare_parameter('continuous_z_down_delta_deg', '0.0,-1.24,4.35,3.83,-0.19')
        self.declare_parameter('continuous_z_up_period_multiplier', 1.0)
        self.declare_parameter('continuous_z_down_period_multiplier', 2.0)
        self.declare_parameter('continuous_log_every', 10)
        self.declare_parameter('continuous_verify_motion', False)
        self.declare_parameter('gripper_step_percent', 3.0)
        self.declare_parameter('gripper_min_percent', 0.0)
        self.declare_parameter('gripper_max_percent', 100.0)
        self.declare_parameter('target_frame', 'gripper_frame_link')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('read_joints_service', '/lerobot/read_joints')
        self.declare_parameter('write_joints_service', '/lerobot/write_joints')
        self.declare_parameter('max_joint_delta_deg', 12.0)
        self.declare_parameter('z_step_m', 0.02)
        self.declare_parameter('z_hold', False)
        self.declare_parameter('z_hold_tolerance_m', 0.004)
        self.declare_parameter('z_hold_max_correction_m', 0.02)
        self.declare_parameter('tune_step_deg', 0.25)
        self.declare_parameter('position_weight', 1.0)
        self.declare_parameter('orientation_weight', 0.0)
        self.declare_parameter('z_min_m', 0.10)
        self.declare_parameter('z_max_m', 0.27)
        self.declare_parameter('z_up_joint_delta_gain', 1.6)
        self.declare_parameter('z_down_joint_delta_gain', 0.75)
        self.declare_parameter('z_up_min_command_delta_deg', 3.0)
        self.declare_parameter('z_down_min_command_delta_deg', 0.0)
        self.declare_parameter('z_up_min_command_joint_names', 'elbow_flex')
        self.declare_parameter('z_down_min_command_joint_names', '')
        self.declare_parameter(
            'config_path',
            '/home/whr/cc_ws/tros_ws/calibration_targets/lerobot_empirical_keyboard_jog.yaml',
        )
        self.declare_parameter(
            'log_path',
            '/home/whr/cc_ws/tros_ws/calibration_targets/lerobot_empirical_keyboard_jog_log.yaml',
        )

        self.joint_state = None
        self.joint_state_count = 0
        self.speed_scale = 1.0
        self.last_key = None
        self.gripper_percent = None
        self.continuous_command_count = 0
        self.records = []
        self.joint_map = {key: np.array(value, dtype=float) for key, value in DEFAULT_MAP.items()}

        self.create_subscription(
            JointState,
            str(self.get_parameter('joint_state_topic').value),
            self.on_joint_state,
            10,
        )
        self.read_client = self.create_client(ReadJoints, str(self.get_parameter('read_joints_service').value))
        self.write_client = self.create_client(WriteJoints, str(self.get_parameter('write_joints_service').value))
        self.kinematics = self.make_kinematics()
        self.load_config()

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

    def send_gripper_only(self, gripper_percent):
        if not bool(self.get_parameter('execute').value):
            return
        if not self.write_client.wait_for_service(timeout_sec=float(self.get_parameter('timeout_sec').value)):
            raise RuntimeError('/lerobot/write_joints service is not available')
        request = WriteJoints.Request()
        request.target_positions = [float(gripper_percent)]
        future = self.write_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=float(self.get_parameter('timeout_sec').value))
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError('failed to execute gripper-only /lerobot/write_joints command')

    def load_config(self):
        path = Path(str(self.get_parameter('config_path').value)).expanduser()
        if not path.exists():
            return
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        joint_map = data.get('joint_map', {})
        for key, value in joint_map.items():
            if key in self.joint_map and len(value) == len(ARM_JOINTS):
                self.joint_map[key] = np.array(value, dtype=float)
        self.get_logger().info(f'loaded empirical jog config: {path}')

    def save_config(self):
        path = Path(str(self.get_parameter('config_path').value)).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'joint_names': ARM_JOINTS,
            'joint_map': {key: value.round(4).tolist() for key, value in self.joint_map.items()},
        }
        path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding='utf-8')
        self.get_logger().info(f'saved config: {path}')

    def save_log(self):
        path = Path(str(self.get_parameter('log_path').value)).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'joint_names': ARM_JOINTS,
            'records': self.records,
        }
        path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding='utf-8')
        self.get_logger().info(f'saved log: {path}')

    def z_compensation_settings(self, dz):
        if dz >= 0.0:
            return (
                float(self.get_parameter('z_up_joint_delta_gain').value),
                abs(float(self.get_parameter('z_up_min_command_delta_deg').value)),
                str(self.get_parameter('z_up_min_command_joint_names').value).strip(),
            )
        return (
            float(self.get_parameter('z_down_joint_delta_gain').value),
            abs(float(self.get_parameter('z_down_min_command_delta_deg').value)),
            str(self.get_parameter('z_down_min_command_joint_names').value).strip(),
        )

    def compensated_z_command(self, q_current, q_target, dz):
        delta = q_target - q_current
        gain, min_delta, min_joint_names = self.z_compensation_settings(dz)
        command_delta = delta * gain
        enabled = {name.strip() for name in min_joint_names.split(',') if name.strip()}
        if min_delta > 1e-9:
            for i, value in enumerate(command_delta):
                if ARM_JOINTS[i] not in enabled:
                    continue
                if abs(delta[i]) > 1e-9 and abs(value) < min_delta:
                    command_delta[i] = math.copysign(min_delta, value)
        return q_current + command_delta

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
        return self.compensated_z_command(q_start, q_target, dz)

    def correct_z_if_needed(self, t_start, q_after, t_after, gripper_percent):
        if not self.z_hold_enabled:
            return q_after, t_after, None
        z_error = float(t_start[2, 3] - t_after[2, 3])
        tolerance = float(self.get_parameter('z_hold_tolerance_m').value)
        if abs(z_error) <= tolerance:
            return q_after, t_after, None
        max_correction = abs(float(self.get_parameter('z_hold_max_correction_m').value))
        correction = max(-max_correction, min(max_correction, z_error))
        try:
            q_correction = self.solve_z_command(q_after, t_after, correction)
        except Exception as exc:
            self.get_logger().warning(f'z-hold skipped: {exc}')
            return q_after, t_after, None
        self.get_logger().info(f'z-hold correction dz={correction * 1000.0:.1f} mm')
        self.send_joints(q_correction, gripper_percent)
        self.spin_for(float(self.get_parameter('settle_sec').value))
        q_final, t_final = self.current_pose()
        return q_final, t_final, {
            'requested_dz_m': correction,
            'after_xyz_m': t_final[:3, 3].round(6).tolist(),
            'actual_delta_xyz_m': (t_final[:3, 3] - t_after[:3, 3]).round(6).tolist(),
            'command_delta_joints_deg': (q_correction - q_after).round(4).tolist(),
            'actual_delta_joints_deg': (q_final - q_after).round(4).tolist(),
        }

    def move_joint_delta(self, key):
        if key not in self.joint_map:
            return
        self.last_key = key
        q_start, t_start = self.current_pose()
        command_delta = self.joint_map[key] * self.speed_scale
        max_delta = float(np.max(np.abs(command_delta)))
        max_allowed = float(self.get_parameter('max_joint_delta_deg').value)
        if max_delta > max_allowed:
            self.get_logger().warning(f'rejected {key}: max joint delta {max_delta:.2f} > {max_allowed:.2f} deg')
            return
        gripper_percent = self.read_gripper_percent()
        q_command = q_start + command_delta
        self.get_logger().info(f'{key}: command_delta={np.round(command_delta, 2).tolist()}, scale={self.speed_scale:.2f}')
        self.send_joints(q_command, gripper_percent)
        self.spin_for(float(self.get_parameter('settle_sec').value))
        q_after, t_after = self.current_pose()
        q_final, t_final, z_hold_record = self.correct_z_if_needed(t_start, q_after, t_after, gripper_percent)

        actual_delta = t_final[:3, 3] - t_start[:3, 3]
        self.get_logger().info(
            f'{key}: actual_delta_mm={np.round(actual_delta * 1000.0, 1).tolist()}, '
            f'z_error={(t_final[2, 3] - t_start[2, 3]) * 1000.0:+.1f} mm'
        )
        record = {
            'key': key,
            'speed_scale': self.speed_scale,
            'start_xyz_m': t_start[:3, 3].round(6).tolist(),
            'after_xyz_m': t_final[:3, 3].round(6).tolist(),
            'actual_delta_xyz_m': actual_delta.round(6).tolist(),
            'start_joints_deg': q_start.round(4).tolist(),
            'command_delta_joints_deg': command_delta.round(4).tolist(),
            'after_joints_deg': q_final.round(4).tolist(),
            'actual_delta_joints_deg': (q_final - q_start).round(4).tolist(),
        }
        if z_hold_record is not None:
            record['z_hold'] = z_hold_record
        self.records.append(record)
        self.save_log()

    def move_z(self, dz, key):
        q_start, t_start = self.current_pose()
        try:
            q_command = self.solve_z_command(q_start, t_start, dz)
        except Exception as exc:
            self.get_logger().warning(f'rejected {key}: {exc}')
            return
        max_delta = float(np.max(np.abs(q_command - q_start)))
        max_allowed = float(self.get_parameter('max_joint_delta_deg').value)
        if max_delta > max_allowed:
            self.get_logger().warning(f'rejected {key}: max joint delta {max_delta:.2f} > {max_allowed:.2f} deg')
            return
        gripper_percent = self.read_gripper_percent()
        self.get_logger().info(f'{key}: z dz={dz * 1000.0:.1f} mm, cmd_delta={np.round(q_command - q_start, 2).tolist()}')
        self.send_joints(q_command, gripper_percent)
        self.spin_for(float(self.get_parameter('settle_sec').value))
        q_after, t_after = self.current_pose()
        actual_delta = t_after[:3, 3] - t_start[:3, 3]
        self.get_logger().info(f'{key}: actual_delta_mm={np.round(actual_delta * 1000.0, 1).tolist()}')
        self.records.append({
            'key': key,
            'requested_dz_m': dz,
            'start_xyz_m': t_start[:3, 3].round(6).tolist(),
            'after_xyz_m': t_after[:3, 3].round(6).tolist(),
            'actual_delta_xyz_m': actual_delta.round(6).tolist(),
            'start_joints_deg': q_start.round(4).tolist(),
            'command_delta_joints_deg': (q_command - q_start).round(4).tolist(),
            'after_joints_deg': q_after.round(4).tolist(),
            'actual_delta_joints_deg': (q_after - q_start).round(4).tolist(),
        })
        self.save_log()

    def adjust_last_key(self, factor):
        if not self.last_key or self.last_key not in self.joint_map:
            self.get_logger().warning('press w/s/a/d first, then tune with +/-')
            return
        self.joint_map[self.last_key] *= factor
        self.get_logger().info(f'{self.last_key} map -> {np.round(self.joint_map[self.last_key], 3).tolist()}')

    def adjust_last_key_joint(self, joint_name, delta_deg):
        if not self.last_key or self.last_key not in self.joint_map:
            self.get_logger().warning('press w/s/a/d first, then tune joint terms')
            return
        index = ARM_JOINTS.index(joint_name)
        self.joint_map[self.last_key][index] += float(delta_deg)
        self.get_logger().info(
            f'{self.last_key} {joint_name} {delta_deg:+.2f} deg -> '
            f'{np.round(self.joint_map[self.last_key], 3).tolist()}'
        )

    def toggle_z_hold(self):
        self.z_hold_enabled = not self.z_hold_enabled
        self.get_logger().info(f'z_hold = {self.z_hold_enabled}')

    def read_key(self):
        if not select.select([sys.stdin], [], [], 0.05)[0]:
            rclpy.spin_once(self, timeout_sec=0.01)
            return None
        return sys.stdin.read(1)

    def read_available_keys(self):
        keys = []
        while select.select([sys.stdin], [], [], 0.0)[0]:
            keys.append(sys.stdin.read(1))
        if not keys:
            rclpy.spin_once(self, timeout_sec=0.01)
        return keys

    def cached_gripper_percent(self):
        if self.gripper_percent is None:
            self.gripper_percent = self.read_gripper_percent()
        return self.gripper_percent

    def clamp_gripper_percent(self, value):
        low = float(self.get_parameter('gripper_min_percent').value)
        high = float(self.get_parameter('gripper_max_percent').value)
        return max(low, min(high, float(value)))

    def move_gripper(self, delta_percent):
        current = self.cached_gripper_percent()
        target = self.clamp_gripper_percent(current + float(delta_percent))
        self.send_gripper_only(target)
        self.gripper_percent = target
        self.get_logger().info(f'gripper: {current:.1f}% -> {target:.1f}%')

    def send_joint_delta_fast(self, key):
        if key not in self.joint_map:
            return
        q_start = self.current_arm_degrees()
        command_delta = (
            self.joint_map[key]
            * self.speed_scale
            * float(self.get_parameter('continuous_step_scale').value)
        )
        max_delta = float(np.max(np.abs(command_delta)))
        max_allowed = float(self.get_parameter('max_joint_delta_deg').value)
        if max_delta > max_allowed:
            self.get_logger().warning(f'rejected {key}: max joint delta {max_delta:.2f} > {max_allowed:.2f} deg')
            return
        self.send_joints(q_start + command_delta, self.cached_gripper_percent())
        if bool(self.get_parameter('continuous_verify_motion').value):
            self.spin_for(0.18)
            q_after = self.current_arm_degrees()
            self.get_logger().info(
                f'continuous {key}: cmd={np.round(command_delta, 2).tolist()}, '
                f'actual_joints={np.round(q_after - q_start, 2).tolist()}'
            )
        self.continuous_command_count += 1
        log_every = max(1, int(self.get_parameter('continuous_log_every').value))
        if self.continuous_command_count % log_every == 0:
            self.get_logger().info(f'continuous {key}: delta={np.round(command_delta, 2).tolist()}')

    def send_z_fast(self, dz, key):
        if bool(self.get_parameter('continuous_z_use_empirical').value):
            q_start = self.current_arm_degrees()
            param_name = 'continuous_z_up_delta_deg' if dz >= 0.0 else 'continuous_z_down_delta_deg'
            command_delta = self.parse_joint_delta_param(param_name)
            max_delta = float(np.max(np.abs(command_delta)))
            max_allowed = float(self.get_parameter('max_joint_delta_deg').value)
            if max_delta > max_allowed:
                self.get_logger().warning(f'rejected {key}: max joint delta {max_delta:.2f} > {max_allowed:.2f} deg')
                return
            self.send_joints(q_start + command_delta, self.cached_gripper_percent())
            if bool(self.get_parameter('continuous_verify_motion').value):
                self.spin_for(0.18)
                q_after = self.current_arm_degrees()
                self.get_logger().info(
                    f'continuous {key}: cmd={np.round(command_delta, 2).tolist()}, '
                    f'actual_joints={np.round(q_after - q_start, 2).tolist()}'
                )
            self.continuous_command_count += 1
            log_every = max(1, int(self.get_parameter('continuous_log_every').value))
            if self.continuous_command_count % log_every == 0:
                self.get_logger().info(f'continuous {key}: empirical_delta={np.round(command_delta, 2).tolist()}')
            return

        q_start, t_start = self.current_pose()
        try:
            q_command = self.solve_z_command(q_start, t_start, dz)
        except Exception as exc:
            self.get_logger().warning(f'rejected {key}: {exc}')
            return
        max_delta = float(np.max(np.abs(q_command - q_start)))
        max_allowed = float(self.get_parameter('max_joint_delta_deg').value)
        if max_delta > max_allowed:
            self.get_logger().warning(f'rejected {key}: max joint delta {max_delta:.2f} > {max_allowed:.2f} deg')
            return
        self.send_joints(q_command, self.cached_gripper_percent())
        self.continuous_command_count += 1
        log_every = max(1, int(self.get_parameter('continuous_log_every').value))
        if self.continuous_command_count % log_every == 0:
            self.get_logger().info(f'continuous {key}: dz={dz * 1000.0:.1f} mm')

    def parse_joint_delta_param(self, param_name):
        text = str(self.get_parameter(param_name).value)
        parts = [part.strip() for part in text.split(',') if part.strip()]
        if len(parts) != len(ARM_JOINTS):
            raise RuntimeError(f'{param_name} must contain {len(ARM_JOINTS)} comma-separated values')
        return np.array([float(part) for part in parts], dtype=float)

    def handle_non_motion_key(self, key):
        if key == '+':
            self.adjust_last_key(1.15)
        elif key == '-':
            self.adjust_last_key(1.0 / 1.15)
        elif key == ',':
            self.adjust_last_key_joint('shoulder_lift', -float(self.get_parameter('tune_step_deg').value))
        elif key == '.':
            self.adjust_last_key_joint('shoulder_lift', float(self.get_parameter('tune_step_deg').value))
        elif key == '[':
            self.adjust_last_key_joint('elbow_flex', -float(self.get_parameter('tune_step_deg').value))
        elif key == ']':
            self.adjust_last_key_joint('elbow_flex', float(self.get_parameter('tune_step_deg').value))
        elif key == ';':
            self.adjust_last_key_joint('wrist_flex', -float(self.get_parameter('tune_step_deg').value))
        elif key == "'":
            self.adjust_last_key_joint('wrist_flex', float(self.get_parameter('tune_step_deg').value))
        elif key == 'h':
            self.toggle_z_hold()
        elif key == '1':
            self.speed_scale = 0.5
            self.get_logger().info('speed scale = 0.5')
        elif key == '2':
            self.speed_scale = 1.0
            self.get_logger().info('speed scale = 1.0')
        elif key == '3':
            self.speed_scale = 1.5
            self.get_logger().info('speed scale = 1.5')
        elif key == 'p':
            self.save_config()

    def run_continuous(self):
        active_key = None
        last_key_time = 0.0
        next_command_time = 0.0
        command_period = max(0.05, float(self.get_parameter('command_period_sec').value))
        key_timeout = max(command_period, float(self.get_parameter('key_timeout_sec').value))
        print('  continuous mode: hold w/s/a/d/r/f to move, release to stop\n')
        with TerminalMode():
            while rclpy.ok():
                now = time.monotonic()
                for key in self.read_available_keys():
                    if key in ('q', '\x03'):
                        return
                    if key in self.joint_map or key in ('r', 'f', 'o', 'c'):
                        active_key = key
                        self.last_key = key if key in self.joint_map else self.last_key
                        last_key_time = now
                    else:
                        self.handle_non_motion_key(key)

                if active_key is not None and now - last_key_time > key_timeout:
                    active_key = None

                if active_key is not None and now >= next_command_time:
                    period_multiplier = 1.0
                    if active_key in self.joint_map:
                        self.send_joint_delta_fast(active_key)
                    elif active_key == 'r':
                        self.send_z_fast(abs(float(self.get_parameter('continuous_z_step_m').value)), active_key)
                        period_multiplier = max(1.0, float(self.get_parameter('continuous_z_up_period_multiplier').value))
                    elif active_key == 'f':
                        self.send_z_fast(-abs(float(self.get_parameter('continuous_z_step_m').value)), active_key)
                        period_multiplier = max(1.0, float(self.get_parameter('continuous_z_down_period_multiplier').value))
                    elif active_key == 'o':
                        self.move_gripper(abs(float(self.get_parameter('gripper_step_percent').value)))
                    elif active_key == 'c':
                        self.move_gripper(-abs(float(self.get_parameter('gripper_step_percent').value)))
                    next_command_time = time.monotonic() + command_period * period_multiplier

    def run(self):
        self.wait_for_joint_state()
        self.z_hold_enabled = bool(self.get_parameter('z_hold').value)
        self.gripper_percent = self.read_gripper_percent()
        print('\nLeRobot empirical keyboard jog')
        print('  w/s: approx +X/-X    a/d: approx +Y/-Y')
        print('  r/f: +Z/-Z           o/c: gripper open/close')
        print('  +/-: tune last horizontal key')
        print('  ,/.: shoulder trim   [/]: elbow trim   ;/\': wrist trim')
        print('  h: toggle Z hold')
        print('  1/2/3: slow/normal/fast    p: save config    q: quit\n')
        if bool(self.get_parameter('continuous_mode').value):
            self.run_continuous()
            self.save_config()
            return
        with TerminalMode():
            while rclpy.ok():
                key = self.read_key()
                if key is None:
                    continue
                if key in ('q', '\x03'):
                    break
                if key in self.joint_map:
                    self.move_joint_delta(key)
                elif key == 'r':
                    self.move_z(abs(float(self.get_parameter('z_step_m').value)), 'r')
                elif key == 'f':
                    self.move_z(-abs(float(self.get_parameter('z_step_m').value)), 'f')
                elif key == 'o':
                    self.move_gripper(abs(float(self.get_parameter('gripper_step_percent').value)))
                elif key == 'c':
                    self.move_gripper(-abs(float(self.get_parameter('gripper_step_percent').value)))
                elif key == '+':
                    self.handle_non_motion_key(key)
                elif key == '-':
                    self.handle_non_motion_key(key)
                elif key == ',':
                    self.handle_non_motion_key(key)
                elif key == '.':
                    self.handle_non_motion_key(key)
                elif key == '[':
                    self.handle_non_motion_key(key)
                elif key == ']':
                    self.handle_non_motion_key(key)
                elif key == ';':
                    self.handle_non_motion_key(key)
                elif key == "'":
                    self.handle_non_motion_key(key)
                elif key == 'h':
                    self.handle_non_motion_key(key)
                elif key == '1':
                    self.handle_non_motion_key(key)
                elif key == '2':
                    self.handle_non_motion_key(key)
                elif key == '3':
                    self.handle_non_motion_key(key)
                elif key == 'p':
                    self.handle_non_motion_key(key)
        self.save_config()
        self.save_log()


def main(args=None):
    rclpy.init(args=args)
    node = LeRobotEmpiricalKeyboardJog()
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
