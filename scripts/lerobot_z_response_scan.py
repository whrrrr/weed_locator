#!/home/whr/miniconda3/envs/lerobot/bin/python
"""Automatic Z response scan for the modified LeRobot SO101 arm."""

import math
import sys
import time
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


class LeRobotZResponseScan(Node):
    """Scan Z up/down response and save actual motion data."""

    def __init__(self):
        super().__init__('lerobot_z_response_scan')

        self.declare_parameter('execute', False)
        self.declare_parameter('levels', 4)
        self.declare_parameter('level_step_m', 0.02)
        self.declare_parameter('test_dz_m', 0.02)
        self.declare_parameter('auto_step_down', False)
        self.declare_parameter('z_min_m', 0.11)
        self.declare_parameter('z_max_m', 0.26)
        self.declare_parameter('settle_sec', 2.0)
        self.declare_parameter('timeout_sec', 10.0)
        self.declare_parameter('target_frame', 'gripper_frame_link')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('read_joints_service', '/lerobot/read_joints')
        self.declare_parameter('write_joints_service', '/lerobot/write_joints')
        self.declare_parameter('max_joint_delta_deg', 14.0)
        self.declare_parameter('position_weight', 1.0)
        self.declare_parameter('orientation_weight', 0.0)

        self.declare_parameter('z_up_joint_delta_gain', 1.6)
        self.declare_parameter('z_down_joint_delta_gain', 0.75)
        self.declare_parameter('z_up_min_command_delta_deg', 3.0)
        self.declare_parameter('z_down_min_command_delta_deg', 0.0)
        self.declare_parameter('min_command_joint_names', 'elbow_flex,wrist_flex')
        self.declare_parameter('z_up_min_command_joint_names', 'elbow_flex')
        self.declare_parameter('z_down_min_command_joint_names', '')

        self.declare_parameter(
            'save_path',
            '/home/whr/cc_ws/tros_ws/calibration_targets/lerobot_z_response_scan.yaml',
        )

        self.joint_state = None
        self.joint_state_count = 0
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
            remaining = deadline - time.monotonic()
            rclpy.spin_once(self, timeout_sec=min(0.05, max(0.0, remaining)))

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

    def make_kinematics(self):
        pkg_dir = get_package_share_directory('weed_locator')
        urdf_path = str(Path(pkg_dir) / 'config' / 'SO101' / 'so101_new_calib.urdf')
        return RobotKinematics(
            urdf_path,
            target_frame_name=str(self.get_parameter('target_frame').value),
            joint_names=ARM_JOINTS,
        )

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

    def check_z_limits(self, label, target_z):
        z_min = float(self.get_parameter('z_min_m').value)
        z_max = float(self.get_parameter('z_max_m').value)
        if target_z < z_min or target_z > z_max:
            raise RuntimeError(
                f'{label} rejected: target z={target_z:.4f} m outside '
                f'[{z_min:.4f}, {z_max:.4f}] m'
            )

    def move_relative_z_once(self, dz, label, gripper_percent):
        q_start, t_start = self.current_pose()
        t_target = t_start.copy()
        t_target[2, 3] += dz
        self.check_z_limits(label, float(t_target[2, 3]))
        q_target = self.kinematics.inverse_kinematics(
            q_start,
            t_target,
            position_weight=float(self.get_parameter('position_weight').value),
            orientation_weight=float(self.get_parameter('orientation_weight').value),
        )
        q_command = self.compensated_command(q_start, q_target, dz)
        t_result = self.kinematics.forward_kinematics(q_target)

        ik_error = float(np.linalg.norm(t_result[:3, 3] - t_target[:3, 3]))
        target_delta = q_target - q_start
        command_delta = q_command - q_start
        max_command_delta = float(np.max(np.abs(command_delta)))
        max_allowed = float(self.get_parameter('max_joint_delta_deg').value)
        if max_command_delta > max_allowed:
            raise RuntimeError(
                f'{label} rejected: command max joint delta {max_command_delta:.2f} deg '
                f'exceeds {max_allowed:.2f} deg'
            )

        self.get_logger().info(
            f'{label}: z={t_start[2, 3]:.4f} -> target {t_target[2, 3]:.4f}, '
            f'command_delta={np.round(command_delta, 2).tolist()}'
        )
        if not bool(self.get_parameter('execute').value):
            return {
                'label': label,
                'dry_run': True,
                'requested_dz_m': float(dz),
                'start_xyz_m': t_start[:3, 3].round(6).tolist(),
                'target_xyz_m': t_target[:3, 3].round(6).tolist(),
                'ik_error_m': ik_error,
                'start_joints_deg': q_start.round(4).tolist(),
                'target_joints_deg': q_target.round(4).tolist(),
                'command_joints_deg': q_command.round(4).tolist(),
                'target_delta_joints_deg': target_delta.round(4).tolist(),
                'command_delta_joints_deg': command_delta.round(4).tolist(),
            }

        self.send_joints(q_command, gripper_percent)
        self.spin_for(float(self.get_parameter('settle_sec').value))
        q_after, t_after = self.current_pose()

        actual_delta = t_after[:3, 3] - t_start[:3, 3]
        desired_delta = t_target[:3, 3] - t_start[:3, 3]
        target_error = float(np.linalg.norm(t_after[:3, 3] - t_target[:3, 3]))
        self.get_logger().info(
            f'{label}: actual_delta_mm={np.round(actual_delta * 1000.0, 1).tolist()}, '
            f'error={target_error * 1000.0:.1f} mm'
        )

        return {
            'label': label,
            'requested_dz_m': float(dz),
            'start_xyz_m': t_start[:3, 3].round(6).tolist(),
            'target_xyz_m': t_target[:3, 3].round(6).tolist(),
            'after_xyz_m': t_after[:3, 3].round(6).tolist(),
            'desired_delta_xyz_m': desired_delta.round(6).tolist(),
            'actual_delta_xyz_m': actual_delta.round(6).tolist(),
            'target_error_m': target_error,
            'ik_error_m': ik_error,
            'start_joints_deg': q_start.round(4).tolist(),
            'target_joints_deg': q_target.round(4).tolist(),
            'command_joints_deg': q_command.round(4).tolist(),
            'after_joints_deg': q_after.round(4).tolist(),
            'target_delta_joints_deg': target_delta.round(4).tolist(),
            'command_delta_joints_deg': command_delta.round(4).tolist(),
            'actual_delta_joints_deg': (q_after - q_start).round(4).tolist(),
        }

    def write_results(self, results):
        save_path = Path(str(self.get_parameter('save_path').value)).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'created_unix_sec': time.time(),
            'execute': bool(self.get_parameter('execute').value),
            'parameters': {
                'levels': int(self.get_parameter('levels').value),
                'level_step_m': float(self.get_parameter('level_step_m').value),
                'test_dz_m': float(self.get_parameter('test_dz_m').value),
                'auto_step_down': bool(self.get_parameter('auto_step_down').value),
                'z_min_m': float(self.get_parameter('z_min_m').value),
                'z_max_m': float(self.get_parameter('z_max_m').value),
                'z_up_joint_delta_gain': float(self.get_parameter('z_up_joint_delta_gain').value),
                'z_down_joint_delta_gain': float(self.get_parameter('z_down_joint_delta_gain').value),
                'z_up_min_command_delta_deg': float(self.get_parameter('z_up_min_command_delta_deg').value),
                'z_down_min_command_delta_deg': float(self.get_parameter('z_down_min_command_delta_deg').value),
                'min_command_joint_names': str(self.get_parameter('min_command_joint_names').value),
                'z_up_min_command_joint_names': str(self.get_parameter('z_up_min_command_joint_names').value),
                'z_down_min_command_joint_names': str(self.get_parameter('z_down_min_command_joint_names').value),
            },
            'records': results,
        }
        save_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding='utf-8')
        self.get_logger().info(f'saved scan: {save_path}')

    def run(self):
        levels = max(1, int(self.get_parameter('levels').value))
        level_step = abs(float(self.get_parameter('level_step_m').value))
        test_dz = abs(float(self.get_parameter('test_dz_m').value))
        auto_step_down = bool(self.get_parameter('auto_step_down').value)
        if not auto_step_down and levels > 1:
            self.get_logger().warning('auto_step_down is false; limiting scan to the current level only')
            levels = 1
        gripper_percent = self.read_gripper_percent()
        results = []

        self.get_logger().info(
            f'z response scan: levels={levels}, level_step={level_step * 1000.0:.1f}mm, '
            f'test_dz={test_dz * 1000.0:.1f}mm, execute={bool(self.get_parameter("execute").value)}'
        )

        for level in range(levels):
            _, t_level = self.current_pose()
            self.get_logger().info(
                f'=== level {level + 1}/{levels}: xyz={np.round(t_level[:3, 3], 5).tolist()} ==='
            )
            results.append(self.move_relative_z_once(test_dz, f'level_{level + 1}_up', gripper_percent))
            results.append(self.move_relative_z_once(-test_dz, f'level_{level + 1}_down_return', gripper_percent))

            if auto_step_down and level < levels - 1:
                results.append(self.move_relative_z_once(-level_step, f'level_{level + 1}_step_down', gripper_percent))

        self.write_results(results)


def main(args=None):
    rclpy.init(args=args)
    node = LeRobotZResponseScan()
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
