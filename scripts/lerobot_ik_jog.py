#!/home/whr/miniconda3/envs/lerobot/bin/python
"""Safe one-shot LeRobot IK jog for SO101.

Default mode is dry-run. Set execute:=true to send the solved joints to the
LeRobot bridge through /lerobot/write_joints.
"""

import math
import sys
import time
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


class LeRobotIKJog(Node):
    """Compute a small Cartesian jog with LeRobot's official kinematics."""

    def __init__(self):
        super().__init__('lerobot_ik_jog')

        self.declare_parameter('dx', 0.0)
        self.declare_parameter('dy', 0.0)
        self.declare_parameter('dz', 0.0)
        self.declare_parameter('relative_frame', 'base')
        self.declare_parameter('execute', False)
        self.declare_parameter('target_frame', 'gripper_frame_link')
        self.declare_parameter('position_weight', 1.0)
        self.declare_parameter('orientation_weight', 0.0)
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('read_joints_service', '/lerobot/read_joints')
        self.declare_parameter('write_joints_service', '/lerobot/write_joints')
        self.declare_parameter('max_joint_delta_deg', 8.0)
        self.declare_parameter('max_position_error_m', 0.002)
        self.declare_parameter('verify_after_execute', True)
        self.declare_parameter('settle_sec', 1.0)
        self.declare_parameter('timeout_sec', 3.0)

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

    def on_joint_state(self, msg):
        self.joint_state = msg
        self.joint_state_count += 1

    def wait_for_joint_state(self):
        deadline = time.monotonic() + float(self.get_parameter('timeout_sec').value)
        while rclpy.ok() and self.joint_state is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.joint_state

    def spin_for(self, duration_sec):
        deadline = time.monotonic() + max(0.0, float(duration_sec))
        while rclpy.ok() and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            rclpy.spin_once(self, timeout_sec=min(0.05, max(0.0, remaining)))

    def current_arm_degrees(self, msg):
        positions_by_name = dict(zip(msg.name, msg.position, strict=False))
        missing = [name for name in ARM_JOINTS if name not in positions_by_name]
        if missing:
            raise RuntimeError(f'/joint_states missing joints: {missing}')
        return np.array([math.degrees(float(positions_by_name[name])) for name in ARM_JOINTS], dtype=float)

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

    def desired_pose(self, current_pose):
        delta = np.array([
            float(self.get_parameter('dx').value),
            float(self.get_parameter('dy').value),
            float(self.get_parameter('dz').value),
        ])
        if np.allclose(delta, 0.0):
            raise RuntimeError('no Cartesian jog requested; set dx, dy, or dz')

        frame = str(self.get_parameter('relative_frame').value).lower().strip()

        target = current_pose.copy()
        if frame == 'tool':
            target[:3, 3] += current_pose[:3, :3] @ delta
        else:
            target[:3, 3] += delta
        return target

    def verify_executed_pose(self, kinematics, start_pose, target_pose):
        start_count = self.joint_state_count
        self.spin_for(float(self.get_parameter('settle_sec').value))
        if self.joint_state_count == start_count:
            self.get_logger().warning('no fresh /joint_states received after command')
            return

        q_actual = self.current_arm_degrees(self.joint_state)
        t_actual = kinematics.forward_kinematics(q_actual)
        actual_delta = t_actual[:3, 3] - start_pose[:3, 3]
        desired_delta = target_pose[:3, 3] - start_pose[:3, 3]
        target_error = float(np.linalg.norm(t_actual[:3, 3] - target_pose[:3, 3]))

        self.get_logger().info(f'actual xyz after command: {np.round(t_actual[:3, 3], 5).tolist()}')
        self.get_logger().info(f'actual delta xyz: {np.round(actual_delta, 5).tolist()}')
        self.get_logger().info(f'desired delta xyz: {np.round(desired_delta, 5).tolist()}')
        self.get_logger().info(f'post-command target error: {target_error:.6f} m')

    def run(self):
        msg = self.wait_for_joint_state()
        if msg is None:
            raise RuntimeError('timed out waiting for /joint_states')

        q_current = self.current_arm_degrees(msg)
        gripper_percent = self.read_gripper_percent()

        kinematics = self.make_kinematics()
        t_current = kinematics.forward_kinematics(q_current)
        t_target = self.desired_pose(t_current)
        q_target = kinematics.inverse_kinematics(
            q_current,
            t_target,
            position_weight=float(self.get_parameter('position_weight').value),
            orientation_weight=float(self.get_parameter('orientation_weight').value),
        )
        t_result = kinematics.forward_kinematics(q_target)

        delta_deg = q_target - q_current
        pos_error = float(np.linalg.norm(t_result[:3, 3] - t_target[:3, 3]))
        max_delta = float(np.max(np.abs(delta_deg)))

        self.get_logger().info(f'current xyz: {np.round(t_current[:3, 3], 5).tolist()}')
        self.get_logger().info(f'target  xyz: {np.round(t_target[:3, 3], 5).tolist()}')
        self.get_logger().info(f'result  xyz: {np.round(t_result[:3, 3], 5).tolist()}')
        self.get_logger().info(f'current joints deg: {np.round(q_current, 3).tolist()}')
        self.get_logger().info(f'target  joints deg: {np.round(q_target, 3).tolist()}')
        self.get_logger().info(f'delta   joints deg: {np.round(delta_deg, 3).tolist()}')
        self.get_logger().info(f'IK position error: {pos_error:.6f} m, max joint delta: {max_delta:.3f} deg')

        max_allowed_delta = float(self.get_parameter('max_joint_delta_deg').value)
        max_allowed_error = float(self.get_parameter('max_position_error_m').value)
        safe = max_delta <= max_allowed_delta and pos_error <= max_allowed_error

        if not safe:
            raise RuntimeError(
                f'IK result rejected: max_delta={max_delta:.3f} deg '
                f'(limit {max_allowed_delta:.3f}), pos_error={pos_error:.6f} m '
                f'(limit {max_allowed_error:.6f})'
            )

        if not bool(self.get_parameter('execute').value):
            self.get_logger().info('dry-run only; set execute:=true to send this motion')
            return

        if not self.write_client.wait_for_service(timeout_sec=float(self.get_parameter('timeout_sec').value)):
            raise RuntimeError('/lerobot/write_joints service is not available')

        request = WriteJoints.Request()
        request.target_positions = q_target.tolist() + [gripper_percent]
        future = self.write_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=float(self.get_parameter('timeout_sec').value))
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError('failed to execute /lerobot/write_joints command')
        self.get_logger().info('IK jog command sent')
        if bool(self.get_parameter('verify_after_execute').value):
            self.verify_executed_pose(kinematics, t_current, t_target)


def main(args=None):
    rclpy.init(args=args)
    node = LeRobotIKJog()
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
