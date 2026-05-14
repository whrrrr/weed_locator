#!/home/whr/miniconda3/envs/lerobot/bin/python
"""One-shot relative joint jog for diagnosing SO101 joint tracking."""

import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from weed_locator.srv import ReadJoints, WriteJoints


ARM_JOINTS = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
]
ALL_JOINTS = ARM_JOINTS + ['gripper']


class LeRobotJointJog(Node):
    """Move one URDF joint by a small relative angle and report tracking."""

    def __init__(self):
        super().__init__('lerobot_joint_jog')

        self.declare_parameter('joint', 'elbow_flex')
        self.declare_parameter('delta_deg', -2.0)
        self.declare_parameter('execute', False)
        self.declare_parameter('max_delta_deg', 8.0)
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('read_joints_service', '/lerobot/read_joints')
        self.declare_parameter('write_joints_service', '/lerobot/write_joints')
        self.declare_parameter('settle_sec', 2.0)
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
            rclpy.spin_once(self, timeout_sec=0.05)
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
        return [math.degrees(float(positions_by_name[name])) for name in ARM_JOINTS]

    def read_gripper_percent(self):
        if not self.read_client.wait_for_service(timeout_sec=float(self.get_parameter('timeout_sec').value)):
            raise RuntimeError('/lerobot/read_joints service is not available')

        future = self.read_client.call_async(ReadJoints.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=float(self.get_parameter('timeout_sec').value))
        result = future.result()
        if result is None or not result.success or len(result.positions) < len(ALL_JOINTS):
            raise RuntimeError('failed to read LeRobot joints for gripper position')
        return float(result.positions[5])

    def send_joints(self, targets):
        if not self.write_client.wait_for_service(timeout_sec=float(self.get_parameter('timeout_sec').value)):
            raise RuntimeError('/lerobot/write_joints service is not available')

        request = WriteJoints.Request()
        request.target_positions = targets
        future = self.write_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=float(self.get_parameter('timeout_sec').value))
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError('failed to execute /lerobot/write_joints command')

    def run(self):
        joint = str(self.get_parameter('joint').value)
        if joint not in ARM_JOINTS:
            raise RuntimeError(f'joint must be one of {ARM_JOINTS}')

        delta_deg = float(self.get_parameter('delta_deg').value)
        max_delta = float(self.get_parameter('max_delta_deg').value)
        if abs(delta_deg) > max_delta:
            raise RuntimeError(f'delta_deg={delta_deg:.3f} exceeds max_delta_deg={max_delta:.3f}')

        msg = self.wait_for_joint_state()
        if msg is None:
            raise RuntimeError('timed out waiting for /joint_states')

        q_start = self.current_arm_degrees(msg)
        gripper_percent = self.read_gripper_percent()
        targets = list(q_start) + [gripper_percent]
        joint_index = ARM_JOINTS.index(joint)
        targets[joint_index] += delta_deg

        self.get_logger().info(f'current joints deg [pan,lift,elbow,wrist,roll]: {[round(v, 3) for v in q_start]}')
        self.get_logger().info(f'target  joints deg [pan,lift,elbow,wrist,roll]: {[round(v, 3) for v in targets[:5]]}')
        self.get_logger().info(f'requested {joint}: {delta_deg:.3f} deg')

        if not bool(self.get_parameter('execute').value):
            self.get_logger().info('dry-run only; set execute:=true to send this joint command')
            return

        start_count = self.joint_state_count
        self.send_joints(targets)
        self.get_logger().info('joint jog command sent')
        self.spin_for(float(self.get_parameter('settle_sec').value))
        if self.joint_state_count == start_count:
            self.get_logger().warning('no fresh /joint_states received after command')
            return

        q_actual = self.current_arm_degrees(self.joint_state)
        actual_delta = q_actual[joint_index] - q_start[joint_index]
        remaining = targets[joint_index] - q_actual[joint_index]
        self.get_logger().info(f'actual  joints deg [pan,lift,elbow,wrist,roll]: {[round(v, 3) for v in q_actual]}')
        self.get_logger().info(f'actual {joint}: {actual_delta:.3f} deg, remaining error: {remaining:.3f} deg')


def main(args=None):
    rclpy.init(args=args)
    node = LeRobotJointJog()
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
