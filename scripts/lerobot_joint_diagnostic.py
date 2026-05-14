#!/home/whr/miniconda3/envs/lerobot/bin/python
"""Repeated joint tracking diagnostic for LeRobot SO101."""

import math
import statistics
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


class LeRobotJointDiagnostic(Node):
    """Run small alternating joint moves and report tracking quality."""

    def __init__(self):
        super().__init__('lerobot_joint_diagnostic')

        self.declare_parameter('joints', 'elbow_flex')
        self.declare_parameter('delta_deg', 2.0)
        self.declare_parameter('cycles', 3)
        self.declare_parameter('execute', False)
        self.declare_parameter('max_delta_deg', 5.0)
        self.declare_parameter('start_negative', True)
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

    def parse_joints(self):
        requested = [item.strip() for item in str(self.get_parameter('joints').value).split(',') if item.strip()]
        invalid = [joint for joint in requested if joint not in ALL_JOINTS]
        if invalid:
            raise RuntimeError(f'invalid joints {invalid}; valid joints are {ALL_JOINTS}')
        if not requested:
            raise RuntimeError('no joints requested')
        return requested

    def unit_for_joint(self, joint):
        return '%' if joint == 'gripper' else 'deg'

    def command_relative(self, joint, delta, step_index):
        msg = self.wait_for_joint_state()
        if msg is None:
            raise RuntimeError('timed out waiting for /joint_states')

        q_start = self.current_arm_degrees(msg)
        gripper_percent = self.read_gripper_percent()
        targets = list(q_start) + [gripper_percent]
        unit = self.unit_for_joint(joint)

        if joint == 'gripper':
            joint_index = len(ALL_JOINTS) - 1
            start_position = gripper_percent
        else:
            joint_index = ARM_JOINTS.index(joint)
            start_position = q_start[joint_index]

        targets[joint_index] += delta
        if joint == 'gripper' and not 0.0 <= targets[joint_index] <= 100.0:
            raise RuntimeError(
                f'gripper target {targets[joint_index]:.2f}% is outside 0-100%; '
                'move gripper away from the limit or reduce delta_deg'
            )

        if not bool(self.get_parameter('execute').value):
            self.get_logger().info(
                f'dry-run step {step_index}: {joint} {delta:+.2f} {unit}, '
                f'from {start_position:.2f} to {targets[joint_index]:.2f}'
            )
            return None

        start_count = self.joint_state_count
        self.send_joints(targets)
        self.spin_for(float(self.get_parameter('settle_sec').value))
        if self.joint_state_count == start_count:
            raise RuntimeError('no fresh /joint_states received after command')

        if joint == 'gripper':
            actual_position = self.read_gripper_percent()
        else:
            q_actual = self.current_arm_degrees(self.joint_state)
            actual_position = q_actual[joint_index]

        actual_delta = actual_position - start_position
        remaining = targets[joint_index] - actual_position
        completion = actual_delta / delta if abs(delta) > 1e-9 else 0.0

        self.get_logger().info(
            f'step {step_index}: {joint} target {delta:+.2f} {unit}, '
            f'actual {actual_delta:+.2f} {unit}, remaining {remaining:+.2f} {unit}, '
            f'completion {completion * 100.0:.0f}%'
        )
        return {
            'joint': joint,
            'target_delta': delta,
            'actual_delta': actual_delta,
            'remaining': remaining,
            'completion': completion,
            'unit': unit,
        }

    def summarize(self, records):
        if not records:
            self.get_logger().info('dry-run complete; set execute:=true to move and collect tracking data')
            return

        for joint in sorted({record['joint'] for record in records}):
            for sign_name, predicate in (
                ('positive', lambda value: value > 0.0),
                ('negative', lambda value: value < 0.0),
            ):
                signed = [record for record in records if record['joint'] == joint and predicate(record['target_delta'])]
                if not signed:
                    continue
                completions = [record['completion'] * 100.0 for record in signed]
                remainings = [abs(record['remaining']) for record in signed]
                self.get_logger().info(
                    f'summary {joint} {sign_name}: '
                    f'completion mean {statistics.mean(completions):.0f}% '
                    f'min {min(completions):.0f}% max {max(completions):.0f}%, '
                    f'abs remaining mean {statistics.mean(remainings):.2f} {signed[0]["unit"]}'
                )

    def run(self):
        joints = self.parse_joints()
        delta = abs(float(self.get_parameter('delta_deg').value))
        max_delta = float(self.get_parameter('max_delta_deg').value)
        if delta <= 0.0:
            raise RuntimeError('delta_deg must be positive')
        if delta > max_delta:
            raise RuntimeError(f'delta_deg={delta:.3f} exceeds max_delta_deg={max_delta:.3f}')

        cycles = int(self.get_parameter('cycles').value)
        if cycles < 1:
            raise RuntimeError('cycles must be >= 1')

        if self.wait_for_joint_state() is None:
            raise RuntimeError('timed out waiting for /joint_states')

        signs = [-1.0, 1.0] if bool(self.get_parameter('start_negative').value) else [1.0, -1.0]
        records = []
        step_index = 1
        units = sorted({self.unit_for_joint(joint) for joint in joints})
        unit_label = units[0] if len(units) == 1 else 'mixed'
        self.get_logger().info(
            f'joint diagnostic: joints={joints}, delta={delta:.2f} {unit_label}, '
            f'cycles={cycles}, execute={bool(self.get_parameter("execute").value)}'
        )
        for joint in joints:
            for _ in range(cycles):
                for sign in signs:
                    record = self.command_relative(joint, sign * delta, step_index)
                    if record is not None:
                        records.append(record)
                    step_index += 1
        self.summarize(records)


def main(args=None):
    rclpy.init(args=args)
    node = LeRobotJointDiagnostic()
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
