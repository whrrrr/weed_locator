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
        self.declare_parameter('urdf_path', '')
        self.declare_parameter('arm_joint_names', ','.join(ARM_JOINTS))
        self.declare_parameter('position_weight', 1.0)
        self.declare_parameter('orientation_weight', 0.0)
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('read_joints_service', '/lerobot/read_joints')
        self.declare_parameter('write_joints_service', '/lerobot/write_joints')
        self.declare_parameter('max_joint_delta_deg', 8.0)
        self.declare_parameter('max_position_error_m', 0.002)
        self.declare_parameter('joint_delta_gain', 1.0)
        self.declare_parameter('z_up_joint_delta_gain', 0.0)
        self.declare_parameter('z_down_joint_delta_gain', 0.0)
        self.declare_parameter('min_command_delta_deg', 0.0)
        self.declare_parameter('z_up_min_command_delta_deg', -1.0)
        self.declare_parameter('z_down_min_command_delta_deg', -1.0)
        self.declare_parameter('min_command_joint_names', '')
        self.declare_parameter('z_up_min_command_joint_names', '')
        self.declare_parameter('z_down_min_command_joint_names', '')
        self.declare_parameter('closed_loop', False)
        self.declare_parameter('closed_loop_iters', 3)
        self.declare_parameter('closed_loop_tolerance_m', 0.004)
        self.declare_parameter('verify_after_execute', True)
        self.declare_parameter('settle_sec', 1.0)
        self.declare_parameter('timeout_sec', 3.0)

        self.joint_state = None
        self.joint_state_count = 0
        self.arm_joints = self.parse_arm_joint_names()
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
            rclpy.spin_once(self, timeout_sec=min(0.05, max(0.0, remaining)))

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

    def read_pose_after_settle(self, kinematics):
        start_count = self.joint_state_count
        self.spin_for(float(self.get_parameter('settle_sec').value))
        if self.joint_state_count == start_count:
            self.get_logger().warning('no fresh /joint_states received after command')
            return None

        q_actual = self.current_arm_degrees(self.joint_state)
        t_actual = kinematics.forward_kinematics(q_actual)
        return q_actual, t_actual

    def verify_executed_pose(self, kinematics, start_pose, target_pose):
        measured = self.read_pose_after_settle(kinematics)
        if measured is None:
            return None

        q_actual, t_actual = measured
        actual_delta = t_actual[:3, 3] - start_pose[:3, 3]
        desired_delta = target_pose[:3, 3] - start_pose[:3, 3]
        target_error = float(np.linalg.norm(t_actual[:3, 3] - target_pose[:3, 3]))

        self.get_logger().info(f'actual xyz after command: {np.round(t_actual[:3, 3], 5).tolist()}')
        self.get_logger().info(f'actual delta xyz: {np.round(actual_delta, 5).tolist()}')
        self.get_logger().info(f'desired delta xyz: {np.round(desired_delta, 5).tolist()}')
        self.get_logger().info(f'post-command target error: {target_error:.6f} m')
        return q_actual, t_actual, target_error

    def compensation_settings(self, cartesian_delta):
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

        return gain, abs(min_delta), min_joint_names

    def apply_command_compensation(self, q_current, q_target, cartesian_delta):
        delta = q_target - q_current
        gain, min_delta, min_joint_names = self.compensation_settings(cartesian_delta)
        if abs(gain - 1.0) < 1e-9 and min_delta <= 1e-9:
            return q_target

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

    def send_joint_command(self, q_command, gripper_percent):
        if not self.write_client.wait_for_service(timeout_sec=float(self.get_parameter('timeout_sec').value)):
            raise RuntimeError('/lerobot/write_joints service is not available')

        targets = [float(value) for value in q_command]
        targets.append(float(gripper_percent))
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

    def closed_loop_correct(self, kinematics, target_pose, gripper_percent):
        max_iters = max(0, int(self.get_parameter('closed_loop_iters').value))
        tolerance = float(self.get_parameter('closed_loop_tolerance_m').value)
        max_allowed_delta = float(self.get_parameter('max_joint_delta_deg').value)
        max_allowed_error = float(self.get_parameter('max_position_error_m').value)

        for iteration in range(1, max_iters + 1):
            measured = self.read_pose_after_settle(kinematics)
            if measured is None:
                return

            q_actual, t_actual = measured
            target_error = float(np.linalg.norm(t_actual[:3, 3] - target_pose[:3, 3]))
            self.get_logger().info(
                f'closed-loop actual xyz: {np.round(t_actual[:3, 3], 5).tolist()}, '
                f'remaining error={target_error * 1000.0:.1f} mm'
            )
            if target_error <= tolerance:
                self.get_logger().info(
                    f'closed-loop converged at iter {iteration}: error={target_error * 1000.0:.1f} mm'
                )
                return

            q_target = kinematics.inverse_kinematics(
                q_actual,
                target_pose,
                position_weight=float(self.get_parameter('position_weight').value),
                orientation_weight=float(self.get_parameter('orientation_weight').value),
            )
            cartesian_delta = target_pose[:3, 3] - t_actual[:3, 3]
            q_command = self.apply_command_compensation(q_actual, q_target, cartesian_delta)
            t_result = kinematics.forward_kinematics(q_target)

            delta_deg = q_target - q_actual
            command_delta_deg = q_command - q_actual
            pos_error = float(np.linalg.norm(t_result[:3, 3] - target_pose[:3, 3]))
            max_delta = float(np.max(np.abs(command_delta_deg)))
            self.get_logger().info(
                f'closed-loop iter {iteration}: remaining={target_error * 1000.0:.1f} mm, '
                f'ik_error={pos_error * 1000.0:.1f} mm, max_joint={max_delta:.2f} deg'
            )
            self.get_logger().info(
                f'closed-loop delta joints deg: {np.round(delta_deg, 3).tolist()}'
            )
            if not np.allclose(command_delta_deg, delta_deg):
                self.get_logger().info(
                    f'closed-loop command delta joints deg: {np.round(command_delta_deg, 3).tolist()}'
                )

            if max_delta > max_allowed_delta or pos_error > max_allowed_error:
                self.get_logger().warning(
                    f'closed-loop stop: max_delta={max_delta:.3f} deg '
                    f'(limit {max_allowed_delta:.3f}), pos_error={pos_error:.6f} m '
                    f'(limit {max_allowed_error:.6f})'
                )
                return

            self.send_joint_command(q_command, gripper_percent)
            self.get_logger().info(f'closed-loop correction {iteration} sent')

        measured = self.read_pose_after_settle(kinematics)
        if measured is not None:
            _, t_final = measured
            final_error = float(np.linalg.norm(t_final[:3, 3] - target_pose[:3, 3]))
            self.get_logger().info(
                f'closed-loop final xyz: {np.round(t_final[:3, 3], 5).tolist()}, '
                f'final error={final_error * 1000.0:.1f} mm'
            )

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
        q_command = self.apply_command_compensation(
            q_current,
            q_target,
            t_target[:3, 3] - t_current[:3, 3],
        )
        t_result = kinematics.forward_kinematics(q_target)

        delta_deg = q_target - q_current
        command_delta_deg = q_command - q_current
        pos_error = float(np.linalg.norm(t_result[:3, 3] - t_target[:3, 3]))
        max_delta = float(np.max(np.abs(command_delta_deg)))

        self.get_logger().info(f'current xyz: {np.round(t_current[:3, 3], 5).tolist()}')
        self.get_logger().info(f'target  xyz: {np.round(t_target[:3, 3], 5).tolist()}')
        self.get_logger().info(f'result  xyz: {np.round(t_result[:3, 3], 5).tolist()}')
        self.get_logger().info(f'current joints deg: {np.round(q_current, 3).tolist()}')
        self.get_logger().info(f'target  joints deg: {np.round(q_target, 3).tolist()}')
        self.get_logger().info(f'delta   joints deg: {np.round(delta_deg, 3).tolist()}')
        if not np.allclose(command_delta_deg, delta_deg):
            self.get_logger().info(f'command joints deg: {np.round(q_command, 3).tolist()}')
            self.get_logger().info(f'command delta joints deg: {np.round(command_delta_deg, 3).tolist()}')
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

        self.send_joint_command(q_command, gripper_percent)
        self.get_logger().info('IK jog command sent')
        if bool(self.get_parameter('closed_loop').value):
            self.closed_loop_correct(kinematics, t_target, gripper_percent)
        elif bool(self.get_parameter('verify_after_execute').value):
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
