#!/home/whr/miniconda3/envs/lerobot/bin/python
"""ROS2 bridge for the official LeRobot SO101 follower driver."""

import json
import math
import threading
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from weed_locator.srv import ReadJoints, WriteJoints


LEROBOT_SRC = Path('/home/whr/lerobot/src')
if LEROBOT_SRC.exists():
    import sys

    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower


ARM_JOINTS = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
]
ALL_JOINTS = ARM_JOINTS + ['gripper']

URDF_LIMITS_RAD = {
    'shoulder_pan': (-1.91986, 1.91986),
    'shoulder_lift': (-1.74533, 1.74533),
    'elbow_flex': (-1.74533, 1.5708),
    'wrist_flex': (-1.65806, 1.65806),
    'wrist_roll': (-2.74385, 2.84121),
    'gripper': (-0.174533, 1.74533),
}


class LeRobotSO101Bridge(Node):
    """Expose LeRobot SO101 follower observations and commands to ROS2."""

    def __init__(self):
        super().__init__('lerobot_so101_bridge')

        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('robot_id', 'my_awesome_follower_arm')
        self.declare_parameter(
            'calibration_dir',
            '/home/whr/.cache/huggingface/lerobot/calibration/robots/so_follower',
        )
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('connect_on_start', True)
        self.declare_parameter('calibrate_on_connect', False)
        self.declare_parameter('use_degrees', True)
        self.declare_parameter('max_relative_target', 10.0)
        self.declare_parameter('command_units', 'degrees')
        self.declare_parameter('wrist_roll_sign', 1.0)
        self.declare_parameter('wrist_roll_offset_deg', 0.0)
        self.declare_parameter('joint_offsets_deg', '')
        self.declare_parameter('publish_joint_states', True)
        self.declare_parameter('publish_raw_observation', True)
        self.declare_parameter('elbow_p_coefficient', 16)
        self.declare_parameter('motor_p_coefficients', '')
        self.declare_parameter('use_gripper', True)
        self.declare_parameter('active_motor_joints', '')
        self.declare_parameter('motor_models', '')

        self.robot = None
        self.last_observation = {}
        self.lock = threading.RLock()

        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.obs_pub = self.create_publisher(String, '/lerobot/observation', 10)
        self.status_pub = self.create_publisher(String, '/lerobot/status', 10)

        self.create_service(Trigger, '/lerobot/connect', self.connect_callback)
        self.create_service(Trigger, '/lerobot/disconnect', self.disconnect_callback)
        self.create_service(ReadJoints, '/lerobot/read_joints', self.read_joints_callback)
        self.create_service(WriteJoints, '/lerobot/write_joints', self.write_joints_callback)

        rate = float(self.get_parameter('publish_rate').value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.timer_callback)

        if bool(self.get_parameter('connect_on_start').value):
            ok, message = self.connect_robot()
            if ok:
                self.get_logger().info(message)
            else:
                self.get_logger().error(message)

        self.get_logger().info('LeRobot SO101 bridge ready')
        self.get_logger().info('Services: /lerobot/connect, /lerobot/disconnect, /lerobot/read_joints, /lerobot/write_joints')

    def make_config(self):
        calibration_dir = str(self.get_parameter('calibration_dir').value).strip()
        max_relative_target = self.get_parameter('max_relative_target').value
        if float(max_relative_target) <= 0.0:
            max_relative_target = None

        return SO101FollowerConfig(
            port=str(self.get_parameter('port').value),
            id=str(self.get_parameter('robot_id').value),
            calibration_dir=Path(calibration_dir) if calibration_dir else None,
            use_degrees=bool(self.get_parameter('use_degrees').value),
            max_relative_target=max_relative_target,
            cameras={},
        )

    def active_motor_joints(self):
        raw = str(self.get_parameter('active_motor_joints').value).strip()
        if raw:
            joints = [joint.strip() for joint in raw.split(',') if joint.strip()]
        elif bool(self.get_parameter('use_gripper').value):
            joints = list(ALL_JOINTS)
        else:
            joints = list(ARM_JOINTS)

        unknown = [joint for joint in joints if joint not in ALL_JOINTS]
        if unknown:
            self.get_logger().warning(f'ignoring unknown active_motor_joints: {unknown}')
        return [joint for joint in joints if joint in ALL_JOINTS]

    def motor_model_overrides(self):
        raw = str(self.get_parameter('motor_models').value).strip()
        overrides = {}
        if not raw:
            return overrides
        for item in raw.split(','):
            item = item.strip()
            if not item:
                continue
            if ':' not in item:
                self.get_logger().warning(f'ignoring malformed motor_models item: {item!r}')
                continue
            joint, model = [part.strip() for part in item.split(':', 1)]
            if joint not in ALL_JOINTS:
                self.get_logger().warning(f'ignoring motor model override for unknown joint: {joint}')
                continue
            overrides[joint] = model
        return overrides

    def motor_p_coefficients(self):
        raw = str(self.get_parameter('motor_p_coefficients').value).strip()
        coefficients = {}
        if raw:
            for item in raw.split(','):
                item = item.strip()
                if not item:
                    continue
                if ':' not in item:
                    self.get_logger().warning(f'ignoring malformed motor_p_coefficients item: {item!r}')
                    continue
                joint, value = [part.strip() for part in item.split(':', 1)]
                if joint not in ALL_JOINTS:
                    self.get_logger().warning(f'ignoring P coefficient for unknown joint: {joint}')
                    continue
                try:
                    p_value = int(value)
                except ValueError:
                    self.get_logger().warning(f'ignoring non-integer P coefficient for {joint}: {value!r}')
                    continue
                if p_value > 0:
                    coefficients[joint] = p_value

        # Backward-compatible single-joint parameter.
        elbow_p = int(self.get_parameter('elbow_p_coefficient').value)
        if elbow_p > 0 and 'elbow_flex' not in coefficients:
            coefficients['elbow_flex'] = elbow_p
        return coefficients

    def joint_offsets_deg(self):
        raw = str(self.get_parameter('joint_offsets_deg').value).strip()
        offsets = {}
        if not raw:
            return offsets
        for item in raw.split(','):
            item = item.strip()
            if not item:
                continue
            if ':' not in item:
                self.get_logger().warning(f'ignoring malformed joint_offsets_deg item: {item!r}')
                continue
            joint, value = [part.strip() for part in item.split(':', 1)]
            if joint not in ALL_JOINTS:
                self.get_logger().warning(f'ignoring offset for unknown joint: {joint}')
                continue
            try:
                offsets[joint] = float(value)
            except ValueError:
                self.get_logger().warning(f'ignoring non-float offset for {joint}: {value!r}')
        return offsets

    def joint_offset_deg(self, joint_name):
        return self.joint_offsets_deg().get(joint_name, 0.0)

    def apply_motor_config(self):
        if self.robot is None:
            return

        active = set(self.active_motor_joints())
        overrides = self.motor_model_overrides()
        removed = []
        if hasattr(self.robot, 'bus') and hasattr(self.robot.bus, 'motors'):
            for joint in list(self.robot.bus.motors):
                if joint not in active:
                    self.robot.bus.motors.pop(joint, None)
                    removed.append(joint)
            for joint, model in overrides.items():
                if joint in self.robot.bus.motors:
                    self.robot.bus.motors[joint].model = model
        for owner in (self.robot, getattr(self.robot, 'bus', None)):
            calibration = getattr(owner, 'calibration', None)
            if isinstance(calibration, dict):
                for joint in list(calibration):
                    if joint not in active:
                        calibration.pop(joint, None)
        if removed:
            self.refresh_bus_cache()
            self.get_logger().info(f'removed inactive motors from LeRobot bus expectation: {removed}')
        if overrides:
            self.refresh_bus_cache()
            self.get_logger().info(f'applied motor model overrides: {overrides}')

    def refresh_bus_cache(self):
        bus = getattr(self.robot, 'bus', None)
        if bus is None or not hasattr(bus, 'motors'):
            return
        if hasattr(bus, '_id_to_model_dict'):
            bus._id_to_model_dict = {motor.id: motor.model for motor in bus.motors.values()}
        if hasattr(bus, '_id_to_name_dict'):
            bus._id_to_name_dict = {motor.id: name for name, motor in bus.motors.items()}
        for cached_name in ('ids', 'models', '_has_different_ctrl_tables'):
            bus.__dict__.pop(cached_name, None)

    def apply_tuning(self):
        coefficients = self.motor_p_coefficients()
        if not coefficients or self.robot is None or not self.robot.is_connected:
            return

        for joint, p_value in coefficients.items():
            if joint not in self.robot.bus.motors:
                self.get_logger().warning(f'skipping P coefficient for inactive motor {joint}')
                continue
            try:
                with self.robot.bus.torque_disabled(joint):
                    self.robot.bus.write('P_Coefficient', joint, p_value, normalize=False, num_retry=5)
                readback = self.robot.bus.read('P_Coefficient', joint, normalize=False, num_retry=5)
                self.get_logger().info(f'{joint} P_Coefficient set to {readback}')
            except Exception as exc:
                self.get_logger().warning(f'failed to set {joint} P_Coefficient={p_value}: {exc}')

    def connect_robot(self):
        with self.lock:
            if self.robot is not None and self.robot.is_connected:
                return True, 'LeRobot SO101 already connected'

            try:
                self.robot = SO101Follower(self.make_config())
                self.apply_motor_config()
                self.robot.connect(calibrate=bool(self.get_parameter('calibrate_on_connect').value))
                self.apply_tuning()
                self.publish_status('connected')
                return True, f'LeRobot SO101 connected on {self.get_parameter("port").value}'
            except Exception as exc:
                self.robot = None
                self.publish_status(f'connect_failed: {exc}')
                return False, f'Failed to connect LeRobot SO101: {exc}'

    def disconnect_robot(self):
        with self.lock:
            if self.robot is None:
                self.publish_status('disconnected')
                return True, 'LeRobot SO101 already disconnected'

            try:
                if self.robot.is_connected:
                    self.robot.disconnect()
                self.robot = None
                self.publish_status('disconnected')
                return True, 'LeRobot SO101 disconnected'
            except Exception as exc:
                self.publish_status(f'disconnect_failed: {exc}')
                return False, f'Failed to disconnect LeRobot SO101: {exc}'

    def connect_callback(self, request, response):
        response.success, response.message = self.connect_robot()
        return response

    def disconnect_callback(self, request, response):
        response.success, response.message = self.disconnect_robot()
        return response

    def timer_callback(self):
        with self.lock:
            if self.robot is None or not self.robot.is_connected:
                return

            try:
                observation = self.robot.get_observation()
            except Exception as exc:
                self.get_logger().error(f'LeRobot observation failed: {exc}')
                self.publish_status(f'observation_failed: {exc}')
                return

            self.last_observation = dict(observation)

        if bool(self.get_parameter('publish_joint_states').value):
            self.publish_joint_state(self.last_observation)

        if bool(self.get_parameter('publish_raw_observation').value):
            msg = String()
            msg.data = json.dumps(self.last_observation, ensure_ascii=False, sort_keys=True)
            self.obs_pub.publish(msg)

    def publish_joint_state(self, observation):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ALL_JOINTS
        msg.position = [
            self.observation_to_ros_radians(joint, observation.get(f'{joint}.pos', 0.0))
            for joint in ALL_JOINTS
        ]
        self.joint_pub.publish(msg)

    def observation_to_ros_radians(self, joint_name, value):
        value = float(value)
        if joint_name == 'gripper':
            return self.percent_to_limit_rad(joint_name, value)

        if bool(self.get_parameter('use_degrees').value):
            if joint_name == 'wrist_roll':
                value = self.correct_wrist_roll_degrees(value)
            value += self.joint_offset_deg(joint_name)
            return math.radians(value)

        return self.normalized_to_limit_rad(joint_name, value)

    def normalized_to_limit_rad(self, joint_name, value):
        low, high = URDF_LIMITS_RAD[joint_name]
        ratio = (float(value) + 100.0) / 200.0
        ratio = max(0.0, min(1.0, ratio))
        return low + ratio * (high - low)

    def percent_to_limit_rad(self, joint_name, value):
        low, high = URDF_LIMITS_RAD[joint_name]
        ratio = float(value) / 100.0
        ratio = max(0.0, min(1.0, ratio))
        return low + ratio * (high - low)

    def read_joints_callback(self, request, response):
        with self.lock:
            if not self.last_observation:
                response.positions = []
                response.success = False
                return response

            response.positions = [float(self.last_observation.get(f'{joint}.pos', 0.0)) for joint in ALL_JOINTS]
            response.success = True
            return response

    def write_joints_callback(self, request, response):
        units = str(self.get_parameter('command_units').value).lower().strip()
        targets = list(request.target_positions)

        active_joints = self.active_motor_joints()
        if len(targets) == 1 and bool(self.get_parameter('use_gripper').value):
            active_joints = ['gripper']
        elif len(targets) < len(active_joints):
            response.success = False
            return response

        action = {}
        for joint, target in zip(active_joints, targets[: len(active_joints)]):
            action[f'{joint}.pos'] = self.command_to_lerobot_value(joint, float(target), units)

        with self.lock:
            if self.robot is None or not self.robot.is_connected:
                response.success = False
                return response

            try:
                sent_action = self.robot.send_action(action)
                self.get_logger().info(f'LeRobot action sent: {sent_action}')
                response.success = True
            except Exception as exc:
                self.get_logger().error(f'LeRobot action failed: {exc}')
                self.publish_status(f'action_failed: {exc}')
                response.success = False

        return response

    def command_to_lerobot_value(self, joint_name, target, units):
        if units == 'radians':
            if joint_name == 'gripper':
                return self.gripper_rad_to_percent(target)
            target_deg = math.degrees(target)
            target_deg -= self.joint_offset_deg(joint_name)
            if joint_name == 'wrist_roll':
                target_deg = self.uncorrect_wrist_roll_degrees(target_deg)
            return target_deg if bool(self.get_parameter('use_degrees').value) else target

        if units == 'normalized' and bool(self.get_parameter('use_degrees').value):
            if joint_name == 'gripper':
                return target
            target_deg = math.degrees(self.normalized_to_limit_rad(joint_name, target))
            target_deg -= self.joint_offset_deg(joint_name)
            if joint_name == 'wrist_roll':
                target_deg = self.uncorrect_wrist_roll_degrees(target_deg)
            return target_deg

        if units == 'degrees':
            target -= self.joint_offset_deg(joint_name)

        if units == 'degrees' and joint_name == 'wrist_roll':
            return self.uncorrect_wrist_roll_degrees(target)

        return target

    def correct_wrist_roll_degrees(self, value):
        sign = float(self.get_parameter('wrist_roll_sign').value)
        offset = float(self.get_parameter('wrist_roll_offset_deg').value)
        return sign * float(value) + offset

    def uncorrect_wrist_roll_degrees(self, value):
        sign = float(self.get_parameter('wrist_roll_sign').value)
        offset = float(self.get_parameter('wrist_roll_offset_deg').value)
        if abs(sign) < 1e-9:
            sign = 1.0
        return (float(value) - offset) / sign

    def gripper_rad_to_percent(self, value):
        low, high = URDF_LIMITS_RAD['gripper']
        ratio = (float(value) - low) / (high - low)
        ratio = max(0.0, min(1.0, ratio))
        return ratio * 100.0

    def publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def destroy_node(self):
        self.disconnect_robot()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LeRobotSO101Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
