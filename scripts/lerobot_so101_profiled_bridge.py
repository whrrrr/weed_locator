#!/home/whr/miniconda3/envs/lerobot/bin/python
"""Profiled ROS2 bridge for testing SO101 internal velocity/acceleration limits.

This keeps the normal LeRobot SO101 bridge behavior, but after connecting it
also writes Feetech profile registers such as Goal_Velocity and Acceleration.
The goal is to let the servo's internal controller move to position targets
more gently, instead of relying only on high-rate ROS-side interpolation.
"""

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


class LeRobotSO101ProfiledBridge(Node):
    """Expose SO101 joints while applying Feetech profile register settings."""

    def __init__(self):
        super().__init__('lerobot_so101_profiled_bridge')

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
        self.declare_parameter('hardware_joint_map', '')
        self.declare_parameter('servo_goal_velocities', '')
        self.declare_parameter('servo_accelerations', '')
        self.declare_parameter('servo_maximum_accelerations', '')
        self.declare_parameter('servo_goal_times', '')
        self.declare_parameter('auto_goal_velocity', False)
        self.declare_parameter('auto_goal_velocity_dt_sec', 0.15)
        self.declare_parameter('auto_goal_velocity_gain', 4.0)
        self.declare_parameter('auto_goal_velocity_min', 10)
        self.declare_parameter('auto_goal_velocity_max', 120)
        self.declare_parameter('auto_goal_velocity_update_threshold', 3)

        self.robot = None
        self.last_observation = {}
        self.last_auto_goal_velocities = {}
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

    def hardware_joint_map(self):
        """Map logical ROS joint names to the actual LeRobot bus motor names.

        This is useful for temporary physical motor swaps without changing
        servo IDs or LeRobot calibration files. Example:
        shoulder_lift:elbow_flex,elbow_flex:shoulder_lift
        """
        raw = str(self.get_parameter('hardware_joint_map').value).strip()
        mapping = {}
        if not raw:
            return mapping
        for item in raw.split(','):
            item = item.strip()
            if not item:
                continue
            if ':' not in item:
                self.get_logger().warning(f'ignoring malformed hardware_joint_map item: {item!r}')
                continue
            logical, hardware = [part.strip() for part in item.split(':', 1)]
            if logical not in ALL_JOINTS or hardware not in ALL_JOINTS:
                self.get_logger().warning(f'ignoring invalid hardware_joint_map item: {item!r}')
                continue
            mapping[logical] = hardware
        return mapping

    def hardware_joint_for(self, logical_joint):
        return self.hardware_joint_map().get(logical_joint, logical_joint)

    def active_hardware_joints(self):
        joints = []
        for logical in self.active_motor_joints():
            hardware = self.hardware_joint_for(logical)
            if hardware not in joints:
                joints.append(hardware)
        return joints

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

    def parse_int_joint_map(self, parameter_name):
        raw = str(self.get_parameter(parameter_name).value).strip()
        values = {}
        if not raw:
            return values
        for item in raw.split(','):
            item = item.strip()
            if not item:
                continue
            if ':' not in item:
                self.get_logger().warning(f'ignoring malformed {parameter_name} item: {item!r}')
                continue
            joint, value = [part.strip() for part in item.split(':', 1)]
            if joint not in ALL_JOINTS:
                self.get_logger().warning(f'ignoring {parameter_name} for unknown joint: {joint}')
                continue
            try:
                values[joint] = int(value)
            except ValueError:
                self.get_logger().warning(f'ignoring non-integer {parameter_name} for {joint}: {value!r}')
        return values

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

    def command_target_to_ros_degrees(self, joint_name, target, units):
        if units == 'radians':
            return math.degrees(float(target))
        if units == 'normalized':
            if joint_name == 'gripper':
                return float(target)
            return math.degrees(self.normalized_to_limit_rad(joint_name, target))
        return float(target)

    def observation_to_ros_degrees(self, joint_name, observation):
        hardware_joint = self.hardware_joint_for(joint_name)
        value = float(observation.get(f'{hardware_joint}.pos', 0.0))
        if joint_name == 'gripper':
            return math.degrees(self.percent_to_limit_rad(joint_name, value))
        if joint_name == 'wrist_roll':
            value = self.correct_wrist_roll_degrees(value)
        return value + self.joint_offset_deg(joint_name)

    def apply_motor_config(self):
        if self.robot is None:
            return

        active = set(self.active_hardware_joints())
        overrides = self.motor_model_overrides()
        removed = []
        if hasattr(self.robot, 'bus') and hasattr(self.robot.bus, 'motors'):
            for joint in list(self.robot.bus.motors):
                if joint not in active:
                    self.robot.bus.motors.pop(joint, None)
                    removed.append(joint)
            for logical_joint, model in overrides.items():
                hardware_joint = self.hardware_joint_for(logical_joint)
                if hardware_joint in self.robot.bus.motors:
                    self.robot.bus.motors[hardware_joint].model = model
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
            mapped = {self.hardware_joint_for(joint): model for joint, model in overrides.items()}
            self.get_logger().info(f'applied motor model overrides: {mapped}')

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
        if self.robot is None or not self.robot.is_connected:
            return

        for logical_joint, p_value in coefficients.items():
            joint = self.hardware_joint_for(logical_joint)
            if joint not in self.robot.bus.motors:
                self.get_logger().warning(f'skipping P coefficient for inactive motor {joint}')
                continue
            try:
                with self.robot.bus.torque_disabled(joint):
                    self.robot.bus.write('P_Coefficient', joint, p_value, normalize=False, num_retry=5)
                readback = self.robot.bus.read('P_Coefficient', joint, normalize=False, num_retry=5)
                self.get_logger().info(f'{logical_joint}->{joint} P_Coefficient set to {readback}')
            except Exception as exc:
                self.get_logger().warning(f'failed to set {logical_joint}->{joint} P_Coefficient={p_value}: {exc}')

        self.apply_profile_registers()

    def apply_profile_registers(self):
        register_specs = [
            ('Goal_Velocity', self.parse_int_joint_map('servo_goal_velocities'), False),
            ('Acceleration', self.parse_int_joint_map('servo_accelerations'), False),
            ('Maximum_Acceleration', self.parse_int_joint_map('servo_maximum_accelerations'), True),
            ('Goal_Time', self.parse_int_joint_map('servo_goal_times'), False),
        ]
        for register, values, torque_disable in register_specs:
            for logical_joint, value in values.items():
                joint = self.hardware_joint_for(logical_joint)
                if joint not in self.robot.bus.motors:
                    self.get_logger().warning(f'skipping {register} for inactive motor {logical_joint}->{joint}')
                    continue
                try:
                    if torque_disable:
                        with self.robot.bus.torque_disabled(joint):
                            self.robot.bus.write(register, joint, value, normalize=False, num_retry=5)
                    else:
                        self.robot.bus.write(register, joint, value, normalize=False, num_retry=5)
                    readback = self.robot.bus.read(register, joint, normalize=False, num_retry=5)
                    self.get_logger().info(f'{logical_joint}->{joint} {register} set to {readback}')
                except Exception as exc:
                    self.get_logger().warning(f'failed to set {logical_joint}->{joint} {register}={value}: {exc}')

    def auto_goal_velocity_value(self, delta_deg):
        dt = max(0.02, float(self.get_parameter('auto_goal_velocity_dt_sec').value))
        gain = max(0.0, float(self.get_parameter('auto_goal_velocity_gain').value))
        min_value = int(self.get_parameter('auto_goal_velocity_min').value)
        max_value = int(self.get_parameter('auto_goal_velocity_max').value)
        if abs(delta_deg) < 0.05:
            return 0
        value = int(round(min_value + gain * abs(delta_deg) / dt))
        return max(0, min(max_value, value))

    def apply_auto_goal_velocities(self, command_items, units):
        if not bool(self.get_parameter('auto_goal_velocity').value):
            return
        if not self.last_observation:
            return

        threshold = int(self.get_parameter('auto_goal_velocity_update_threshold').value)
        for logical_joint, target in command_items:
            if logical_joint == 'gripper':
                continue
            hardware_joint = self.hardware_joint_for(logical_joint)
            if hardware_joint not in self.robot.bus.motors:
                continue
            target_deg = self.command_target_to_ros_degrees(logical_joint, target, units)
            current_deg = self.observation_to_ros_degrees(logical_joint, self.last_observation)
            raw_velocity = self.auto_goal_velocity_value(target_deg - current_deg)
            if raw_velocity <= 0:
                continue
            previous = self.last_auto_goal_velocities.get(hardware_joint)
            if previous is not None and abs(int(previous) - raw_velocity) < threshold:
                continue
            try:
                self.robot.bus.write('Goal_Velocity', hardware_joint, raw_velocity, normalize=False, num_retry=3)
                self.last_auto_goal_velocities[hardware_joint] = raw_velocity
                self.get_logger().info(
                    f'auto Goal_Velocity {logical_joint}->{hardware_joint}: '
                    f'delta={target_deg - current_deg:+.2f} deg -> {raw_velocity}'
                )
            except Exception as exc:
                self.get_logger().warning(
                    f'failed auto Goal_Velocity for {logical_joint}->{hardware_joint}: {exc}'
                )

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
            self.observation_to_ros_radians(
                joint,
                observation.get(f'{self.hardware_joint_for(joint)}.pos', 0.0),
            )
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

            response.positions = [
                float(self.last_observation.get(f'{self.hardware_joint_for(joint)}.pos', 0.0))
                for joint in ALL_JOINTS
            ]
            response.success = True
            return response

    def write_joints_callback(self, request, response):
        units = str(self.get_parameter('command_units').value).lower().strip()
        targets = list(request.target_positions)

        active_joints = self.active_motor_joints()
        if len(targets) == 1 and bool(self.get_parameter('use_gripper').value):
            active_joints = ['gripper']

        action = {}
        command_items = []
        for joint, target in zip(active_joints, targets[: len(active_joints)]):
            target = float(target)
            if math.isnan(target):
                continue
            hardware_joint = self.hardware_joint_for(joint)
            action[f'{hardware_joint}.pos'] = self.command_to_lerobot_value(joint, float(target), units)
            command_items.append((joint, target))

        if not action:
            response.success = True
            return response

        with self.lock:
            if self.robot is None or not self.robot.is_connected:
                response.success = False
                return response

            try:
                self.apply_auto_goal_velocities(command_items, units)
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
    node = LeRobotSO101ProfiledBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
