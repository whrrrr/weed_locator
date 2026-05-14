#!/home/whr/miniconda3/envs/lerobot/bin/python
"""Set one SO101 motor ID/baudrate using LeRobot's official motor setup path."""

import sys
from pathlib import Path

import rclpy
from rclpy.node import Node


LEROBOT_SRC = Path('/home/whr/lerobot/src')
if LEROBOT_SRC.exists():
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower


MOTOR_IDS = {
    'shoulder_pan': 1,
    'shoulder_lift': 2,
    'elbow_flex': 3,
    'wrist_flex': 4,
    'wrist_roll': 5,
    'gripper': 6,
}


class LeRobotSetupSingleMotor(Node):
    """Program one SO101 motor to the ID expected by its joint name."""

    def __init__(self):
        super().__init__('lerobot_setup_single_motor')

        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('joint', 'elbow_flex')
        self.declare_parameter('initial_id', -1)
        self.declare_parameter('initial_baudrate', -1)
        self.declare_parameter('confirm', False)

    def optional_positive_int(self, name):
        value = int(self.get_parameter(name).value)
        return None if value < 0 else value

    def run(self):
        joint = str(self.get_parameter('joint').value)
        if joint not in MOTOR_IDS:
            raise RuntimeError(f'joint must be one of {list(MOTOR_IDS)}')

        if not bool(self.get_parameter('confirm').value):
            raise RuntimeError(
                'refusing to write motor ID without confirm:=true. '
                'Disconnect all other motors and leave only the target motor on the bus.'
            )

        port = str(self.get_parameter('port').value)
        target_id = MOTOR_IDS[joint]
        initial_id = self.optional_positive_int('initial_id')
        initial_baudrate = self.optional_positive_int('initial_baudrate')

        self.get_logger().warning(
            f'setting the only connected motor on {port} to joint {joint} '
            f'(target ID {target_id})'
        )

        config = SO101FollowerConfig(port=port, id='setup_single_motor', use_degrees=True, cameras={})
        robot = SO101Follower(config)
        robot.bus.setup_motor(joint, initial_baudrate=initial_baudrate, initial_id=initial_id)
        self.get_logger().info(f'{joint} motor id set to {target_id}')


def main(args=None):
    rclpy.init(args=args)
    node = LeRobotSetupSingleMotor()
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
