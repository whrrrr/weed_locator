
#!/usr/bin/env python3
"""Control the ESP32 conveyor G-code and publish conveyor state."""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String


class ConveyorController(Node):
    """Small ROS wrapper for the original PC M200-M206 conveyor commands."""

    COMMANDS = {
        'left_low': ('M204', 'low', 'left'),
        'left_medium': ('M205', 'medium', 'left'),
        'left_mid': ('M205', 'medium', 'left'),
        'left_high': ('M206', 'high', 'left'),
        'right_low': ('M201', 'low', 'right'),
        'right_medium': ('M202', 'medium', 'right'),
        'right_mid': ('M202', 'medium', 'right'),
        'right_high': ('M203', 'high', 'right'),
        '向左转动-低速': ('M204', 'low', 'left'),
        '向左转动-中速': ('M205', 'medium', 'left'),
        '向左转动-高速': ('M206', 'high', 'left'),
        '向右转动-低速': ('M201', 'low', 'right'),
        '向右转动-中速': ('M202', 'medium', 'right'),
        '向右转动-高速': ('M203', 'high', 'right'),
    }
    STOP_WORDS = {'stop', '停止', 'M200', 'm200'}

    def __init__(self):
        super().__init__('conveyor_controller')

        self.gcode_pub = self.create_publisher(String, '/delta_arm/gcode_raw', 10)
        self.running_pub = self.create_publisher(Bool, '/conveyor/running', 10)
        self.speed_pub = self.create_publisher(String, '/conveyor/speed_mode', 10)
        self.direction_pub = self.create_publisher(String, '/conveyor/direction', 10)

        self.create_subscription(String, '/conveyor/cmd', self.on_cmd, 10)

        self.get_logger().info('conveyor_controller 已启动')
        self.get_logger().info('订阅: /conveyor/cmd，例如 left_low/right_medium/right_high/stop')
        self.get_logger().info('发布: /delta_arm/gcode_raw, /conveyor/running, /conveyor/speed_mode, /conveyor/direction')

    def on_cmd(self, msg: String):
        cmd = msg.data.strip()
        if cmd in self.STOP_WORDS:
            self._send_gcode('M200')
            self._publish_state(False)
            self.get_logger().info('传送带停止: M200')
            return

        item = self.COMMANDS.get(cmd)
        if item is None:
            self.get_logger().warn('未知传送带命令: %r' % cmd)
            return

        gcode, speed, direction = item
        self._send_gcode(gcode)
        self._publish_state(True, speed=speed, direction=direction)
        self.get_logger().info('传送带启动: %s speed=%s direction=%s' % (gcode, speed, direction))

    def _send_gcode(self, gcode: str):
        msg = String()
        msg.data = gcode
        self.gcode_pub.publish(msg)

    def _publish_state(self, running: bool, speed: str = '', direction: str = ''):
        running_msg = Bool()
        running_msg.data = running
        self.running_pub.publish(running_msg)

        if speed:
            speed_msg = String()
            speed_msg.data = speed
            self.speed_pub.publish(speed_msg)

        if direction:
            direction_msg = String()
            direction_msg.data = direction
            self.direction_pub.publish(direction_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ConveyorController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
