#!/usr/bin/env python3
"""Delta 机械臂键盘点动测试工具."""

import select
import sys
import termios
import time
import tty

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Bool, Empty


class DeltaKeyboardJog(Node):
    def __init__(self):
        super().__init__('delta_keyboard_jog')

        self.declare_parameter('home_x', 0.0)
        self.declare_parameter('home_y', 0.0)
        self.declare_parameter('home_z', -140.0)
        self.declare_parameter('step_xy', 10.0)
        self.declare_parameter('step_z', 5.0)
        self.declare_parameter('home_wait_sec', 3.0)
        self.declare_parameter('command_wait_sec', 0.4)

        self.home_x = float(self.get_parameter('home_x').value)
        self.home_y = float(self.get_parameter('home_y').value)
        self.home_z = float(self.get_parameter('home_z').value)
        self.step_xy = float(self.get_parameter('step_xy').value)
        self.step_z = float(self.get_parameter('step_z').value)
        self.home_wait_sec = float(self.get_parameter('home_wait_sec').value)
        self.command_wait_sec = float(self.get_parameter('command_wait_sec').value)

        self.current_x = self.home_x
        self.current_y = self.home_y
        self.current_z = self.home_z

        self.move_pub = self.create_publisher(Point, '/delta_arm/move_to', 10)
        self.home_pub = self.create_publisher(Empty, '/delta_arm/home', 10)
        self.motor_pub = self.create_publisher(Bool, '/delta_arm/motor_enable', 10)

        self.print_help()

    def print_help(self):
        self.get_logger().info('Delta 键盘点动测试已启动')
        self.get_logger().info('启动后将先开电机并回零')
        self.get_logger().info('按键说明:')
        self.get_logger().info('  w/s: Y- / Y+')
        self.get_logger().info('  a/d: X- / X+')
        self.get_logger().info('  r/f: Z+ / Z-')
        self.get_logger().info('  z/x: XY步长 -/+')
        self.get_logger().info('  c/v: Z步长 -/+')
        self.get_logger().info('  h: 回零')
        self.get_logger().info('  p: 打印当前坐标')
        self.get_logger().info('  q: 退出')

    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            if select.select([sys.stdin], [], [], 0.1)[0]:
                return sys.stdin.read(1)
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def publish_motor(self, enabled: bool):
        msg = Bool()
        msg.data = enabled
        self.motor_pub.publish(msg)
        self.get_logger().info(f'motor {"ENABLE" if enabled else "DISABLE"}')

    def publish_home(self):
        self.home_pub.publish(Empty())
        self.current_x = self.home_x
        self.current_y = self.home_y
        self.current_z = self.home_z
        self.get_logger().info('发送回零')

    def publish_move(self):
        msg = Point()
        msg.x = self.current_x
        msg.y = self.current_y
        msg.z = self.current_z
        self.move_pub.publish(msg)
        self.get_logger().info(
            f'发送目标: x={self.current_x:.1f}, y={self.current_y:.1f}, z={self.current_z:.1f}'
        )

    def print_pose(self):
        self.get_logger().info(
            f'当前记录坐标: x={self.current_x:.1f}, y={self.current_y:.1f}, z={self.current_z:.1f} | step_xy={self.step_xy:.1f}, step_z={self.step_z:.1f}'
        )

    def startup_sequence(self):
        self.publish_motor(True)
        time.sleep(self.command_wait_sec)
        self.publish_home()
        time.sleep(self.home_wait_sec)
        self.print_pose()

    def run(self):
        self.startup_sequence()

        while rclpy.ok():
            key = self.get_key()
            if key is None:
                continue

            moved = False
            if key == 'w':
                self.current_y -= self.step_xy
                moved = True
            elif key == 's':
                self.current_y += self.step_xy
                moved = True
            elif key == 'a':
                self.current_x -= self.step_xy
                moved = True
            elif key == 'd':
                self.current_x += self.step_xy
                moved = True
            elif key == 'r':
                self.current_z += self.step_z
                moved = True
            elif key == 'f':
                self.current_z -= self.step_z
                moved = True
            elif key == 'z':
                self.step_xy = max(1.0, self.step_xy - 1.0)
                self.print_pose()
            elif key == 'x':
                self.step_xy += 1.0
                self.print_pose()
            elif key == 'c':
                self.step_z = max(1.0, self.step_z - 1.0)
                self.print_pose()
            elif key == 'v':
                self.step_z += 1.0
                self.print_pose()
            elif key == 'h':
                self.publish_home()
                time.sleep(self.home_wait_sec)
                self.print_pose()
            elif key == 'p':
                self.print_pose()
            elif key == 'q' or key == '\x03':
                self.get_logger().info('退出点动测试')
                break

            if moved:
                self.publish_move()
                time.sleep(self.command_wait_sec)
                self.print_pose()


def main(args=None):
    rclpy.init(args=args)
    node = DeltaKeyboardJog()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
