#!/usr/bin/env python3
"""Move the Delta arm along the measured workspace boundary."""

import math
import time

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Bool, Empty


BOUNDARY_POINTS = [
    (-40.0, 70.0, -180.0),
    (-50.0, 50.0, -180.0),
    (-60.0, 20.0, -180.0),
    (-70.0, 10.0, -180.0),
    (-80.0, 0.0, -180.0),
    (-90.0, -10.0, -180.0),
    (40.0, 70.0, -180.0),
    (50.0, 50.0, -180.0),
    (60.0, 20.0, -180.0),
    (70.0, 10.0, -180.0),
    (80.0, 0.0, -180.0),
    (90.0, -10.0, -180.0),
    (110.0, -40.0, -180.0),
    (0.0, 120.0, -180.0),
    (0.0, -60.0, -180.0),
    (-110.0, -40.0, -180.0),
]


def sort_boundary(points):
    xy = [(x, y) for x, y, _ in points]
    cx = sum(x for x, _ in xy) / len(xy)
    cy = sum(y for _, y in xy) / len(xy)
    sorted_xy = sorted(xy, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return sorted_xy, (cx, cy)


def shrink_boundary(points, center, margin):
    cx, cy = center
    out = []
    for x, y in points:
        dx = x - cx
        dy = y - cy
        dist = math.hypot(dx, dy)
        if dist <= margin:
            out.append((cx, cy))
        else:
            ratio = (dist - margin) / dist
            out.append((cx + dx * ratio, cy + dy * ratio))
    return out


class WorkspaceBoundaryRunner(Node):
    def __init__(self):
        super().__init__('workspace_boundary_runner')

        self.declare_parameter('z', -180.0)
        self.declare_parameter('safety_margin', 10.0)
        self.declare_parameter('point_delay_sec', 1.5)
        self.declare_parameter('home_first', True)
        self.declare_parameter('home_wait_sec', 3.0)
        self.declare_parameter('repeat', False)
        self.declare_parameter('close_loop', True)

        self.z = float(self.get_parameter('z').value)
        self.safety_margin = float(self.get_parameter('safety_margin').value)
        self.point_delay_sec = float(self.get_parameter('point_delay_sec').value)
        self.home_first = bool(self.get_parameter('home_first').value)
        self.home_wait_sec = float(self.get_parameter('home_wait_sec').value)
        self.repeat = bool(self.get_parameter('repeat').value)
        self.close_loop = bool(self.get_parameter('close_loop').value)

        boundary, center = sort_boundary(BOUNDARY_POINTS)
        self.path = shrink_boundary(boundary, center, self.safety_margin)
        if self.close_loop and self.path:
            self.path.append(self.path[0])

        self.move_pub = self.create_publisher(Point, '/delta_arm/move_to', 10)
        self.home_pub = self.create_publisher(Empty, '/delta_arm/home', 10)
        self.motor_pub = self.create_publisher(Bool, '/delta_arm/motor_enable', 10)

        self.get_logger().info('workspace_boundary_runner 已启动')
        self.get_logger().info(
            f'z={self.z:.1f}, safety_margin={self.safety_margin:.1f}, points={len(self.path)}'
        )

    def publish_motor(self, enabled=True):
        msg = Bool()
        msg.data = enabled
        self.motor_pub.publish(msg)
        self.get_logger().info(f'motor {"ENABLE" if enabled else "DISABLE"}')

    def publish_home(self):
        self.home_pub.publish(Empty())
        self.get_logger().info('发送回零')

    def publish_move(self, x, y, z):
        msg = Point()
        msg.x = float(x)
        msg.y = float(y)
        msg.z = float(z)
        self.move_pub.publish(msg)
        self.get_logger().info(f'边界点: x={x:.1f}, y={y:.1f}, z={z:.1f}')

    def run_once(self):
        for x, y in self.path:
            if not rclpy.ok():
                return
            self.publish_move(x, y, self.z)
            time.sleep(self.point_delay_sec)

    def run(self):
        self.publish_motor(True)
        time.sleep(0.5)

        if self.home_first:
            self.publish_home()
            time.sleep(self.home_wait_sec)

        while rclpy.ok():
            self.run_once()
            if not self.repeat:
                break

        self.get_logger().info('边界运动测试结束')


def main(args=None):
    rclpy.init(args=args)
    node = WorkspaceBoundaryRunner()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
