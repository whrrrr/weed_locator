#!/usr/bin/env python3
"""Schedule Delta pick targets from camera pixel detections and conveyor speed."""

import time
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Bool, String


@dataclass
class ScheduledTarget:
    x: float
    y: float
    z: float
    due_time: float


class ConveyorPickScheduler(Node):
    """ROS version of the original PC camera/conveyor prediction queue."""

    SPEED_ALIASES = {
        'low': 'low',
        'medium': 'medium',
        'mid': 'medium',
        'high': 'high',
        '低速': 'low',
        '中速': 'medium',
        '高速': 'high',
    }

    def __init__(self):
        super().__init__('conveyor_pick_scheduler')

        self.declare_parameter('image_width', 640.0)
        self.declare_parameter('top_line_y', 45.0)
        self.declare_parameter('bottom_line_y', 460.0)
        self.declare_parameter('belt_width_mm', 100.0)
        self.declare_parameter('pick_z', -180.0)
        self.declare_parameter('running', False)
        self.declare_parameter('speed_mode', 'medium')
        self.declare_parameter('min_add_interval_sec', 1.0)
        self.declare_parameter('rearm_x_mm', 10.0)
        self.declare_parameter('timer_period_sec', 0.1)
        self.declare_parameter('due_tolerance_sec', 0.2)

        # Same empirical values as CameraDetect.py.
        self.declare_parameter('low_trigger_mm', 1.5)
        self.declare_parameter('medium_trigger_mm', 3.5)
        self.declare_parameter('high_trigger_mm', 7.5)
        self.declare_parameter('trigger_use_abs_x', True)
        self.declare_parameter('low_delay_sec', 122.22)
        self.declare_parameter('medium_delay_sec', 48.88)
        self.declare_parameter('high_delay_sec', 22.5)

        self.running = bool(self.get_parameter('running').value)
        self.speed_mode = self._normalize_speed(str(self.get_parameter('speed_mode').value))
        self.last_add_time = 0.0
        self.trigger_armed = True
        self.queue = []

        self.pick_pub = self.create_publisher(Point, '/delta_arm/pick_target_raw', 10)
        self.queue_pub = self.create_publisher(String, '/conveyor_pick_scheduler/queue', 10)
        self.camera_mm_pub = self.create_publisher(Point, '/weed_detector/camera_mm', 10)

        self.create_subscription(Point, '/weed_detector/pixel_center', self.on_pixel_center, 10)
        self.create_subscription(Bool, '/conveyor/running', self.on_running, 10)
        self.create_subscription(String, '/conveyor/speed_mode', self.on_speed_mode, 10)

        period = float(self.get_parameter('timer_period_sec').value)
        self.create_timer(period, self.on_timer)

        self.get_logger().info('conveyor_pick_scheduler 已启动')
        self.get_logger().info('订阅: /weed_detector/pixel_center, /conveyor/running, /conveyor/speed_mode')
        self.get_logger().info('发布: /delta_arm/pick_target_raw, /conveyor_pick_scheduler/queue')

    def on_running(self, msg: Bool):
        self.running = bool(msg.data)
        self.get_logger().info(f'传送带状态: {"运行" if self.running else "停止"}')

    def on_speed_mode(self, msg: String):
        speed = self._normalize_speed(msg.data)
        if speed is None:
            self.get_logger().warn(f'未知速度模式: {msg.data!r}, 可用: low/medium/high 或 低速/中速/高速')
            return
        self.speed_mode = speed
        self.get_logger().info(f'传送带速度: {self.speed_mode}')

    def on_pixel_center(self, msg: Point):
        """Input point uses x/y as image pixel center. z is ignored."""
        mapped = self._pixel_to_mm(msg.x, msg.y)
        if mapped is None:
            return

        left_move_x_mm, left_move_y_mm = mapped
        self.publish_camera_mm(left_move_x_mm, left_move_y_mm)

        if not self.running:
            return

        now = time.time()
        min_interval = float(self.get_parameter('min_add_interval_sec').value)
        if now - self.last_add_time < min_interval:
            return

        trigger = self._trigger_width_mm()
        trigger_x_mm = abs(left_move_x_mm) if bool(self.get_parameter('trigger_use_abs_x').value) else left_move_x_mm
        rearm_x_mm = float(self.get_parameter('rearm_x_mm').value)
        if trigger_x_mm >= rearm_x_mm:
            self.trigger_armed = True

        if not (0.0 < trigger_x_mm < trigger):
            return
        if not self.trigger_armed:
            return

        delay = self._delay_sec()
        target = ScheduledTarget(
            x=0.0,
            y=left_move_y_mm,
            z=float(self.get_parameter('pick_z').value),
            due_time=now + delay,
        )
        self.queue.append(target)
        self.trigger_armed = False
        self.last_add_time = now
        self.get_logger().info(
            '目标入队: pixel=(%.1f, %.1f), camera_mm=(%.2f, %.2f), trigger_x=%.2f, %.2fs 后触发, pick=(%.1f, %.1f, %.1f)'
            % (msg.x, msg.y, left_move_x_mm, left_move_y_mm, trigger_x_mm, delay, target.x, target.y, target.z)
        )
        self._publish_queue_status()

    def on_timer(self):
        if not self.queue:
            return

        now = time.time()
        due_tolerance = float(self.get_parameter('due_tolerance_sec').value)
        remaining = []

        for target in self.queue:
            left = target.due_time - now
            if 0.0 <= left < due_tolerance or left < 0.0:
                msg = Point()
                msg.x = target.x
                msg.y = target.y
                msg.z = target.z
                self.pick_pub.publish(msg)
                self.get_logger().info('触发抓取: x=%.1f y=%.1f z=%.1f' % (msg.x, msg.y, msg.z))
            else:
                remaining.append(target)

        if len(remaining) != len(self.queue):
            self.queue = remaining
            self._publish_queue_status()

    def _pixel_to_mm(self, x_pixel: float, y_pixel: float):
        image_width = float(self.get_parameter('image_width').value)
        top_line_y = float(self.get_parameter('top_line_y').value)
        bottom_line_y = float(self.get_parameter('bottom_line_y').value)
        belt_width_mm = float(self.get_parameter('belt_width_mm').value)

        pixel_span = bottom_line_y - top_line_y
        if pixel_span <= 0.0:
            self.get_logger().error('bottom_line_y 必须大于 top_line_y')
            return None

        zero_x = image_width / 2.0
        zero_y = (top_line_y + bottom_line_y) / 2.0
        image_slide_rate = belt_width_mm / pixel_span

        left_move_x_mm = (x_pixel - zero_x) * image_slide_rate
        left_move_y_mm = (y_pixel - zero_y) * image_slide_rate
        return left_move_x_mm, left_move_y_mm

    def publish_camera_mm(self, x_mm: float, y_mm: float):
        msg = Point()
        msg.x = float(x_mm)
        msg.y = float(y_mm)
        msg.z = 0.0
        self.camera_mm_pub.publish(msg)

    def _trigger_width_mm(self) -> float:
        return float(self.get_parameter(f'{self.speed_mode}_trigger_mm').value)

    def _delay_sec(self) -> float:
        return float(self.get_parameter(f'{self.speed_mode}_delay_sec').value)

    def _normalize_speed(self, speed):
        return self.SPEED_ALIASES.get(str(speed).strip())

    def _publish_queue_status(self):
        now = time.time()
        lines = []
        for target in self.queue:
            lines.append('%.1fs -> (%.1f, %.1f, %.1f)' % (target.due_time - now, target.x, target.y, target.z))
        msg = String()
        msg.data = '\n'.join(lines)
        self.queue_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ConveyorPickScheduler()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
