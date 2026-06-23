#!/usr/bin/env python3
"""Keyboard controller for safe Delta hand-eye calibration waypoint setup."""

import math
import select
import sys
import termios
import time
import tty
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Bool, Empty


BOUNDARY_POINTS_XY = [
    (-40.0, 70.0),
    (-50.0, 50.0),
    (-60.0, 20.0),
    (-70.0, 10.0),
    (-80.0, 0.0),
    (-90.0, -10.0),
    (-110.0, -40.0),
    (0.0, -60.0),
    (110.0, -40.0),
    (90.0, -10.0),
    (80.0, 0.0),
    (70.0, 10.0),
    (60.0, 20.0),
    (50.0, 50.0),
    (40.0, 70.0),
    (0.0, 120.0),
]


def sort_boundary(points):
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx)), (cx, cy)


def shrink_polygon(points, margin):
    if margin <= 0.0:
        return list(points)
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    shrunk = []
    for x, y in points:
        dx = x - cx
        dy = y - cy
        dist = math.hypot(dx, dy)
        if dist <= margin:
            shrunk.append((x, y))
        else:
            scale = (dist - margin) / dist
            shrunk.append((cx + dx * scale, cy + dy * scale))
    return shrunk


def point_in_polygon(x, y, polygon):
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


class DeltaSafeCalibrationControl(Node):
    def __init__(self):
        super().__init__('delta_safe_calibration_control')

        self.declare_parameter(
            'waypoint_path',
            '/home/wyy/gpt_dev_ws/calibration_targets/delta_safe_9_waypoints.yaml',
        )
        self.declare_parameter('home_x_mm', 0.0)
        self.declare_parameter('home_y_mm', 0.0)
        self.declare_parameter('home_z_mm', 0.0)
        self.declare_parameter('safe_xy_z_mm', -210.0)
        self.declare_parameter('min_z_mm', -320.0)
        self.declare_parameter('max_z_mm', 0.0)
        self.declare_parameter('min_x_mm', -90.0)
        self.declare_parameter('max_x_mm', 90.0)
        self.declare_parameter('min_y_mm', -60.0)
        self.declare_parameter('max_y_mm', 100.0)
        self.declare_parameter('use_polygon_workspace', True)
        self.declare_parameter('polygon_margin_mm', 8.0)
        self.declare_parameter('jog_step_xy_mm', 5.0)
        self.declare_parameter('jog_step_z_mm', 5.0)
        self.declare_parameter('move_settle_sec', 1.0)
        self.declare_parameter('home_wait_sec', 4.0)
        self.declare_parameter('auto_run_delay_sec', 2.0)

        self.waypoint_path = Path(str(self.get_parameter('waypoint_path').value)).expanduser()
        self.safe_xy_z_mm = float(self.get_parameter('safe_xy_z_mm').value)
        self.min_z_mm = float(self.get_parameter('min_z_mm').value)
        self.max_z_mm = float(self.get_parameter('max_z_mm').value)
        self.min_x_mm = float(self.get_parameter('min_x_mm').value)
        self.max_x_mm = float(self.get_parameter('max_x_mm').value)
        self.min_y_mm = float(self.get_parameter('min_y_mm').value)
        self.max_y_mm = float(self.get_parameter('max_y_mm').value)
        self.use_polygon_workspace = bool(self.get_parameter('use_polygon_workspace').value)
        self.polygon_margin_mm = float(self.get_parameter('polygon_margin_mm').value)
        self.jog_step_xy_mm = float(self.get_parameter('jog_step_xy_mm').value)
        self.jog_step_z_mm = float(self.get_parameter('jog_step_z_mm').value)

        boundary, _center = sort_boundary(BOUNDARY_POINTS_XY)
        self.safe_polygon = shrink_polygon(boundary, self.polygon_margin_mm)

        self.current_xyz = [
            float(self.get_parameter('home_x_mm').value),
            float(self.get_parameter('home_y_mm').value),
            float(self.get_parameter('home_z_mm').value),
        ]
        self.waypoints = []

        self.move_pub = self.create_publisher(Point, '/delta_arm/move_to', 10)
        self.motor_pub = self.create_publisher(Bool, '/delta_arm/motor_enable', 10)
        self.home_pub = self.create_publisher(Empty, '/delta_arm/home', 10)

        self.load_waypoints(log_missing=False)
        self.print_help()

    def print_help(self):
        self.get_logger().info('Delta safe calibration control')
        self.get_logger().info('limits: box x=[%.1f, %.1f] y=[%.1f, %.1f] z=[%.1f, %.1f], safe_xy_z=%.1f, polygon=%s margin=%.1f'
                               % (self.min_x_mm, self.max_x_mm, self.min_y_mm, self.max_y_mm,
                                  self.min_z_mm, self.max_z_mm, self.safe_xy_z_mm,
                                  self.use_polygon_workspace, self.polygon_margin_mm))
        self.get_logger().info('keys:')
        self.get_logger().info('  e/d: motor enable/disable')
        self.get_logger().info('  0: G28 home, then current pose becomes home')
        self.get_logger().info('  U/O: Z up/down only; Z up is capped at max_z')
        self.get_logger().info('  I/K: Y +/-, J/L: X -/+; XY move first descends to safe_xy_z if needed')
        self.get_logger().info('  z/x: XY step -/+, c/v: Z step -/+')
        self.get_logger().info('  p: save current pose as next waypoint, max 9')
        self.get_logger().info('  1..9: move to saved waypoint')
        self.get_logger().info('  a: run waypoints 1..8 for translation calibration')
        self.get_logger().info('  R: move to waypoint 9 and hold for manual board rotation')
        self.get_logger().info('  y: list waypoints, o: write waypoints, r: reload, m: clear in memory')
        self.get_logger().info('  q: quit')

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

    def publish_motor(self, enabled):
        msg = Bool()
        msg.data = bool(enabled)
        self.motor_pub.publish(msg)
        self.get_logger().info('motor %s' % ('ENABLE' if enabled else 'DISABLE'))

    def publish_home(self):
        self.home_pub.publish(Empty())
        self.current_xyz = [
            float(self.get_parameter('home_x_mm').value),
            float(self.get_parameter('home_y_mm').value),
            float(self.get_parameter('home_z_mm').value),
        ]
        self.get_logger().info('G28 home sent; current pose reset to %s' % self.round_xyz(self.current_xyz))
        time.sleep(float(self.get_parameter('home_wait_sec').value))

    @staticmethod
    def round_xyz(xyz):
        return [round(float(v), 1) for v in xyz]

    def validate_target(self, xyz, allow_home=False):
        x, y, z = [float(v) for v in xyz]
        if allow_home:
            return True
        if not (self.min_x_mm <= x <= self.max_x_mm):
            self.get_logger().error('blocked: x=%.1f outside [%.1f, %.1f]' % (x, self.min_x_mm, self.max_x_mm))
            return False
        if not (self.min_y_mm <= y <= self.max_y_mm):
            self.get_logger().error('blocked: y=%.1f outside [%.1f, %.1f]' % (y, self.min_y_mm, self.max_y_mm))
            return False
        if not (self.min_z_mm <= z <= self.max_z_mm):
            self.get_logger().error('blocked: z=%.1f outside [%.1f, %.1f]' % (z, self.min_z_mm, self.max_z_mm))
            return False
        if self.use_polygon_workspace and not point_in_polygon(x, y, self.safe_polygon):
            self.get_logger().error('blocked: xy=(%.1f, %.1f) outside measured safe polygon' % (x, y))
            return False
        return True

    def publish_move_direct(self, xyz):
        msg = Point()
        msg.x, msg.y, msg.z = [float(v) for v in xyz]
        self.move_pub.publish(msg)
        self.current_xyz = [msg.x, msg.y, msg.z]
        self.get_logger().info('move -> x=%.1f y=%.1f z=%.1f' % (msg.x, msg.y, msg.z))

    def move_to(self, xyz):
        target = [float(v) for v in xyz]
        current = list(self.current_xyz)
        xy_changes = math.hypot(target[0] - current[0], target[1] - current[1]) > 1e-6

        if xy_changes and current[2] > self.safe_xy_z_mm:
            guard = [current[0], current[1], self.safe_xy_z_mm]
            if not self.validate_target(guard):
                return False
            self.get_logger().warning(
                'XY requested while z=%.1f is above safe_xy_z=%.1f; descending Z first'
                % (current[2], self.safe_xy_z_mm)
            )
            self.publish_move_direct(guard)
            time.sleep(float(self.get_parameter('move_settle_sec').value))

        if xy_changes and target[2] > self.safe_xy_z_mm:
            self.get_logger().warning(
                'XY target z=%.1f is above safe_xy_z=%.1f; using safe z'
                % (target[2], self.safe_xy_z_mm)
            )
            target[2] = self.safe_xy_z_mm

        if not self.validate_target(target):
            return False
        self.publish_move_direct(target)
        time.sleep(float(self.get_parameter('move_settle_sec').value))
        return True

    def jog(self, dx=0.0, dy=0.0, dz=0.0):
        target = [
            self.current_xyz[0] + dx,
            self.current_xyz[1] + dy,
            self.current_xyz[2] + dz,
        ]
        if dz > 0.0 and target[2] > self.max_z_mm:
            target[2] = self.max_z_mm
            self.get_logger().warning('Z up capped at max_z=%.1f' % self.max_z_mm)
        elif dz < 0.0 and target[2] < self.min_z_mm:
            target[2] = self.min_z_mm
            self.get_logger().warning('Z down capped at min_z=%.1f' % self.min_z_mm)
        self.move_to(target)

    def save_current_waypoint(self):
        if not self.validate_target(self.current_xyz):
            self.get_logger().error('current pose is not safe; waypoint not saved')
            return
        if self.current_xyz[2] > self.safe_xy_z_mm:
            self.get_logger().error(
                'waypoint not saved: z=%.1f is above safe_xy_z=%.1f'
                % (self.current_xyz[2], self.safe_xy_z_mm)
            )
            return
        if len(self.waypoints) >= 9:
            self.get_logger().error('already have 9 waypoints; press m to clear or edit yaml')
            return
        name = 'pt_%02d' % (len(self.waypoints) + 1)
        self.waypoints.append({'name': name, 'xyz': list(self.current_xyz)})
        self.get_logger().info('saved %s = %s' % (name, self.round_xyz(self.current_xyz)))

    def list_waypoints(self):
        if not self.waypoints:
            self.get_logger().info('no waypoints loaded/saved')
            return
        for index, wp in enumerate(self.waypoints, start=1):
            role = 'translation' if index <= 8 else 'rotation-hold'
            self.get_logger().info('%d %s %-13s %s' % (index, wp['name'], role, self.round_xyz(wp['xyz'])))

    def write_waypoints(self):
        self.waypoint_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'description': 'Nine manually selected safe Delta hand-eye calibration waypoints. Points 1-8 are translation samples; point 9 is manual rotation hold.',
            'safety': {
                'safe_xy_z_mm': self.safe_xy_z_mm,
                'min_z_mm': self.min_z_mm,
                'max_z_mm': self.max_z_mm,
                'box': {
                    'x_mm': [self.min_x_mm, self.max_x_mm],
                    'y_mm': [self.min_y_mm, self.max_y_mm],
                },
                'use_polygon_workspace': self.use_polygon_workspace,
                'polygon_margin_mm': self.polygon_margin_mm,
            },
            'waypoints': [
                {
                    'name': wp['name'],
                    'x_mm': float(wp['xyz'][0]),
                    'y_mm': float(wp['xyz'][1]),
                    'z_mm': float(wp['xyz'][2]),
                }
                for wp in self.waypoints
            ],
        }
        with self.waypoint_path.open('w', encoding='utf-8') as file:
            yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
        self.get_logger().info('wrote %d waypoints to %s' % (len(self.waypoints), self.waypoint_path))

    def load_waypoints(self, log_missing=True):
        if not self.waypoint_path.exists():
            if log_missing:
                self.get_logger().warning('waypoint file not found: %s' % self.waypoint_path)
            return False
        data = yaml.safe_load(self.waypoint_path.read_text(encoding='utf-8')) or {}
        loaded = []
        for index, item in enumerate(data.get('waypoints', []), start=1):
            xyz = [float(item['x_mm']), float(item['y_mm']), float(item['z_mm'])]
            if xyz[2] > self.safe_xy_z_mm:
                self.get_logger().warning('skipping waypoint #%d above safe_xy_z: %s' % (index, xyz))
                continue
            if not self.validate_target(xyz):
                self.get_logger().warning('skipping unsafe waypoint #%d: %s' % (index, xyz))
                continue
            loaded.append({'name': str(item.get('name', 'pt_%02d' % index)), 'xyz': xyz})
        self.waypoints = loaded[:9]
        self.get_logger().info('loaded %d waypoints from %s' % (len(self.waypoints), self.waypoint_path))
        return True

    def move_to_waypoint(self, index):
        if index < 1 or index > len(self.waypoints):
            self.get_logger().error('waypoint %d not available; saved count=%d' % (index, len(self.waypoints)))
            return False
        wp = self.waypoints[index - 1]
        self.get_logger().info('moving to waypoint %d %s %s' % (index, wp['name'], self.round_xyz(wp['xyz'])))
        return self.move_to(wp['xyz'])

    def run_translation_waypoints(self):
        if len(self.waypoints) < 8:
            self.get_logger().error('need at least 8 waypoints for translation calibration, got %d' % len(self.waypoints))
            return
        for index in range(1, 9):
            if not rclpy.ok():
                break
            self.move_to_waypoint(index)
            time.sleep(float(self.get_parameter('auto_run_delay_sec').value))

    def move_to_rotation_hold(self):
        if not self.move_to_waypoint(9):
            return
        self.publish_motor(True)
        self.get_logger().info('holding waypoint 9 for manual board rotation')

    def run(self):
        if not sys.stdin.isatty():
            self.get_logger().error('stdin is not a TTY; run this with ros2 run in a terminal, not inside ros2 launch')
            return
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            key = self.get_key()
            if key is None:
                continue
            if key == 'e':
                self.publish_motor(True)
            elif key == 'd':
                self.publish_motor(False)
            elif key == '0':
                self.publish_home()
            elif key == 'I':
                self.jog(dy=self.jog_step_xy_mm)
            elif key == 'K':
                self.jog(dy=-self.jog_step_xy_mm)
            elif key == 'J':
                self.jog(dx=-self.jog_step_xy_mm)
            elif key == 'L':
                self.jog(dx=self.jog_step_xy_mm)
            elif key == 'U':
                self.jog(dz=self.jog_step_z_mm)
            elif key == 'O':
                self.jog(dz=-self.jog_step_z_mm)
            elif key == 'z':
                self.jog_step_xy_mm = max(1.0, self.jog_step_xy_mm - 1.0)
                self.get_logger().info('jog_step_xy=%.1f' % self.jog_step_xy_mm)
            elif key == 'x':
                self.jog_step_xy_mm += 1.0
                self.get_logger().info('jog_step_xy=%.1f' % self.jog_step_xy_mm)
            elif key == 'c':
                self.jog_step_z_mm = max(1.0, self.jog_step_z_mm - 1.0)
                self.get_logger().info('jog_step_z=%.1f' % self.jog_step_z_mm)
            elif key == 'v':
                self.jog_step_z_mm += 1.0
                self.get_logger().info('jog_step_z=%.1f' % self.jog_step_z_mm)
            elif key == 'p':
                self.save_current_waypoint()
            elif key == 'y':
                self.list_waypoints()
            elif key == 'o':
                self.write_waypoints()
            elif key == 'r':
                self.load_waypoints()
            elif key == 'm':
                self.waypoints = []
                self.get_logger().info('cleared in-memory waypoints')
            elif key in '123456789':
                self.move_to_waypoint(int(key))
            elif key == 'a':
                self.run_translation_waypoints()
            elif key == 'R':
                self.move_to_rotation_hold()
            elif key == 'q' or key == '\x03':
                break


def main(args=None):
    rclpy.init(args=args)
    node = DeltaSafeCalibrationControl()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
