#!/usr/bin/env python3
"""Preview or execute an existing Delta waypoint YAML path."""

import math
import time
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Bool, Empty


def load_waypoints(path):
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    waypoints = []
    for index, item in enumerate(data.get('waypoints', []), start=1):
        try:
            waypoints.append(
                {
                    'name': str(item.get('name', f'pt_{index:03d}')),
                    'x': float(item['x_mm']),
                    'y': float(item['y_mm']),
                    'z': float(item['z_mm']),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'invalid waypoint #{index}: {item}') from exc
    return waypoints


def distance(a, b):
    dx = b['x'] - a['x']
    dy = b['y'] - a['y']
    dz = b['z'] - a['z']
    return math.sqrt(dx * dx + dy * dy + dz * dz), math.hypot(dx, dy), abs(dz)


def waypoint_slice(waypoints, start_index, end_index):
    start = max(1, int(start_index)) - 1
    end = int(end_index)
    if end <= 0:
        end = len(waypoints)
    end = min(end, len(waypoints))
    if start >= end:
        return []
    return waypoints[start:end]


class DeltaWaypointRunner(Node):
    def __init__(self):
        super().__init__('delta_waypoint_runner')

        self.declare_parameter(
            'waypoint_path',
            '/home/whr/cc_ws/tros_ws/calibration_targets/delta_refined_layered_waypoints_ordered.yaml',
        )
        self.declare_parameter(
            'preview_svg_path',
            '/home/whr/cc_ws/tros_ws/calibration_targets/delta_waypoint_runner_preview.svg',
        )
        self.declare_parameter('execute', False)
        self.declare_parameter('start_index', 1)
        self.declare_parameter('end_index', 0)
        self.declare_parameter('point_delay_sec', 2.0)
        self.declare_parameter('enable_motor', True)
        self.declare_parameter('home_first', False)
        self.declare_parameter('home_wait_sec', 4.0)
        self.declare_parameter('subscriber_wait_sec', 3.0)
        self.declare_parameter('home_publish_count', 2)
        self.declare_parameter('max_allowed_jump_mm', 60.0)
        self.declare_parameter('move_topic', '/delta_arm/move_to')
        self.declare_parameter('motor_topic', '/delta_arm/motor_enable')
        self.declare_parameter('home_topic', '/delta_arm/home')

        self.move_pub = self.create_publisher(Point, str(self.get_parameter('move_topic').value), 10)
        self.motor_pub = self.create_publisher(Bool, str(self.get_parameter('motor_topic').value), 10)
        self.home_pub = self.create_publisher(Empty, str(self.get_parameter('home_topic').value), 10)

        self.timer = self.create_timer(0.1, self.run_once)
        self.has_run = False
        self.done = False

    def run_once(self):
        if self.has_run:
            return
        self.has_run = True

        waypoint_path = Path(str(self.get_parameter('waypoint_path').value)).expanduser()
        preview_path = Path(str(self.get_parameter('preview_svg_path').value)).expanduser()
        execute = bool(self.get_parameter('execute').value)
        start_index = int(self.get_parameter('start_index').value)
        end_index = int(self.get_parameter('end_index').value)
        max_allowed_jump = float(self.get_parameter('max_allowed_jump_mm').value)

        waypoints = waypoint_slice(load_waypoints(waypoint_path), start_index, end_index)
        if not waypoints:
            self.get_logger().error('no waypoints selected')
            self.done = True
            return

        stats = self.report_path(waypoints)
        self.write_preview_svg(waypoints, preview_path)
        self.get_logger().info(f'loaded waypoints: {waypoint_path}')
        self.get_logger().info(f'preview svg: {preview_path}')

        if execute and stats['max_d3'] > max_allowed_jump:
            self.get_logger().error(
                'execute refused: max jump %.1f mm exceeds max_allowed_jump_mm=%.1f. Use an ordered path or raise the limit deliberately.'
                % (stats['max_d3'], max_allowed_jump)
            )
            self.done = True
            return

        if execute:
            self.execute_path(waypoints)
        else:
            self.get_logger().info('execute=false, preview/report only')

        self.done = True

    def report_path(self, waypoints):
        by_z = {}
        for waypoint in waypoints:
            by_z.setdefault(int(round(waypoint['z'])), []).append(waypoint)

        self.get_logger().info('selected waypoints: %d' % len(waypoints))
        for z in sorted(by_z.keys(), reverse=True):
            pts = by_z[z]
            xs = [p['x'] for p in pts]
            ys = [p['y'] for p in pts]
            self.get_logger().info(
                'layer z=%d: n=%d, x=[%.1f, %.1f], y=[%.1f, %.1f]'
                % (z, len(pts), min(xs), max(xs), min(ys), max(ys))
            )

        jumps = []
        for index, (a, b) in enumerate(zip(waypoints, waypoints[1:]), start=1):
            d3, dxy, dz = distance(a, b)
            jumps.append((d3, dxy, dz, index, a, b))

        if not jumps:
            return {'max_d3': 0.0, 'max_dxy': 0.0, 'max_dz': 0.0}

        max_d3 = max(j[0] for j in jumps)
        max_dxy = max(j[1] for j in jumps)
        max_dz = max(j[2] for j in jumps)
        self.get_logger().info('jump summary: max_3d=%.1f mm, max_xy=%.1f mm, max_z=%.1f mm' % (max_d3, max_dxy, max_dz))
        for d3, dxy, dz, index, a, b in sorted(jumps, reverse=True)[:10]:
            self.get_logger().info(
                'jump %03d->%03d: d3=%.1f dxy=%.1f dz=%.1f from=(%.1f, %.1f, %.1f) to=(%.1f, %.1f, %.1f)'
                % (index, index + 1, d3, dxy, dz, a['x'], a['y'], a['z'], b['x'], b['y'], b['z'])
            )
        return {'max_d3': max_d3, 'max_dxy': max_dxy, 'max_dz': max_dz}

    def execute_path(self, waypoints):
        point_delay = float(self.get_parameter('point_delay_sec').value)
        self.wait_for_subscriber(self.move_pub, 'move topic')
        if bool(self.get_parameter('enable_motor').value):
            self.wait_for_subscriber(self.motor_pub, 'motor topic')
            msg = Bool()
            msg.data = True
            self.motor_pub.publish(msg)
            self.get_logger().info('motor ENABLE')
            time.sleep(0.5)

        if bool(self.get_parameter('home_first').value):
            self.wait_for_subscriber(self.home_pub, 'home topic')
            count = max(1, int(self.get_parameter('home_publish_count').value))
            for _ in range(count):
                self.home_pub.publish(Empty())
                time.sleep(0.1)
            self.get_logger().info('home command sent x%d' % count)
            time.sleep(float(self.get_parameter('home_wait_sec').value))

        for index, waypoint in enumerate(waypoints, start=1):
            if not rclpy.ok():
                break
            msg = Point()
            msg.x = waypoint['x']
            msg.y = waypoint['y']
            msg.z = waypoint['z']
            self.move_pub.publish(msg)
            self.get_logger().info(
                'move %03d/%03d: %s -> x=%.1f y=%.1f z=%.1f'
                % (index, len(waypoints), waypoint['name'], msg.x, msg.y, msg.z)
            )
            time.sleep(max(0.0, point_delay))

        self.get_logger().info('waypoint path finished')

    def wait_for_subscriber(self, publisher, label):
        deadline = time.time() + max(0.0, float(self.get_parameter('subscriber_wait_sec').value))
        while rclpy.ok() and publisher.get_subscription_count() == 0 and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        count = publisher.get_subscription_count()
        if count == 0:
            self.get_logger().warning(f'{label}: no subscribers matched before command publish')
        else:
            self.get_logger().info(f'{label}: subscribers={count}')

    def write_preview_svg(self, waypoints, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        xs = [p['x'] for p in waypoints]
        ys = [p['y'] for p in waypoints]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad = 30.0
        width = 900
        height = 760
        left = 70
        top = 70
        plot_w = width - 140
        plot_h = height - 150

        span_x = max(1.0, max_x - min_x + 2.0 * pad)
        span_y = max(1.0, max_y - min_y + 2.0 * pad)
        origin_x = min_x - pad
        origin_y = min_y - pad

        def sx(x):
            return left + (x - origin_x) / span_x * plot_w

        def sy(y):
            return top + plot_h - (y - origin_y) / span_y * plot_h

        colors = {
            -180: '#1f77b4',
            -190: '#2ca02c',
            -200: '#ff7f0e',
            -210: '#d62728',
            -220: '#9467bd',
            -230: '#8c564b',
        }

        parts = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">' % (width, height, width, height),
            '<style>text{font-family:Arial,sans-serif;font-size:13px}.small{font-size:11px}.axis{stroke:#aaa;stroke-width:1}.path{fill:none;stroke:#222;stroke-width:1.2;opacity:.45}.pt{stroke:#111;stroke-width:.6}</style>',
            '<rect width="100%" height="100%" fill="#fff"/>',
            '<text x="24" y="32">Delta waypoint runner preview</text>',
            '<text class="small" x="24" y="52">points=%d, x=[%.1f, %.1f], y=[%.1f, %.1f]</text>' % (len(waypoints), min_x, max_x, min_y, max_y),
            '<rect x="%d" y="%d" width="%d" height="%d" fill="#fafafa" stroke="#ccc"/>' % (left, top, plot_w, plot_h),
            '<line class="axis" x1="%g" y1="%g" x2="%g" y2="%g"/>' % (sx(0.0), top, sx(0.0), top + plot_h),
            '<line class="axis" x1="%g" y1="%g" x2="%g" y2="%g"/>' % (left, sy(0.0), left + plot_w, sy(0.0)),
        ]

        path_points = ' '.join('%.1f,%.1f' % (sx(p['x']), sy(p['y'])) for p in waypoints)
        parts.append('<polyline class="path" points="%s"/>' % path_points)
        for index, point in enumerate(waypoints, start=1):
            color = colors.get(int(round(point['z'])), '#555')
            parts.append(
                '<circle class="pt" cx="%.1f" cy="%.1f" r="4.2" fill="%s"><title>%03d %s x=%.1f y=%.1f z=%.1f</title></circle>'
                % (sx(point['x']), sy(point['y']), color, index, point['name'], point['x'], point['y'], point['z'])
            )
            if index in (1, len(waypoints)):
                parts.append('<text class="small" x="%.1f" y="%.1f">%s</text>' % (sx(point['x']) + 6, sy(point['y']) - 6, 'start' if index == 1 else 'end'))

        legend_x = left
        legend_y = height - 55
        for offset, z in enumerate(sorted(set(int(round(p['z'])) for p in waypoints), reverse=True)):
            x = legend_x + offset * 105
            color = colors.get(z, '#555')
            parts.append('<rect x="%d" y="%d" width="14" height="14" fill="%s" stroke="#111"/>' % (x, legend_y, color))
            parts.append('<text class="small" x="%d" y="%d">z=%d</text>' % (x + 20, legend_y + 12, z))

        parts.append('</svg>')
        path.write_text('\n'.join(parts) + '\n', encoding='utf-8')


def main(args=None):
    rclpy.init(args=args)
    node = DeltaWaypointRunner()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
