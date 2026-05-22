#!/usr/bin/env python3
"""Generate and optionally run Delta calibration waypoints from workspace slices."""

import math
import time
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Bool, Empty


def parse_z_list(text):
    return [int(round(float(item.strip()))) for item in str(text).split(',') if item.strip()]


def convex_hull(points_xy):
    points = sorted(set((float(x), float(y)) for x, y in points_xy))
    if len(points) <= 1:
        return points

    def cross(origin, a_pt, b_pt):
        return (a_pt[0] - origin[0]) * (b_pt[1] - origin[1]) - (a_pt[1] - origin[1]) * (b_pt[0] - origin[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def point_on_segment(point, a_pt, b_pt, eps=1e-6):
    px, py = point
    ax, ay = a_pt
    bx, by = b_pt
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > eps:
        return False
    dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
    return dot <= eps


def point_in_polygon(point, polygon):
    if len(polygon) < 3:
        return False
    if any(point_on_segment(point, polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon))):
        return True

    x, y = point
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if (y1 > y) != (y2 > y):
            intersect_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intersect_x:
                inside = not inside
    return inside


def generate_grid(hull, z_mm, step_mm):
    xs = [point[0] for point in hull]
    ys = [point[1] for point in hull]
    x_start = math.ceil(min(xs) / step_mm) * step_mm
    y_start = math.ceil(min(ys) / step_mm) * step_mm
    x_end = max(xs)
    y_end = max(ys)

    waypoints = []
    row = 0
    y = y_start
    while y <= y_end + 1e-6:
        row_points = []
        x = x_start
        while x <= x_end + 1e-6:
            if point_in_polygon((x, y), hull):
                row_points.append((float(x), float(y), float(z_mm)))
            x += step_mm
        if row % 2 == 1:
            row_points.reverse()
        waypoints.extend(row_points)
        y += step_mm
        row += 1
    return waypoints


class DeltaWorkspaceAutoRunner(Node):
    def __init__(self):
        super().__init__('delta_workspace_auto_runner')

        self.declare_parameter('boundary_path', '/home/whr/cc_ws/tros_ws/calibration_targets/delta_workspace_slices.yaml')
        self.declare_parameter('waypoint_path', '/home/whr/cc_ws/tros_ws/calibration_targets/delta_auto_waypoints.yaml')
        self.declare_parameter('preview_svg_path', '/home/whr/cc_ws/tros_ws/calibration_targets/delta_auto_waypoints_preview.svg')
        self.declare_parameter('shallow_source_z_mm', -180.0)
        self.declare_parameter('deep_source_z_mm', -210.0)
        self.declare_parameter('shallow_target_zs_mm', '-180,-190,-200')
        self.declare_parameter('deep_target_zs_mm', '-210,-220,-230')
        self.declare_parameter('shallow_step_xy_mm', 45.0)
        self.declare_parameter('deep_step_xy_mm', 45.0)
        self.declare_parameter('execute', False)
        self.declare_parameter('home_first', False)
        self.declare_parameter('start_from_center', True)
        self.declare_parameter('point_delay_sec', 1.5)
        self.declare_parameter('home_wait_sec', 4.0)

        self.boundary_path = Path(str(self.get_parameter('boundary_path').value)).expanduser()
        self.waypoint_path = Path(str(self.get_parameter('waypoint_path').value)).expanduser()
        self.preview_svg_path = Path(str(self.get_parameter('preview_svg_path').value)).expanduser()
        self.shallow_source_z = int(round(float(self.get_parameter('shallow_source_z_mm').value)))
        self.deep_source_z = int(round(float(self.get_parameter('deep_source_z_mm').value)))
        self.shallow_target_zs = parse_z_list(self.get_parameter('shallow_target_zs_mm').value)
        self.deep_target_zs = parse_z_list(self.get_parameter('deep_target_zs_mm').value)
        self.shallow_step = float(self.get_parameter('shallow_step_xy_mm').value)
        self.deep_step = float(self.get_parameter('deep_step_xy_mm').value)
        self.execute = bool(self.get_parameter('execute').value)
        self.home_first = bool(self.get_parameter('home_first').value)
        self.start_from_center = bool(self.get_parameter('start_from_center').value)
        self.point_delay_sec = float(self.get_parameter('point_delay_sec').value)
        self.home_wait_sec = float(self.get_parameter('home_wait_sec').value)

        self.move_pub = self.create_publisher(Point, '/delta_arm/move_to', 10)
        self.home_pub = self.create_publisher(Empty, '/delta_arm/home', 10)
        self.motor_pub = self.create_publisher(Bool, '/delta_arm/motor_enable', 10)

    def load_layers(self):
        data = yaml.safe_load(self.boundary_path.read_text(encoding='utf-8')) or {}
        layers = {}
        for layer in data.get('layers', []):
            z_key = int(round(float(layer['z_mm'])))
            layers[z_key] = [(float(point[0]), float(point[1])) for point in layer.get('points', [])]
        return layers

    def build_waypoints(self):
        layers = self.load_layers()
        if self.shallow_source_z not in layers:
            raise RuntimeError(f'missing shallow source layer z={self.shallow_source_z} in {self.boundary_path}')
        if self.deep_source_z not in layers:
            raise RuntimeError(f'missing deep source layer z={self.deep_source_z} in {self.boundary_path}')

        specs = []
        shallow_hull = convex_hull(layers[self.shallow_source_z])
        deep_hull = convex_hull(layers[self.deep_source_z])
        specs.extend((z, shallow_hull, self.shallow_step, self.shallow_source_z) for z in self.shallow_target_zs)
        specs.extend((z, deep_hull, self.deep_step, self.deep_source_z) for z in self.deep_target_zs)

        waypoints = []
        per_layer = []
        for z_mm, hull, step, source_z in specs:
            points = generate_grid(hull, z_mm, step)
            per_layer.append({'z_mm': z_mm, 'source_z_mm': source_z, 'step_mm': step, 'count': len(points), 'hull': hull})
            waypoints.extend(points)
        return waypoints, per_layer

    def save_waypoints(self, waypoints, per_layer):
        self.waypoint_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'description': 'Auto-generated Delta calibration preview waypoints',
            'boundary_path': str(self.boundary_path),
            'layers': per_layer,
            'waypoints': [
                {'x_mm': float(x), 'y_mm': float(y), 'z_mm': float(z)}
                for x, y, z in waypoints
            ],
        }
        self.waypoint_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')

    def write_preview_svg(self, waypoints, per_layer):
        if not waypoints:
            return
        self.preview_svg_path.parent.mkdir(parents=True, exist_ok=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        xs = [point[0] for point in waypoints]
        ys = [point[1] for point in waypoints]
        margin = 30.0
        xmin, xmax = min(xs) - margin, max(xs) + margin
        ymin, ymax = min(ys) - margin, max(ys) + margin
        width, height = 980, 820
        left, right, top, bottom = 80, 240, 70, 80
        plot_w = width - left - right
        plot_h = height - top - bottom

        def sx(x_val):
            return left + (x_val - xmin) / (xmax - xmin) * plot_w

        def sy(y_val):
            return top + (ymax - y_val) / (ymax - ymin) * plot_h

        def poly_attr(points):
            return ' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x, y in points)

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#223;font-size:14px}.small{font-size:12px;fill:#445}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#666;stroke-width:1}.label{font-size:20px;font-weight:700}</style>',
            '<rect width="100%" height="100%" fill="#fbfaf6"/>',
            f'<text class="label" x="{left}" y="36">Delta auto waypoint preview</text>',
            f'<text class="small" x="{left}" y="58">waypoints={len(waypoints)}, source={self.boundary_path}</text>',
        ]
        x_grid = math.ceil(xmin / 20.0) * 20.0
        while x_grid <= xmax:
            lines.append(f'<line class="grid" x1="{sx(x_grid):.1f}" y1="{top}" x2="{sx(x_grid):.1f}" y2="{height-bottom}"/>')
            x_grid += 20.0
        y_grid = math.ceil(ymin / 20.0) * 20.0
        while y_grid <= ymax:
            lines.append(f'<line class="grid" x1="{left}" y1="{sy(y_grid):.1f}" x2="{width-right}" y2="{sy(y_grid):.1f}"/>')
            y_grid += 20.0
        lines.append(f'<line class="axis" x1="{left}" y1="{sy(0):.1f}" x2="{width-right}" y2="{sy(0):.1f}"/>')
        lines.append(f'<line class="axis" x1="{sx(0):.1f}" y1="{top}" x2="{sx(0):.1f}" y2="{height-bottom}"/>')

        by_layer = {}
        for point in waypoints:
            by_layer.setdefault(int(round(point[2])), []).append(point)
        for index, layer in enumerate(per_layer):
            z_mm = int(layer['z_mm'])
            color = colors[index % len(colors)]
            hull = layer['hull']
            points = by_layer.get(z_mm, [])
            if len(hull) >= 3:
                lines.append(f'<polygon points="{poly_attr(hull)}" fill="{color}" fill-opacity="0.08" stroke="{color}" stroke-width="2"/>')
            for x, y, _z in points:
                lines.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}" stroke="white" stroke-width="1"/>')
            legend_y = 100 + index * 24
            lines.append(f'<rect x="{width-right+30}" y="{legend_y-13}" width="14" height="14" fill="{color}"/>')
            lines.append(f'<text class="small" x="{width-right+52}" y="{legend_y-2}">z={z_mm}, src={layer["source_z_mm"]}, step={layer["step_mm"]:.0f}, n={layer["count"]}</text>')

        lines.append('</svg>')
        self.preview_svg_path.write_text('\n'.join(lines), encoding='utf-8')

    def publish_motor(self, enabled=True):
        msg = Bool()
        msg.data = bool(enabled)
        self.motor_pub.publish(msg)

    def publish_home(self):
        self.home_pub.publish(Empty())

    def publish_move(self, xyz):
        msg = Point()
        msg.x = float(xyz[0])
        msg.y = float(xyz[1])
        msg.z = float(xyz[2])
        self.move_pub.publish(msg)
        self.get_logger().info('move x=%.1f y=%.1f z=%.1f' % (msg.x, msg.y, msg.z))

    def run_moves(self, waypoints):
        if not waypoints:
            self.get_logger().warning('no waypoints generated')
            return
        self.publish_motor(True)
        time.sleep(0.5)
        if self.home_first:
            self.get_logger().info('sending home')
            self.publish_home()
            time.sleep(self.home_wait_sec)
        if self.start_from_center:
            self.publish_move((0.0, 0.0, waypoints[0][2]))
            time.sleep(self.point_delay_sec)
        for index, point in enumerate(waypoints, start=1):
            if not rclpy.ok():
                break
            self.get_logger().info('waypoint %d/%d' % (index, len(waypoints)))
            self.publish_move(point)
            time.sleep(self.point_delay_sec)

    def run(self):
        waypoints, per_layer = self.build_waypoints()
        self.save_waypoints(waypoints, per_layer)
        self.write_preview_svg(waypoints, per_layer)
        for layer in per_layer:
            self.get_logger().info(
                'layer z=%s from z=%s: step=%.1f mm, points=%d'
                % (layer['z_mm'], layer['source_z_mm'], layer['step_mm'], layer['count'])
            )
        self.get_logger().info(f'saved waypoints: {self.waypoint_path}')
        self.get_logger().info(f'saved preview: {self.preview_svg_path}')
        if not self.execute:
            self.get_logger().info('execute=false, preview only. Set -p execute:=true to move the arm once.')
            return
        self.run_moves(waypoints)


def main(args=None):
    rclpy.init(args=args)
    node = DeltaWorkspaceAutoRunner()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
