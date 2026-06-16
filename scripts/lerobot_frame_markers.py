#!/usr/bin/env python3
"""Publish a simple RViz skeleton for LeRobot frames.

This intentionally avoids mesh visuals. It is meant for calibration: compare
joint-frame positions and axes against the physical modified arm without being
misled by stale STL geometry.
"""

import math

import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


DEFAULT_FRAMES = [
    'base_link',
    'shoulder_link',
    'upper_arm_link',
    'wrist_link',
    'gripper_link',
    'gripper_frame_link',
]

DEFAULT_EDGES = [
    ('base_link', 'shoulder_link'),
    ('shoulder_link', 'upper_arm_link'),
    ('upper_arm_link', 'wrist_link'),
    ('wrist_link', 'gripper_link'),
    ('gripper_link', 'gripper_frame_link'),
]


class LeRobotFrameMarkers(Node):
    def __init__(self):
        super().__init__('lerobot_frame_markers')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('frames', ','.join(DEFAULT_FRAMES))
        self.declare_parameter(
            'edges',
            ';'.join(f'{parent},{child}' for parent, child in DEFAULT_EDGES),
        )
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('sphere_scale_m', 0.022)
        self.declare_parameter('axis_length_m', 0.055)
        self.declare_parameter('line_width_m', 0.006)
        self.declare_parameter('show_labels', True)
        self.declare_parameter('show_axes', True)

        self.base_frame = str(self.get_parameter('base_frame').value)
        self.frames = self.parse_frames(str(self.get_parameter('frames').value))
        self.edges = self.parse_edges(str(self.get_parameter('edges').value))
        self.sphere_scale_m = float(self.get_parameter('sphere_scale_m').value)
        self.axis_length_m = float(self.get_parameter('axis_length_m').value)
        self.line_width_m = float(self.get_parameter('line_width_m').value)
        self.show_labels = bool(self.get_parameter('show_labels').value)
        self.show_axes = bool(self.get_parameter('show_axes').value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.marker_pub = self.create_publisher(MarkerArray, '/lerobot/frame_markers', 10)

        rate_hz = max(1.0, float(self.get_parameter('rate_hz').value))
        self.timer = self.create_timer(1.0 / rate_hz, self.on_timer)
        self.get_logger().info(
            f'publishing calibration frame markers in {self.base_frame}: '
            '/lerobot/frame_markers'
        )

    @staticmethod
    def parse_frames(text):
        frames = [item.strip() for item in text.split(',') if item.strip()]
        return frames or DEFAULT_FRAMES

    @staticmethod
    def parse_edges(text):
        edges = []
        for item in text.split(';'):
            parts = [part.strip() for part in item.split(',') if part.strip()]
            if len(parts) == 2:
                edges.append((parts[0], parts[1]))
        return edges or DEFAULT_EDGES

    def lookup(self, frame):
        try:
            return self.tf_buffer.lookup_transform(
                self.base_frame,
                frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.02),
            )
        except TransformException:
            return None

    def on_timer(self):
        transforms = {}
        for frame in self.frames:
            transform = self.lookup(frame)
            if transform is not None:
                transforms[frame] = transform

        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        marker_id = 0

        for frame, transform in transforms.items():
            point = self.point_from_transform(transform)
            markers.markers.append(self.make_sphere(marker_id, frame, point, stamp))
            marker_id += 1
            if self.show_labels:
                markers.markers.append(self.make_label(marker_id, frame, point, stamp))
                marker_id += 1
            if self.show_axes:
                for marker in self.make_axes(marker_id, transform, stamp):
                    markers.markers.append(marker)
                    marker_id += 1

        for parent, child in self.edges:
            if parent not in transforms or child not in transforms:
                continue
            markers.markers.append(
                self.make_edge(
                    marker_id,
                    parent,
                    child,
                    self.point_from_transform(transforms[parent]),
                    self.point_from_transform(transforms[child]),
                    stamp,
                )
            )
            marker_id += 1

        self.marker_pub.publish(markers)

    def point_from_transform(self, transform):
        point = Point()
        point.x = transform.transform.translation.x
        point.y = transform.transform.translation.y
        point.z = transform.transform.translation.z
        return point

    def make_sphere(self, marker_id, frame, point, stamp):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = stamp
        marker.ns = 'frames'
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = point
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.sphere_scale_m
        marker.scale.y = self.sphere_scale_m
        marker.scale.z = self.sphere_scale_m
        if frame == 'gripper_frame_link':
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        else:
            marker.color.r = 1.0
            marker.color.g = 0.85
            marker.color.b = 0.0
        marker.color.a = 0.9
        return marker

    def make_label(self, marker_id, frame, point, stamp):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = stamp
        marker.ns = 'labels'
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = point.x
        marker.pose.position.y = point.y
        marker.pose.position.z = point.z + 0.03
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.025
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.95
        marker.text = frame
        return marker

    def make_edge(self, marker_id, parent, child, p0, p1, stamp):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = stamp
        marker.ns = 'links'
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.line_width_m
        marker.color.r = 0.2
        marker.color.g = 0.7
        marker.color.b = 1.0
        marker.color.a = 0.9
        marker.points = [p0, p1]
        return marker

    def make_axes(self, start_id, transform, stamp):
        q = transform.transform.rotation
        origin = transform.transform.translation
        rotation = self.quaternion_to_matrix(q.x, q.y, q.z, q.w)
        colors = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.25, 1.0),
        ]
        axes = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        markers = []
        for index, axis in enumerate(axes):
            start = Point()
            start.x = origin.x
            start.y = origin.y
            start.z = origin.z
            end = Point()
            end.x = origin.x + self.axis_length_m * (
                rotation[0][0] * axis[0] + rotation[0][1] * axis[1] + rotation[0][2] * axis[2]
            )
            end.y = origin.y + self.axis_length_m * (
                rotation[1][0] * axis[0] + rotation[1][1] * axis[1] + rotation[1][2] * axis[2]
            )
            end.z = origin.z + self.axis_length_m * (
                rotation[2][0] * axis[0] + rotation[2][1] * axis[1] + rotation[2][2] * axis[2]
            )
            marker = Marker()
            marker.header.frame_id = self.base_frame
            marker.header.stamp = stamp
            marker.ns = 'axes'
            marker.id = start_id + index
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.line_width_m * 0.6
            marker.color.r = colors[index][0]
            marker.color.g = colors[index][1]
            marker.color.b = colors[index][2]
            marker.color.a = 0.95
            marker.points = [start, end]
            markers.append(marker)
        return markers

    @staticmethod
    def quaternion_to_matrix(x, y, z, w):
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        return [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]


def main(args=None):
    rclpy.init(args=args)
    node = LeRobotFrameMarkers()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
