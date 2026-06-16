#!/usr/bin/env python3
"""Publish RViz markers for a LeRobot TCP frame and its trajectory."""

import math

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


class LeRobotTcpMarker(Node):
    def __init__(self):
        super().__init__('lerobot_tcp_marker')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tcp_frame', 'gripper_frame_link')
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('max_path_points', 1000)
        self.declare_parameter('min_path_step_m', 0.001)
        self.declare_parameter('marker_scale_m', 0.018)
        self.declare_parameter('axis_length_m', 0.05)
        self.declare_parameter('publish_axes', True)

        self.base_frame = str(self.get_parameter('base_frame').value)
        self.tcp_frame = str(self.get_parameter('tcp_frame').value)
        self.max_path_points = int(self.get_parameter('max_path_points').value)
        self.min_path_step_m = float(self.get_parameter('min_path_step_m').value)
        self.marker_scale_m = float(self.get_parameter('marker_scale_m').value)
        self.axis_length_m = float(self.get_parameter('axis_length_m').value)
        self.publish_axes = bool(self.get_parameter('publish_axes').value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.marker_pub = self.create_publisher(MarkerArray, '/lerobot/tcp_markers', 10)
        self.path_pub = self.create_publisher(Path, '/lerobot/tcp_path', 10)
        self.path = Path()
        self.path.header.frame_id = self.base_frame

        rate_hz = max(1.0, float(self.get_parameter('rate_hz').value))
        self.timer = self.create_timer(1.0 / rate_hz, self.on_timer)
        self.get_logger().info(
            f'publishing TCP marker for {self.base_frame} -> {self.tcp_frame}: '
            '/lerobot/tcp_markers, /lerobot/tcp_path'
        )

    def on_timer(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tcp_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.02),
            )
        except TransformException:
            return

        stamp = self.get_clock().now().to_msg()
        point = Point()
        point.x = transform.transform.translation.x
        point.y = transform.transform.translation.y
        point.z = transform.transform.translation.z

        self.update_path(point, stamp)
        self.publish_markers(transform, point, stamp)

    def update_path(self, point, stamp):
        if self.path.poses:
            last = self.path.poses[-1].pose.position
            distance = math.sqrt(
                (point.x - last.x) ** 2
                + (point.y - last.y) ** 2
                + (point.z - last.z) ** 2
            )
            if distance < self.min_path_step_m:
                return

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = stamp
        pose.pose.position = point
        pose.pose.orientation.w = 1.0
        self.path.header.stamp = stamp
        self.path.poses.append(pose)
        if len(self.path.poses) > self.max_path_points:
            self.path.poses = self.path.poses[-self.max_path_points :]
        self.path_pub.publish(self.path)

    def publish_markers(self, transform, point, stamp):
        markers = MarkerArray()

        sphere = Marker()
        sphere.header.frame_id = self.base_frame
        sphere.header.stamp = stamp
        sphere.ns = 'tcp'
        sphere.id = 0
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position = point
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = self.marker_scale_m
        sphere.scale.y = self.marker_scale_m
        sphere.scale.z = self.marker_scale_m
        sphere.color.r = 0.0
        sphere.color.g = 1.0
        sphere.color.b = 0.0
        sphere.color.a = 0.9
        markers.markers.append(sphere)

        trail = Marker()
        trail.header.frame_id = self.base_frame
        trail.header.stamp = stamp
        trail.ns = 'tcp'
        trail.id = 1
        trail.type = Marker.LINE_STRIP
        trail.action = Marker.ADD
        trail.pose.orientation.w = 1.0
        trail.scale.x = 0.004
        trail.color.r = 1.0
        trail.color.g = 0.0
        trail.color.b = 0.0
        trail.color.a = 0.85
        trail.points = [pose.pose.position for pose in self.path.poses]
        markers.markers.append(trail)

        if self.publish_axes:
            for marker in self.axis_markers(transform, stamp):
                markers.markers.append(marker)

        self.marker_pub.publish(markers)

    def axis_markers(self, transform, stamp):
        axes = [
            ('x', (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            ('y', (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            ('z', (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        ]
        q = transform.transform.rotation
        origin = transform.transform.translation
        rotation = self.quaternion_to_matrix(q.x, q.y, q.z, q.w)
        markers = []
        for index, (_, axis, color) in enumerate(axes, start=10):
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
            start = Point()
            start.x = origin.x
            start.y = origin.y
            start.z = origin.z

            marker = Marker()
            marker.header.frame_id = self.base_frame
            marker.header.stamp = stamp
            marker.ns = 'tcp_axes'
            marker.id = index
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.003
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.9
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
    node = LeRobotTcpMarker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
