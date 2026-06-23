#!/usr/bin/env python3
"""Publish an RGB point cloud from aligned depth and color images."""

import struct
import threading

import cv2
import numpy as np
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField


class ColoredPointCloudFromRgbd(Node):
    def __init__(self):
        super().__init__('colored_pointcloud_from_rgbd')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/depth/camera_info')
        self.declare_parameter('points_topic', '/camera/depth_colored/points')
        self.declare_parameter('min_depth_mm', 250.0)
        self.declare_parameter('max_depth_mm', 3000.0)
        self.declare_parameter('stride', 2)

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_color = None
        self.latest_info = None

        self.points_pub = self.create_publisher(
            PointCloud2,
            str(self.get_parameter('points_topic').value),
            10,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter('color_topic').value),
            self.on_color,
            10,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter('camera_info_topic').value),
            self.on_info,
            10,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter('depth_topic').value),
            self.on_depth,
            10,
        )
        self.get_logger().info(
            'colored point cloud: %s + %s -> %s'
            % (
                str(self.get_parameter('depth_topic').value),
                str(self.get_parameter('color_topic').value),
                str(self.get_parameter('points_topic').value),
            )
        )

    def on_color(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warning(f'failed to convert color image: {exc}')
            return
        with self.lock:
            self.latest_color = image

    def on_info(self, msg):
        with self.lock:
            self.latest_info = msg

    def on_depth(self, msg):
        with self.lock:
            color = None if self.latest_color is None else self.latest_color.copy()
            info = self.latest_info
        if color is None or info is None:
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().warning(f'failed to convert depth image: {exc}')
            return

        depth = np.asarray(depth)
        if depth.dtype == np.float32 or depth.dtype == np.float64:
            depth_mm = depth.astype(np.float32) * 1000.0
        else:
            depth_mm = depth.astype(np.float32)

        height, width = depth_mm.shape[:2]
        if color.shape[0] != height or color.shape[1] != width:
            color = cv2.resize(color, (width, height), interpolation=cv2.INTER_LINEAR)

        stride = max(1, int(self.get_parameter('stride').value))
        min_mm = float(self.get_parameter('min_depth_mm').value)
        max_mm = float(self.get_parameter('max_depth_mm').value)

        depth_sample = depth_mm[0:height:stride, 0:width:stride]
        color_sample = color[0:height:stride, 0:width:stride]
        sample_h, sample_w = depth_sample.shape[:2]

        u = np.arange(0, width, stride, dtype=np.float32)[:sample_w]
        v = np.arange(0, height, stride, dtype=np.float32)[:sample_h]
        uu, vv = np.meshgrid(u, v)

        fx = float(info.k[0])
        fy = float(info.k[4])
        cx = float(info.k[2])
        cy = float(info.k[5])

        z = depth_sample / 1000.0
        valid = np.isfinite(z) & (depth_sample >= min_mm) & (depth_sample <= max_mm)
        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy

        b = color_sample[:, :, 0].astype(np.uint32)
        g = color_sample[:, :, 1].astype(np.uint32)
        r = color_sample[:, :, 2].astype(np.uint32)
        rgb_uint = (r << 16) | (g << 8) | b
        rgb_float = rgb_uint.view(np.float32)

        points = np.empty(sample_h * sample_w, dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.float32),
        ])
        points['x'] = x.reshape(-1)
        points['y'] = y.reshape(-1)
        points['z'] = z.reshape(-1)
        points['rgb'] = rgb_float.reshape(-1)

        invalid = ~valid.reshape(-1)
        points['x'][invalid] = np.nan
        points['y'][invalid] = np.nan
        points['z'][invalid] = np.nan

        cloud = PointCloud2()
        cloud.header = msg.header
        if info.header.frame_id:
            cloud.header.frame_id = info.header.frame_id
        cloud.height = sample_h
        cloud.width = sample_w
        cloud.is_bigendian = False
        cloud.is_dense = False
        cloud.point_step = 16
        cloud.row_step = cloud.point_step * sample_w
        cloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.data = points.tobytes()
        try:
            self.points_pub.publish(cloud)
        except RCLError:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = ColoredPointCloudFromRgbd()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
