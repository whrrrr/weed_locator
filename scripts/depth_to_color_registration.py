#!/usr/bin/env python3
"""Project native depth pixels into the RGB camera image grid.

The Astra used here publishes a 640x400 depth stream and a 640x480 UVC RGB
stream. They cannot be indexed with the same pixel coordinates. This node uses
the camera's factory intrinsics and RGB-to-depth extrinsics to publish a real
RGB-aligned depth image.
"""

import copy
import time

import cv2
import numpy as np
import rclpy
from astra_camera_msgs.srv import GetCameraParams
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class DepthToColorRegistration(Node):
    def __init__(self):
        super().__init__('depth_to_color_registration')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('depth_camera_info_topic', '/camera/depth/camera_info')
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('color_camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('output_depth_topic', '/camera/depth_registered/image_raw')
        self.declare_parameter('output_camera_info_topic', '/camera/depth_registered/camera_info')
        self.declare_parameter('camera_params_service', '/camera/get_camera_params')
        self.declare_parameter('max_sync_age_sec', 0.12)
        self.declare_parameter('output_rate_hz', 10.0)
        self.declare_parameter('min_depth_m', 0.05)
        self.declare_parameter('max_depth_m', 2.0)

        self.bridge = CvBridge()
        self.depth_info = None
        self.color_info = None
        self.latest_color_header = None
        self.depth_rays = None
        self.depth_ray_signature = None
        self.depth_to_color_rotation = None
        self.depth_to_color_translation_m = None
        self.params_request = None
        self.last_log_time = 0.0
        self.last_process_time = 0.0

        self.depth_pub = self.create_publisher(
            Image, str(self.get_parameter('output_depth_topic').value), 10
        )
        self.info_pub = self.create_publisher(
            CameraInfo, str(self.get_parameter('output_camera_info_topic').value), 10
        )
        self.create_subscription(
            Image, str(self.get_parameter('depth_topic').value), self.on_depth, 10
        )
        self.create_subscription(
            CameraInfo, str(self.get_parameter('depth_camera_info_topic').value), self.on_depth_info, 10
        )
        self.create_subscription(
            Image, str(self.get_parameter('color_topic').value), self.on_color, 10
        )
        self.create_subscription(
            CameraInfo, str(self.get_parameter('color_camera_info_topic').value), self.on_color_info, 10
        )
        self.camera_params_client = self.create_client(
            GetCameraParams, str(self.get_parameter('camera_params_service').value)
        )
        self.create_timer(1.0, self.request_camera_params)
        self.get_logger().info('Depth-to-color registration node ready')

    def request_camera_params(self):
        if self.depth_to_color_rotation is not None or self.params_request is not None:
            return
        if not self.camera_params_client.service_is_ready():
            self.get_logger().info('waiting for camera factory-parameter service')
            return
        self.params_request = self.camera_params_client.call_async(GetCameraParams.Request())
        self.params_request.add_done_callback(self.on_camera_params)

    def on_camera_params(self, future):
        self.params_request = None
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error('failed to read factory camera parameters: %s' % exc)
            return
        if not response.success:
            self.get_logger().error('camera rejected factory-parameter request')
            return

        # Driver comments define left as depth and right as RGB. r2l means:
        # p_depth = R_r2l * p_color + t_r2l. Invert it for depth -> RGB.
        color_to_depth_r = np.asarray(response.r2l_r, dtype=float).reshape(3, 3)
        color_to_depth_t_mm = np.asarray(response.r2l_t, dtype=float).reshape(3)
        self.depth_to_color_rotation = color_to_depth_r.T
        self.depth_to_color_translation_m = -self.depth_to_color_rotation @ (color_to_depth_t_mm / 1000.0)
        self.get_logger().info(
            'loaded factory RGB/depth extrinsics; depth->color translation mm=%s'
            % np.round(self.depth_to_color_translation_m * 1000.0, 3).tolist()
        )

    def on_depth_info(self, msg):
        self.depth_info = msg
        self.depth_rays = None

    def on_color_info(self, msg):
        self.color_info = msg

    def on_color(self, msg):
        self.latest_color_header = msg.header

    @staticmethod
    def stamp_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def build_depth_rays(self, width, height):
        if self.depth_info is None:
            return False
        k = np.asarray(self.depth_info.k, dtype=np.float64).reshape(3, 3)
        d = np.asarray(self.depth_info.d, dtype=np.float64).reshape(-1)
        signature = (width, height, tuple(np.round(k.reshape(-1), 9)), tuple(np.round(d, 9)))
        if self.depth_ray_signature == signature and self.depth_rays is not None:
            return True

        u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        # This Astra reports zero depth distortion. Avoid an expensive OpenCV
        # undistortion over all 256k depth pixels when the pinhole formula is exact.
        if np.allclose(d, 0.0):
            rays = np.column_stack([
                (u.reshape(-1) - k[0, 2]) / k[0, 0],
                (v.reshape(-1) - k[1, 2]) / k[1, 1],
            ])
        else:
            pixels = np.stack([u.reshape(-1), v.reshape(-1)], axis=1).reshape(-1, 1, 2)
            rays = cv2.undistortPoints(pixels, k, d).reshape(-1, 2)
        self.depth_rays = rays.astype(np.float32)
        self.depth_ray_signature = signature
        return True

    def on_depth(self, msg):
        now = time.monotonic()
        output_rate_hz = max(0.1, float(self.get_parameter('output_rate_hz').value))
        if now - self.last_process_time < 1.0 / output_rate_hz:
            return
        self.last_process_time = now
        if (
            self.depth_to_color_rotation is None or self.depth_info is None or self.color_info is None or
            self.latest_color_header is None
        ):
            return
        if abs(self.stamp_seconds(msg.header.stamp) - self.stamp_seconds(self.latest_color_header.stamp)) > float(
            self.get_parameter('max_sync_age_sec').value
        ):
            return
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().warning('depth conversion failed: %s' % exc)
            return
        if depth_raw.ndim != 2:
            return
        if msg.encoding in ('16UC1', 'mono16'):
            depth_m = depth_raw.astype(np.float32).reshape(-1) / 1000.0
        elif msg.encoding in ('32FC1', '32FC'):
            depth_m = depth_raw.astype(np.float32).reshape(-1)
        else:
            self.get_logger().warning('unsupported depth encoding: %s' % msg.encoding)
            return

        depth_height, depth_width = depth_raw.shape[:2]
        if not self.build_depth_rays(depth_width, depth_height):
            return
        color_width = int(self.color_info.width)
        color_height = int(self.color_info.height)
        if color_width <= 0 or color_height <= 0:
            return

        min_depth = float(self.get_parameter('min_depth_m').value)
        max_depth = float(self.get_parameter('max_depth_m').value)
        valid = np.isfinite(depth_m) & (depth_m >= min_depth) & (depth_m <= max_depth)
        registered_mm = np.full(color_width * color_height, np.iinfo(np.uint16).max, dtype=np.uint16)
        if np.any(valid):
            z_depth = depth_m[valid]
            rays = self.depth_rays[valid]
            x_depth = rays[:, 0] * z_depth
            y_depth = rays[:, 1] * z_depth
            rotation = self.depth_to_color_rotation
            translation = self.depth_to_color_translation_m
            x_color = rotation[0, 0] * x_depth + rotation[0, 1] * y_depth + rotation[0, 2] * z_depth + translation[0]
            y_color = rotation[1, 0] * x_depth + rotation[1, 1] * y_depth + rotation[1, 2] * z_depth + translation[1]
            z_color = rotation[2, 0] * x_depth + rotation[2, 1] * y_depth + rotation[2, 2] * z_depth + translation[2]
            valid_color = z_color > 1e-6
            x_color = x_color[valid_color]
            y_color = y_color[valid_color]
            z_color = z_color[valid_color]
            if z_color.size:
                color_k = np.asarray(self.color_info.k, dtype=np.float64).reshape(3, 3)
                color_d = np.asarray(self.color_info.d, dtype=np.float64).reshape(-1)
                x = x_color / z_color
                y = y_color / z_color
                r2 = x * x + y * y
                k1, k2 = color_d[0:2] if color_d.size >= 2 else (0.0, 0.0)
                p1, p2 = color_d[2:4] if color_d.size >= 4 else (0.0, 0.0)
                k3 = color_d[4] if color_d.size >= 5 else 0.0
                radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
                xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
                yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
                u = np.rint(color_k[0, 0] * xd + color_k[0, 2]).astype(np.int32)
                v = np.rint(color_k[1, 1] * yd + color_k[1, 2]).astype(np.int32)
                inside = (u >= 0) & (u < color_width) & (v >= 0) & (v < color_height)
                indices = v[inside] * color_width + u[inside]
                depth_mm = np.clip(np.rint(z_color[inside] * 1000.0), 1, 65535).astype(np.uint16)
                np.minimum.at(registered_mm, indices, depth_mm)

        registered_mm[registered_mm == np.iinfo(np.uint16).max] = 0
        registered = registered_mm.reshape(color_height, color_width)
        output = self.bridge.cv2_to_imgmsg(registered, encoding='16UC1')
        output.header = copy.deepcopy(self.latest_color_header)
        self.depth_pub.publish(output)
        output_info = copy.deepcopy(self.color_info)
        output_info.header = copy.deepcopy(self.latest_color_header)
        output_info.width = color_width
        output_info.height = color_height
        self.info_pub.publish(output_info)

        if now - self.last_log_time > 2.0:
            self.last_log_time = now
            filled = int(np.count_nonzero(registered))
            self.get_logger().info(
                'published registered depth %dx%d, valid pixels=%d (%.1f%%)'
                % (color_width, color_height, filled, 100.0 * filled / (color_width * color_height))
            )


def main(args=None):
    rclpy.init(args=args)
    node = DepthToColorRegistration()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
