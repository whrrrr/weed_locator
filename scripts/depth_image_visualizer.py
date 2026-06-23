#!/usr/bin/env python3
"""Convert a raw depth image into a display-friendly color image."""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class DepthImageVisualizer(Node):
    def __init__(self):
        super().__init__('depth_image_visualizer')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('visual_topic', '/camera/depth/visual')
        self.declare_parameter('min_depth_mm', 250.0)
        self.declare_parameter('max_depth_mm', 1500.0)
        self.declare_parameter('invalid_is_black', True)

        self.bridge = CvBridge()
        self.pub = self.create_publisher(
            Image,
            str(self.get_parameter('visual_topic').value),
            10,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter('depth_topic').value),
            self.on_depth,
            10,
        )
        self.get_logger().info(
            'depth visualizer: %s -> %s'
            % (
                str(self.get_parameter('depth_topic').value),
                str(self.get_parameter('visual_topic').value),
            )
        )

    def on_depth(self, msg):
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

        min_mm = float(self.get_parameter('min_depth_mm').value)
        max_mm = float(self.get_parameter('max_depth_mm').value)
        if max_mm <= min_mm:
            max_mm = min_mm + 1.0

        valid = np.isfinite(depth_mm) & (depth_mm > 0.0)
        clipped = np.clip(depth_mm, min_mm, max_mm)
        normalized = ((clipped - min_mm) * 255.0 / (max_mm - min_mm)).astype(np.uint8)

        # Invert so nearer objects are brighter before applying the colormap.
        normalized = 255 - normalized
        if bool(self.get_parameter('invalid_is_black').value):
            normalized[~valid] = 0
        else:
            normalized[~valid] = 255

        color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        color[~valid] = (0, 0, 0) if bool(self.get_parameter('invalid_is_black').value) else (255, 255, 255)

        out = self.bridge.cv2_to_imgmsg(color, encoding='bgr8')
        out.header = msg.header
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DepthImageVisualizer()
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
