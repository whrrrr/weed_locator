#!/usr/bin/env python3
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class OpenCVCameraPublisher(Node):
    def __init__(self):
        super().__init__('opencv_camera_publisher')

        self.declare_parameter('device', '/dev/video2')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('frame_id', 'camera_color_optical_frame')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('fx', 600.0)
        self.declare_parameter('fy', 600.0)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 240.0)
        self.declare_parameter('distortion', [0.0, 0.0, 0.0, 0.0, 0.0])

        self.device = str(self.get_parameter('device').value)
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        fps = float(self.get_parameter('fps').value)
        self.frame_id = str(self.get_parameter('frame_id').value)

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(
            Image,
            str(self.get_parameter('image_topic').value),
            10,
        )
        self.info_pub = self.create_publisher(
            CameraInfo,
            str(self.get_parameter('camera_info_topic').value),
            10,
        )

        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f'failed to open camera device: {self.device}')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        period = 1.0 / max(1.0, fps)
        self.timer = self.create_timer(period, self.publish_frame)
        self.last_warn_time = 0.0
        self.get_logger().info(
            f'OpenCV camera publishing {self.device} {self.width}x{self.height}@{fps:.1f}'
        )

    def make_camera_info(self, stamp):
        fx = float(self.get_parameter('fx').value)
        fy = float(self.get_parameter('fy').value)
        cx = float(self.get_parameter('cx').value)
        cy = float(self.get_parameter('cy').value)
        distortion = list(self.get_parameter('distortion').value)
        if len(distortion) != 5:
            distortion = [0.0, 0.0, 0.0, 0.0, 0.0]

        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.width = self.width
        msg.height = self.height
        msg.distortion_model = 'plumb_bob'
        msg.d = [float(value) for value in distortion]
        msg.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0,
        ]
        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]
        msg.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        return msg

    def publish_frame(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            now = time.time()
            if now - self.last_warn_time > 2.0:
                self.get_logger().warning(f'failed to read frame from {self.device}')
                self.last_warn_time = now
            return

        stamp = self.get_clock().now().to_msg()
        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = self.frame_id
        self.image_pub.publish(image_msg)
        self.info_pub.publish(self.make_camera_info(stamp))

    def close(self):
        if self.cap is not None:
            self.cap.release()


def main(args=None):
    rclpy.init(args=args)
    node = OpenCVCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
