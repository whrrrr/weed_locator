#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from message_filters import ApproximateTimeSynchronizer, Subscriber
from ai_msgs.msg import PerceptionTargets
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
import numpy as np
import cv2
import time


class WeedFusionNode(Node):
    def __init__(self):
        super().__init__('weed_fusion_node')

        # 参数
        self.declare_parameter('depth_scale', 0.001)  # 毫米转米
        self.declare_parameter('min_depth', 0.1)
        self.declare_parameter('max_depth', 10.0)

        # 硬编码相机内参（GS132GS 1088x1280）
        self.camera_matrix = np.array([
            [800.0, 0.0, 544.0],
            [0.0, 800.0, 640.0],
            [0.0, 0.0, 1.0]
        ])
        self.fx = self.camera_matrix[0, 0]
        self.fy = self.camera_matrix[1, 1]
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]

        # 深度图和检测图像的分辨率比例
        self.depth_width = 640
        self.depth_height = 352
        self.det_width = 1088
        self.det_height = 1280
        self.scale_x = self.depth_width / self.det_width  # 0.588
        self.scale_y = self.depth_height / self.det_height  # 0.275

        self.get_logger().info(f'Camera: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')
        self.get_logger().info(f'Scale: x={self.scale_x:.3f}, y={self.scale_y:.3f}')

        # 订阅检测和深度图
        det_sub = Subscriber(self, PerceptionTargets, '/hobot_dnn_detection')
        depth_sub = Subscriber(self, Image, '/StereoNetNode/stereonet_depth')

        self.sync = ApproximateTimeSynchronizer([det_sub, depth_sub], queue_size=10, slop=0.3)
        self.sync.registerCallback(self.fusion_callback)

        self.pose_pub = self.create_publisher(PoseArray, '/weed_3d_coordinates', 10)
        self.bridge = CvBridge()
        self.get_logger().info('WeedFusionNode started!')

        # 统计用
        self.cb_count = 0
        self.pub_count = 0
        self.last_stat_time = time.time()

    def fusion_callback(self, det_msg, depth_msg):
        self.cb_count += 1

        # 转换深度图为 numpy
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return

        # 确保是 float 或 uint16
        if depth_image.dtype == np.uint16:
            depth_image = depth_image.astype(np.float32)

        depth_scale = self.get_parameter('depth_scale').value
        min_depth = self.get_parameter('min_depth').value
        max_depth = self.get_parameter('max_depth').value

        pose_array = PoseArray()
        pose_array.header = det_msg.header

        for target in det_msg.targets:
            for roi in target.rois:
                rect = roi.rect

                # 检测框中心点（1088x1280 坐标系）
                u_det = int(rect.x_offset + rect.width / 2)
                v_det = int(rect.y_offset + rect.height / 2)

                # 直接 scale 映射到深度图坐标（无 resize，无插值）
                u_depth = int(u_det * self.scale_x)
                v_depth = int(v_det * self.scale_y)

                # 坐标边界检查
                if u_depth >= self.depth_width or v_depth >= self.depth_height:
                    continue

                Z_mm = depth_image[v_depth, u_depth]

                if np.isnan(Z_mm) or Z_mm <= 0:
                    continue

                # 转米
                Z = Z_mm * depth_scale

                if Z < min_depth or Z > max_depth:
                    continue

                # 2D -> 3D反投影（相机坐标系）
                X = (u_det - self.cx) * Z / self.fx
                Y = (v_det - self.cy) * Z / self.fy

                pose = Pose()
                pose.position.x = X
                pose.position.y = Y
                pose.position.z = Z
                pose_array.poses.append(pose)

                self.get_logger().debug(
                    f'Det({u_det},{v_det}) -> Depth({u_depth},{v_depth}) -> '
                    f'3D({X:.3f}, {Y:.3f}, {Z:.3f})'
                )

        # 发布结果
        if len(pose_array.poses) > 0:
            self.pub_count += 1
            self.pose_pub.publish(pose_array)

        # 每2秒打印一次统计
        now = time.time()
        if now - self.last_stat_time >= 2.0:
            elapsed = now - self.last_stat_time
            self.get_logger().info(
                f'CB: {self.cb_count/elapsed:.1f}Hz, PUB: {self.pub_count/elapsed:.1f}Hz, '
                f'ratio: {self.pub_count}/{self.cb_count}'
            )
            self.cb_count = 0
            self.pub_count = 0
            self.last_stat_time = now


def main(args=None):
    rclpy.init(args=args)
    node = WeedFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()