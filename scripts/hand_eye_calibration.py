#!/usr/bin/env python3
"""
手眼标定节点 (Eye-to-Hand Calibration)
用于建立: 相机坐标系 → 机械臂基座坐标系 的转换关系

订阅话题:
- /joint_states: 机械臂关节状态 (来自 dynamixel_node, 编码器格式)
- /weed_3d_coordinates: 相机坐标系下的目标位置 (来自 weed_fusion_node)

发布话题:
- /target/base_point: 基坐标系下的目标位置
- /calibration/status: 标定状态

编码器 -> 弧度 转换:
- ID1 (Rotation): 800-3200 -> -1.91986~1.91986 rad
- ID2 (Pitch): 1000-3000 -> -1.74533~1.74533 rad
- ID3 (Elbow): 1000-3000 -> -1.74533~1.5708 rad
- ID4 (Wrist_Pitch): 500-3500 -> -1.65806~1.65806 rad
- ID5 (Wrist_Roll): 500-3500 -> -2.79253~2.79253 rad
- ID6 (Jaw): 500-3500 -> -0.174533~1.74533 rad
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, PointStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
import numpy as np
import yaml
import os


class HandEyeCalibrationNode(Node):
    """
    手眼标定节点 (Eye-to-Hand)
    
    原理:
    - 机械臂末端依次触碰空间中的N个标定点
    - 记录每个点的关节角度和相机检测到的位置
    - 通过SVD求解转换矩阵: P_base = R @ P_camera + t
    """
    
    def __init__(self):
        super().__init__('hand_eye_calibration')
        
        # 参数
        self.declare_parameter('calibration_points_count', 8)
        self.declare_parameter('save_path', '~/.ros/calibration/hand_eye.yaml')
        self.declare_parameter('use_existing_calibration', True)
        
        self.calibration_points_count = self.get_parameter('calibration_points_count').value
        self.save_path = os.path.expanduser(self.get_parameter('save_path').value)
        self.use_existing = self.get_parameter('use_existing_calibration').value
        
        # 编码器 -> 弧度 映射参数
        # [encoder_min, encoder_max, angle_min, angle_max]
        self.encoder_to_angle_params = [
            [800, 3200, -1.91986, 1.91986],    # ID1: Rotation
            [1000, 3000, -1.74533, 1.74533],   # ID2: Pitch
            [1000, 3000, -1.74533, 1.5708],     # ID3: Elbow
            [500, 3500, -1.65806, 1.65806],     # ID4: Wrist_Pitch
            [500, 3500, -2.79253, 2.79253],     # ID5: Wrist_Roll
            [500, 3500, -0.174533, 1.74533],     # ID6: Jaw
        ]
        
        # 标定数据
        self.camera_points = []
        self.base_points = []
        
        # 当前状态
        self.current_joint_angles = None  # 弧度
        self.T_base_camera = None
        self.is_calibrated = False
        
        # SO-ARM101 连杆长度 (米)
        self.link_lengths = {
            'base_to_shoulder': 0.0949,
            'shoulder_to_upper': 0.1126,
            'upper_to_lower': 0.1349,
            'lower_to_wrist': 0.0611,
            'wrist_to_gripper': 0.0181
        }
        
        # 订阅关节状态 (来自 dynamixel_node)
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10)
        
        # 订阅相机目标 (来自 weed_fusion_node)
        self.target_sub = self.create_subscription(
            PoseArray,
            '/weed_3d_coordinates',
            self.target_callback,
            10)
        
        # 发布基座标系下的目标
        self.base_target_pub = self.create_publisher(
            PointStamped,
            '/target/base_point',
            10)
        
        # 发布标定状态
        self.status_pub = self.create_publisher(Bool, '/calibration/status', 10)
        
        # 加载已有标定
        if self.use_existing:
            self.load_calibration()
        
        self.get_logger().info('手眼标定节点已初始化')
        self.get_logger().info(f'标定点数: {self.calibration_points_count}')
        
    def encoder_to_radians(self, encoder_values):
        """
        将编码器值转换为弧度
        
        Args:
            encoder_values: list 编码器值 [2048, 2048, ...]
            
        Returns:
            numpy array 弧度值
        """
        radians = []
        for i, enc in enumerate(encoder_values):
            if i >= len(self.encoder_to_angle_params):
                break
            enc_min, enc_max, ang_min, ang_max = self.encoder_to_angle_params[i]
            # 线性插值
            ratio = (enc - enc_min) / (enc_max - enc_min)
            angle = ang_min + ratio * (ang_max - ang_min)
            radians.append(angle)
        return np.array(radians)
        
    def joint_callback(self, msg):
        """
        接收关节状态 (编码器格式)
        
        msg.position 包含编码器值，如 [2048.0, 2048.0, ...]
        """
        if len(msg.position) >= 6:
            self.current_joint_angles = self.encoder_to_radians(list(msg.position)[:6])
        
    def target_callback(self, msg):
        """接收相机坐标系下的目标位置"""
        if len(msg.poses) == 0:
            return
            
        pose = msg.poses[0]
        camera_point = np.array([pose.position.x, pose.position.y, pose.position.z])
        
        if self.is_calibrated and self.T_base_camera is not None:
            base_point = self.camera_to_base(camera_point)
            self.publish_base_target(base_point)
        
    def forward_kinematics(self, joint_angles):
        """
        正运动学计算末端位置 (简化模型)
        
        Returns:
            end_effector_pos: [x, y, z] in base frame
        """
        theta1, theta2, theta3, theta4, theta5, theta6 = joint_angles
        
        L1 = self.link_lengths['base_to_shoulder']
        L2 = self.link_lengths['shoulder_to_upper']
        L3 = self.link_lengths['upper_to_lower']
        L4 = self.link_lengths['lower_to_wrist']
        L5 = self.link_lengths['wrist_to_gripper']
        
        # 简化: 手臂在 xz 平面内运动
        alpha = theta2 + theta3 + theta4
        
        r = L2 * np.sin(theta2) + L3 * np.sin(theta2 + theta3) + \
            L4 * np.sin(alpha)
        
        z = L1 + L2 * np.cos(theta2) + L3 * np.cos(theta2 + theta3) + \
            L4 * np.cos(alpha) + L5
        
        x = r * np.sin(theta1)
        y = r * np.cos(theta1)
        
        return np.array([x, y, z])
        
    def compute_calibration(self):
        """使用SVD计算相机到基座的转换矩阵"""
        if len(self.camera_points) < 4:
            self.get_logger().error(f'标定点不足，需要至少4个，当前{len(self.camera_points)}个')
            return False
            
        valid_indices = []
        for i, p in enumerate(self.camera_points):
            if not np.any(np.isnan(p)) and not np.any(np.isinf(p)):
                valid_indices.append(i)
        
        if len(valid_indices) < 4:
            self.get_logger().error('有效标定点不足')
            return False
        
        camera_pts = np.array([self.camera_points[i] for i in valid_indices])
        base_pts = np.array([self.base_points[i] for i in valid_indices])
        
        # Umeyama 算法
        centroid_camera = np.mean(camera_pts, axis=0)
        centroid_base = np.mean(base_pts, axis=0)
        
        camera_centered = camera_pts - centroid_camera
        base_centered = base_pts - centroid_base
        
        H = camera_centered.T @ base_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        var_camera = np.mean(np.sum(camera_centered**2, axis=1))
        s = np.trace(base_centered.T @ (R @ camera_centered.T).T) / var_camera
        t = centroid_base - s * R @ centroid_camera
        
        T = np.eye(4)
        T[:3, :3] = s * R
        T[:3, 3] = t
        
        self.T_base_camera = T
        self.T_camera_base = np.linalg.inv(T)
        self.is_calibrated = True
        
        self.get_logger().info('手眼标定完成!')
        self.get_logger().info(f'旋转矩阵:\n{R}')
        self.get_logger().info(f'平移向量: {t}')
        
        self.save_calibration()
        return True
        
    def camera_to_base(self, camera_point):
        """将相机坐标系下的点转换到基座标系"""
        if self.T_base_camera is None:
            return None
        point_h = np.append(camera_point, 1.0)
        base_h = self.T_base_camera @ point_h
        return base_h[:3]
        
    def publish_base_target(self, base_point):
        """发布基座标系下的目标点"""
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.point.x = base_point[0]
        msg.point.y = base_point[1]
        msg.point.z = base_point[2]
        self.base_target_pub.publish(msg)
        
    def save_calibration(self, filename=None):
        """保存标定结果"""
        if self.T_base_camera is None:
            return False
        if filename is None:
            filename = self.save_path
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        result = {
            'T_base_camera': self.T_base_camera.tolist(),
            'T_camera_base': self.T_camera_base.tolist(),
            'calibration_points': len(self.camera_points),
        }
        
        with open(filename, 'w') as f:
            yaml.dump(result, f)
        self.get_logger().info(f'标定结果已保存: {filename}')
        return True
        
    def load_calibration(self, filename=None):
        """加载标定结果"""
        if filename is None:
            filename = self.save_path
        if not os.path.exists(filename):
            self.get_logger().warn(f'标定文件不存在: {filename}')
            return False
            
        try:
            with open(filename, 'r') as f:
                result = yaml.safe_load(f)
            self.T_base_camera = np.array(result['T_base_camera'])
            self.T_camera_base = np.array(result['T_camera_base'])
            self.is_calibrated = True
            self.get_logger().info(f'已加载标定结果: {filename}')
            return True
        except Exception as e:
            self.get_logger().error(f'加载标定失败: {e}')
            return False


def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()