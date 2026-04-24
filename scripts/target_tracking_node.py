#!/usr/bin/env python3
"""
目标追踪控制节点
订阅基座标系下的目标位置，进行逆运动学求解，控制机械臂移动到目标点

话题订阅:
- /target/base_point: 基座标系下的目标位置 (geometry_msgs/PointStamped)
- /joint_states: 机械臂关节状态 (sensor_msgs/JointState, 编码器格式)

服务调用:
- /dynamixel/write_joints: 写入关节目标位置 (编码器格式)

编码器 <-> 弧度 转换:
- ID1 (Rotation): 800-3200 <-> -1.91986~1.91986 rad
- ID2 (Pitch): 1000-3000 <-> -1.74533~1.74533 rad
- ID3 (Elbow): 1000-3000 <-> -1.74533~1.5708 rad
- ID4 (Wrist_Pitch): 500-3500 <-> -1.65806~1.65806 rad
- ID5 (Wrist_Roll): 500-3500 <-> -2.79253~2.79253 rad
- ID6 (Jaw): 500-3500 <-> -0.174533~1.74533 rad
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from weed_locator.srv import WriteJoints, MoveJoint
import numpy as np


class TargetTrackingNode(Node):
    """
    目标追踪控制节点
    
    数据流:
    /target/base_point (基座坐标) 
        -> 逆运动学 
        -> 关节角度 (弧度)
        -> 编码器转换
        -> /dynamixel/write_joints (编码器值)
        -> 机械臂运动
    """
    
    def __init__(self):
        super().__init__('target_tracking_node')
        
        # 参数
        self.declare_parameter('step_size', 0.1)
        self.declare_parameter('max_iterations', 100)
        self.declare_parameter('position_tolerance', 0.01)
        
        self.step_size = self.get_parameter('step_size').value
        self.max_iterations = self.get_parameter('max_iterations').value
        self.position_tolerance = self.get_parameter('position_tolerance').value
        
        # 编码器 -> 弧度 映射参数
        self.encoder_to_angle_params = [
            [800, 3200, -1.91986, 1.91986],    # ID1: Rotation
            [1000, 3000, -1.74533, 1.74533],   # ID2: Pitch
            [1000, 3000, -1.74533, 1.5708],     # ID3: Elbow
            [500, 3500, -1.65806, 1.65806],     # ID4: Wrist_Pitch
            [500, 3500, -2.79253, 2.79253],     # ID5: Wrist_Roll
            [500, 3500, -0.174533, 1.74533],     # ID6: Jaw
        ]
        
        # 弧度 -> 编码器 映射参数 (反向)
        self.angle_to_encoder_params = [
            [800, 3200, -1.91986, 1.91986],
            [1000, 3000, -1.74533, 1.74533],
            [1000, 3000, -1.74533, 1.5708],
            [500, 3500, -1.65806, 1.65806],
            [500, 3500, -2.79253, 2.79253],
            [500, 3500, -0.174533, 1.74533],
        ]
        
        # 关节角度限制 (弧度)
        self.joint_limits = [
            [-1.91986, 1.91986],
            [-1.74533, 1.74533],
            [-1.74533, 1.5708],
            [-1.65806, 1.65806],
            [-2.79253, 2.79253],
            [-0.174533, 1.74533],
        ]
        
        # 连杆长度 (米)
        self.link_lengths = {
            'base_to_shoulder': 0.0949,
            'shoulder_to_upper': 0.1126,
            'upper_to_lower': 0.1349,
            'lower_to_wrist': 0.0611,
            'wrist_to_gripper': 0.0181
        }
        
        # 当前状态
        self.current_joint_angles = np.zeros(6)  # 弧度
        self.target_position = None
        
        # 订阅目标点
        self.target_sub = self.create_subscription(
            PointStamped,
            '/target/base_point',
            self.target_callback,
            10)
        
        # 订阅关节状态 (来自 dynamixel_node)
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10)
        
        # 服务客户端
        self.write_joints_client = self.create_client(WriteJoints, '/dynamixel/write_joints')
        
        while not self.write_joints_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待 /dynamixel/write_joints 服务...')
        
        self.get_logger().info('目标追踪节点已初始化')
        
    def encoder_to_radians(self, encoder_values):
        """将编码器值转换为弧度"""
        radians = []
        for i, enc in enumerate(encoder_values):
            if i >= len(self.encoder_to_angle_params):
                break
            enc_min, enc_max, ang_min, ang_max = self.encoder_to_angle_params[i]
            ratio = (enc - enc_min) / (enc_max - enc_min)
            angle = ang_min + ratio * (ang_max - ang_min)
            radians.append(angle)
        return np.array(radians)
        
    def radians_to_encoder(self, radians_values):
        """将弧度值转换为编码器值"""
        encoder = []
        for i, ang in enumerate(radians_values):
            if i >= len(self.angle_to_encoder_params):
                break
            enc_min, enc_max, ang_min, ang_max = self.angle_to_encoder_params[i]
            ratio = (ang - ang_min) / (ang_max - ang_min)
            val = int(enc_min + ratio * (enc_max - enc_min))
            val = max(enc_min, min(enc_max, val))
            encoder.append(val)
        return np.array(encoder)
        
    def joint_callback(self, msg):
        """接收关节状态 (编码器格式)"""
        if len(msg.position) >= 6:
            self.current_joint_angles = self.encoder_to_radians(list(msg.position)[:6])
        
    def target_callback(self, msg):
        """接收目标点"""
        target = np.array([msg.point.x, msg.point.y, msg.point.z])
        self.get_logger().info(f'收到目标点: {target}')
        
        # 逆运动学求解
        joint_solution = self.inverse_kinematics(target)
        
        if joint_solution is not None:
            self.get_logger().info(f'逆运动学解: {np.degrees(joint_solution)} degrees')
            self.move_to_joint_angles(joint_solution)
        else:
            self.get_logger().error('逆运动学求解失败')
            
    def inverse_kinematics(self, target_position):
        """
        逆运动学求解 (简化几何模型)
        
        Args:
            target_position: [x, y, z] in base frame
            
        Returns:
            joint_angles: 6个关节角度 (弧度), or None if failed
        """
        x, y, z = target_position
        
        L1 = self.link_lengths['base_to_shoulder']
        L2 = self.link_lengths['shoulder_to_upper']
        L3 = self.link_lengths['upper_to_lower']
        L4 = self.link_lengths['lower_to_wrist']
        
        # 1. 基座旋转
        theta1 = np.arctan2(x, y) if (x**2 + y**2) > 0 else 0.0
        
        # 2. 水平距离
        r = np.sqrt(x**2 + y**2)
        z_target = z - L1 - self.link_lengths['wrist_to_gripper']
        
        # 3. 可达范围检查
        max_reach = L2 + L3 + L4
        min_reach = np.sqrt((L2 + L3 - L4)**2)
        distance = np.sqrt(r**2 + z_target**2)
        
        if distance > max_reach or distance < min_reach:
            self.get_logger().error(f'目标距离 {distance:.3f} 超出范围 [{min_reach:.3f}, {max_reach:.3f}]')
            return None
            
        # 4. 求解手臂角度
        cos_alpha = (r**2 + z_target**2 - L3**2) / (2 * L2 * np.sqrt(r**2 + z_target**2))
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        alpha = np.arccos(cos_alpha)
        
        target_angle = np.arctan2(z_target, r)
        theta2 = target_angle - alpha
        theta3 = 0.0
        theta4 = -(theta2 + theta3)
        
        # 保持当前值
        theta5 = self.current_joint_angles[4] if len(self.current_joint_angles) > 4 else 0.0
        theta6 = self.current_joint_angles[5] if len(self.current_joint_angles) > 5 else 0.5
        
        joint_angles = np.array([theta1, theta2, theta3, theta4, theta5, theta6])
        
        # 角度限制
        for i in range(6):
            if joint_angles[i] < self.joint_limits[i][0]:
                joint_angles[i] = self.joint_limits[i][0]
            elif joint_angles[i] > self.joint_limits[i][1]:
                joint_angles[i] = self.joint_limits[i][1]
                
        return joint_angles
        
    def move_to_joint_angles(self, target_joints):
        """控制机械臂移动到目标关节角度"""
        # 弧度 -> 编码器
        encoder_values = self.radians_to_encoder(target_joints)
        
        request = WriteJoints.Request()
        request.target_positions = encoder_values.tolist()
        
        future = self.write_joints_client.call_async(request)
        self.get_logger().info(f'发送目标: encoder={encoder_values}')


def main(args=None):
    rclpy.init(args=args)
    node = TargetTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()