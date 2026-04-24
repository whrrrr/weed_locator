#!/usr/bin/env python3
"""
Fake Dynamixel 节点 - 用于RVIZ仿真
模拟dynamixel服务，不实际发送命令到硬件
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from weed_locator.srv import ReadJoints, WriteJoints, MoveJoint
import numpy as np


class FakeDynamixelController(Node):
    """仿真Dynamixel控制器 - 不连接真实硬件"""

    def __init__(self):
        super().__init__('fake_dynamixel_controller')

        # 参数声明
        self.declare_parameter('joint_ids', [1, 2, 3, 4, 5, 6])
        self.declare_parameter('publish_rate', 10.0)

        self.joint_ids = self.get_parameter('joint_ids').value
        self.publish_rate = self.get_parameter('publish_rate').value

        # URDF中的实际关节名称
        self.joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        
        # 关节位置限制 [min, max] (编码器单位)
        self.joint_limits = [
            [800, 3200],   # shoulder_pan: 基座
            [1000, 3000],  # shoulder_lift: 肩
            [1000, 3000],  # elbow_flex: 肘
            [500, 3500],   # wrist_flex: 腕俯仰
            [500, 3500],   # wrist_roll: 腕旋转
            [500, 3500],   # gripper: 夹爪
        ]

        # 弧度转编码器参数 (与URDF关节顺序对应)
        self.radian_to_encoder_params = [
            [800, 3200, -1.91986, 1.91986],   # shoulder_pan: 基座
            [1000, 3000, -1.74533, 1.74533],  # shoulder_lift: 肩
            [1000, 3000, -1.74533, 1.5708],   # elbow_flex: 肘
            [500, 3500, -1.65806, 1.65806],  # wrist_flex: 腕俯仰
            [500, 3500, -2.79253, 2.79253],  # wrist_roll: 腕旋转
            [500, 3500, -0.174533, 1.74533], # gripper: 夹爪
        ]

        # 关节位置发布者
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)

        # 定时器: 发布关节位置
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)

        # 当前关节位置 (编码器值)
        self.current_positions = [2048.0] * 6

        # 当前关节角度 (弧度) - 用于TF发布
        self.current_joint_angles = [0.0] * 6

        # 创建服务
        self.read_joints_srv = self.create_service(
            ReadJoints, '/dynamixel/read_joints', self.read_joints_callback)
        self.write_joints_srv = self.create_service(
            WriteJoints, '/dynamixel/write_joints', self.write_joints_callback)
        self.move_joint_srv = self.create_service(
            MoveJoint, '/dynamixel/move_joint', self.move_joint_callback)

        self.get_logger().info('【仿真模式】Fake Dynamixel 控制器已启动')
        self.get_logger().info('服务: /dynamixel/read_joints, /dynamixel/write_joints, /dynamixel/move_joint')

    def encoder_to_radians(self, encoder_values):
        """将编码器值转换为弧度"""
        radians = []
        for i, enc in enumerate(encoder_values):
            if i >= len(self.radian_to_encoder_params):
                radians.append(0.0)
                continue
            enc_min, enc_max, ang_min, ang_max = self.radian_to_encoder_params[i]
            ratio = (enc - enc_min) / (enc_max - enc_min)
            ang = ang_min + ratio * (ang_max - ang_min)
            radians.append(ang)
        return radians

    def radians_to_encoder(self, radians_values):
        """将弧度值转换为编码器值"""
        encoder_values = []
        for i, ang in enumerate(radians_values):
            if i >= len(self.radian_to_encoder_params):
                encoder_values.append(2048.0)
                continue
            enc_min, enc_max, ang_min, ang_max = self.radian_to_encoder_params[i]
            ratio = (ang - ang_min) / (ang_max - ang_min)
            val = enc_min + ratio * (enc_max - enc_min)
            val = max(float(enc_min), min(float(enc_max), val))
            encoder_values.append(float(val))
        return encoder_values

    def timer_callback(self):
        """定时发布关节状态"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.current_joint_angles
        self.joint_pub.publish(msg)

    def read_joints_callback(self, request, response):
        """读取所有关节位置服务"""
        response.positions = self.current_positions
        response.success = True
        return response

    def write_joints_callback(self, request, response):
        """写入所有关节目标位置服务"""
        try:
            target_positions = list(request.target_positions)
            # IK可能返回12个值（包括floating base），但我们只有6个关节
            # 只取前6个弧度值
            radians_values = target_positions[:6]
            
            # IK返回的是弧度值，直接使用
            self.current_joint_angles = radians_values
            
            # 如果需要编码器值（用于真实硬件），可以转换后存储
            # 但仿真模式下我们只需要弧度值发布到/joint_states
            # self.current_positions = self.radians_to_encoder(radians_values)
            
            self.get_logger().debug(f'【仿真】收到目标弧度: {radians_values}')
            response.success = True
        except Exception as e:
            self.get_logger().error(f'写入关节失败: {e}')
            response.success = False
        return response

    def move_joint_callback(self, request, response):
        """单关节移动服务"""
        joint_id = request.joint_id
        target = request.target_position

        if joint_id not in self.joint_ids:
            self.get_logger().error(f'无效的关节ID: {joint_id}')
            response.success = False
            return response

        idx = self.joint_ids.index(joint_id)
        pos = int(np.clip(target, self.joint_limits[idx][0], self.joint_limits[idx][1]))

        self.current_positions[idx] = float(pos)
        self.current_joint_angles = self.encoder_to_radians(self.current_positions)

        response.success = True
        return response


def main(args=None):
    rclpy.init(args=args)
    node = FakeDynamixelController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
