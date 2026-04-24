#!/usr/bin/env python3
"""
关节状态桥接节点

作用：
1. 订阅 ik_service 发布的关节角度 (弧度)
2. 转换为编码器值
3. 发布到 /joint_states 供RVIZ显示

这样RVIZ可以显示由IK求解得到的关节角度
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np


class JointStateBridge(Node):
    """关节状态桥接器"""

    def __init__(self):
        super().__init__('joint_state_bridge')

        # 弧度 -> 编码器参数
        self.radian_to_encoder_params = [
            [800, 3200, -1.91986, 1.91986],   # Joint1: 基座
            [1000, 3000, -1.74533, 1.74533],  # Joint2: 肩
            [1000, 3000, -1.74533, 1.5708],   # Joint3: 肘
            [500, 3500, -1.65806, 1.65806],  # Joint4: 腕俯仰
            [500, 3500, -2.79253, 2.79253],  # Joint5: 腕旋转
            [500, 3500, -0.174533, 1.74533], # Joint6: 夹爪
        ]

        # 关节名称映射 (ik_service用的名字 vs 标准名字)
        self.ik_joint_names = [
            'joint_1', 'joint_2', 'joint_3', 'joint_4', 
            'joint_5', 'joint_6', 'left_inner_knuckle_joint'
        ]
        
        self.std_joint_names = [
            'joint_1', 'joint_2', 'joint_3', 'joint_4',
            'joint_5', 'joint_6', 'left_inner_knuckle_joint'
        ]

        # 当前关节角度 (弧度)
        self.current_joint_angles = [0.0] * 7

        # 关节状态发布者
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)

        # 订阅 ik_service 的关节状态
        self.create_subscription(
            JointState,
            '/ik_joint_states',
            self.ik_joint_states_callback,
            10
        )

        self.get_logger().info('关节状态桥接节点已启动')

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

    def ik_joint_states_callback(self, msg):
        """处理IK求解器发布的关节状态"""
        # 更新当前角度
        for i, name in enumerate(msg.name):
            if name in self.ik_joint_names:
                idx = self.ik_joint_names.index(name)
                if idx < len(msg.position) and i < len(self.current_joint_angles):
                    self.current_joint_angles[idx] = msg.position[i]

        # 发布到 /joint_states
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = self.std_joint_names
        joint_state.position = self.current_joint_angles
        self.joint_pub.publish(joint_state)


def main(args=None):
    rclpy.init(args=args)
    node = JointStateBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
