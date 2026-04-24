#!/usr/bin/env python3
"""机械臂键盘控制节点 - 用于测试逆运动学"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from weed_locator.srv import SolveIK
from weed_locator.srv import WriteJoints
import sys
import termios
import tty
import select
import time
import numpy as np

class ArmKeyboardControl(Node):
    def __init__(self):
        super().__init__('arm_keyboard_control')
        self.ik_client = self.create_client(SolveIK, '/weed_locator/solve_ik')
        self.dynamixel_client = self.create_client(WriteJoints, '/dynamixel/write_joints')
        
        # 弧度 -> 编码器 映射参数 [enc_min, enc_max, ang_min, ang_max]
        self.angle_to_encoder_params = [
            [800, 3200, -1.91986, 1.91986],   # Joint1: 基座
            [1000, 3000, -1.74533, 1.74533],  # Joint2: 肩
            [1000, 3000, -1.74533, 1.5708],   # Joint3: 肘
            [500, 3500, -1.65806, 1.65806],  # Joint4: 腕俯仰
            [500, 3500, -2.79253, 2.79253],  # Joint5: 腕旋转
            [500, 3500, -0.174533, 1.74533], # Joint6: 夹爪
        ]
        
        # 当前末端位置 (单位: 米)
        self.x = 0.10
        self.y = 0.0
        self.z = 0.15
        
        # 上一次发送的目标位置（用于检测是否需要新的IK求解）
        self.last_sent_x = None
        self.last_sent_y = None
        self.last_sent_z = None
        
        # 位置变化阈值（米），小于此值认为位置没变化
        self.position_threshold = 0.001  # 1mm
        
        self.get_logger().info('机械臂键盘控制已启动')
        self.get_logger().info('控制说明:')
        self.get_logger().info('  w/s: 前后移动 (X轴)')
        self.get_logger().info('  a/d: 左右移动 (Y轴)')
        self.get_logger().info('  r/f: 上下移动 (Z轴)')
        self.get_logger().info('  q: 退出')
        self.get_logger().info(f'初始位置: x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}')
        
    def has_position_changed(self):
        """检测目标位置是否发生变化"""
        if self.last_sent_x is None:
            return True
        dx = abs(self.x - self.last_sent_x)
        dy = abs(self.y - self.last_sent_y)
        dz = abs(self.z - self.last_sent_z)
        return (dx + dy + dz) > self.position_threshold
    
    def update_last_sent_position(self):
        """更新最后发送的位置"""
        self.last_sent_x = self.x
        self.last_sent_y = self.y
        self.last_sent_z = self.z
        
    def call_ik_service(self):
        """调用逆运动学服务"""
        request = SolveIK.Request()
        request.target_pose.position.x = self.x
        request.target_pose.position.y = self.y
        request.target_pose.position.z = self.z
        
        self.get_logger().info(f'发送目标位置: x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}')
        
        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            result = future.result()
            if result.success:
                joints = result.joint_positions
                self.get_logger().info(f'逆运动学求解成功, 关节角度: {[f"{j:.3f}" for j in joints]}')
                
                # 调用 dynamixel 执行关节角度
                self.call_dynamixel(joints)
            else:
                self.get_logger().error(f'逆运动学求解失败: {result.message}')
        else:
            self.get_logger().error('服务调用失败')
    
    def radians_to_encoder(self, radians_values):
        """将弧度值转换为编码器值"""
        encoder_values = []
        for i, ang in enumerate(radians_values):
            if i >= len(self.angle_to_encoder_params):
                break
            enc_min, enc_max, ang_min, ang_max = self.angle_to_encoder_params[i]
            ratio = (ang - ang_min) / (ang_max - ang_min)
            val = enc_min + ratio * (enc_max - enc_min)
            val = max(float(enc_min), min(float(enc_max), val))
            encoder_values.append(float(val))
        return encoder_values
    
    def call_dynamixel(self, joints):
        """调用 dynamixel 执行关节角度"""
        # 将弧度转换为编码器值
        encoder_values = self.radians_to_encoder(joints)
        self.get_logger().info(f'转换后的编码器值: {encoder_values}')
        
        request = WriteJoints.Request()
        request.target_positions = encoder_values
        future = self.dynamixel_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None and future.result().success:
            self.get_logger().info('Dynamixel 执行成功')
        else:
            self.get_logger().error('Dynamixel 执行失败')
    
    def get_key(self):
        """获取键盘输入"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    
    def run(self):
        """运行键盘控制循环 - 只有位置变化时才发送IK求解"""
        step = 0.01  # 移动步长 1cm
        
        self.get_logger().info('进入持续控制模式 - 只有位置变化时才调用IK求解')
        self.get_logger().info('提示: 按 w/a/s/d/r/f 移动机械臂')
        
        # 首先调用一次 IK 获取初始位置
        self.call_ik_service()
        self.update_last_sent_position()
        
        while rclpy.ok():
            # 检查是否有键盘输入（非阻塞）
            if select.select([sys.stdin], [], [], 0.0)[0]:
                key = self.get_key()
                
                if key == 'w':
                    self.x += step
                    self.get_logger().info(f'目标: x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}')
                elif key == 's':
                    self.x -= step
                    self.get_logger().info(f'目标: x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}')
                elif key == 'a':
                    self.y -= step
                    self.get_logger().info(f'目标: x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}')
                elif key == 'd':
                    self.y += step
                    self.get_logger().info(f'目标: x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}')
                elif key == 'r':
                    self.z += step
                    self.get_logger().info(f'目标: x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}')
                elif key == 'f':
                    self.z -= step
                    self.get_logger().info(f'目标: x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}')
                elif key == 'q':
                    self.get_logger().info('退出控制')
                    break
                elif key == '\x03':  # Ctrl+C
                    break
            
            # 只有当目标位置发生变化时才调用IK求解
            if self.has_position_changed():
                self.call_ik_service()
                self.update_last_sent_position()
            
            # 小延时，控制循环频率约10Hz
            time.sleep(0.1)

def main(args=None):
    rclpy.init(args=args)
    controller = ArmKeyboardControl()
    try:
        controller.run()
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
