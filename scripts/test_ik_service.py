#!/usr/bin/env python3
"""
测试 IK 服务的脚本
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from weed_locator.srv import SolveIK
import time

def main(args=None):
    rclpy.init(args=args)
    node = Node('test_ik_client')
    
    # 创建服务客户端
    client = node.create_client(SolveIK, 'solve_ik')
    
    # 等待服务上线
    node.get_logger().info('等待 ik_service 上线...')
    while not client.wait_for_service(timeout_sec=2.0):
        if not rclpy.ok():
            node.get_logger().error('等待服务时被中断')
            return
        node.get_logger().info('服务不可用，继续等待...')
    
    node.get_logger().info('服务已上线，开始测试...')
    
    # 测试几个目标位置
    test_positions = [
        (0.3, 0.0, 0.2),   # 右前方，低位
        (0.4, 0.0, 0.2),   # 更右，低位
        (0.3, 0.1, 0.2),   # 右前方偏左
        (0.3, -0.1, 0.2),  # 右前方偏右
        (0.35, 0.0, 0.25), # 中间，中等高度
        (0.35, 0.0, 0.15), # 中间，低位
    ]
    
    for i, (x, y, z) in enumerate(test_positions):
        # 构造请求
        request = SolveIK.Request()
        request.target_pose.position.x = x
        request.target_pose.position.y = y
        request.target_pose.position.z = z
        request.target_pose.orientation.w = 1.0  # 默认朝上
        
        node.get_logger().info(f'测试 {i+1}: 目标位置 ({x}, {y}, {z})')
        
        # 调用服务
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future)
        
        if future.result() is not None:
            response = future.result()
            if response.success:
                joints = [f'{j:.4f}' for j in response.joint_positions]
                node.get_logger().info(f'成功! 关节角度: {joints}')
            else:
                node.get_logger().warn(f'失败: {response.message}')
        else:
            node.get_logger().error(f'服务调用失败: {future.exception()}')
        
        time.sleep(0.5)
    
    node.get_logger().info('测试完成!')
    rclpy.shutdown()

if __name__ == '__main__':
    main()
