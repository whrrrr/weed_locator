#!/usr/bin/env python3
"""
测试 IK 服务的脚本
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from weed_locator.srv import SolveIK, WriteJoints
import argparse
import time

def main(args=None):
    parser = argparse.ArgumentParser(description='测试 IK 服务，并把结果写入 RViz 仿真关节')
    parser.add_argument('x', nargs='?', type=float, help='目标 x 坐标，单位 m')
    parser.add_argument('y', nargs='?', type=float, help='目标 y 坐标，单位 m')
    parser.add_argument('z', nargs='?', type=float, help='目标 z 坐标，单位 m')
    parsed_args, ros_args = parser.parse_known_args(args)

    if any(value is not None for value in [parsed_args.x, parsed_args.y, parsed_args.z]):
        if None in [parsed_args.x, parsed_args.y, parsed_args.z]:
            parser.error('如果传目标点，需要同时提供 x y z，例如: test_ik_service -- 0.3 0.0 0.2')
        test_positions = [(parsed_args.x, parsed_args.y, parsed_args.z)]
    else:
        # 不传参数时，测试几个默认目标位置
        test_positions = [
            (0.3, 0.0, 0.2),   # 右前方，低位
            (0.4, 0.0, 0.2),   # 更右，低位
            (0.3, 0.1, 0.2),   # 右前方偏左
            (0.3, -0.1, 0.2),  # 右前方偏右
            (0.35, 0.0, 0.25), # 中间，中等高度
            (0.35, 0.0, 0.15), # 中间，低位
        ]

    rclpy.init(args=ros_args)
    node = Node('test_ik_client')
    
    # 创建服务客户端
    ik_client = node.create_client(SolveIK, '/weed_locator/solve_ik')
    write_joints_client = node.create_client(WriteJoints, '/dynamixel/write_joints')
    
    # 等待服务上线
    node.get_logger().info('等待 ik_service 上线...')
    while not ik_client.wait_for_service(timeout_sec=2.0):
        if not rclpy.ok():
            node.get_logger().error('等待服务时被中断')
            return
        node.get_logger().info('服务不可用，继续等待...')

    node.get_logger().info('等待 fake_dynamixel_node 上线...')
    while not write_joints_client.wait_for_service(timeout_sec=2.0):
        if not rclpy.ok():
            node.get_logger().error('等待服务时被中断')
            return
        node.get_logger().info('/dynamixel/write_joints 服务不可用，继续等待...')
    
    node.get_logger().info('服务已上线，开始测试...')
    
    for i, (x, y, z) in enumerate(test_positions):
        # 构造请求
        request = SolveIK.Request()
        request.target_pose.position.x = x
        request.target_pose.position.y = y
        request.target_pose.position.z = z
        request.target_pose.orientation.w = 1.0  # 默认朝上
        
        node.get_logger().info(f'测试 {i+1}: 目标位置 ({x}, {y}, {z})')
        
        # 调用服务
        future = ik_client.call_async(request)
        rclpy.spin_until_future_complete(node, future)
        
        if future.result() is not None:
            response = future.result()
            if response.success:
                joints = [f'{j:.4f}' for j in response.joint_positions]
                node.get_logger().info(f'成功! 关节角度: {joints}')

                write_request = WriteJoints.Request()
                write_request.target_positions = list(response.joint_positions)
                write_future = write_joints_client.call_async(write_request)
                rclpy.spin_until_future_complete(node, write_future)

                if write_future.result() is not None and write_future.result().success:
                    node.get_logger().info('已写入 /dynamixel/write_joints，RViz 应更新姿态')
                else:
                    node.get_logger().warn('写入 /dynamixel/write_joints 失败')
            else:
                node.get_logger().warn(f'失败: {response.message}')
        else:
            node.get_logger().error(f'服务调用失败: {future.exception()}')
        
        time.sleep(0.5)
    
    node.get_logger().info('测试完成!')
    rclpy.shutdown()

if __name__ == '__main__':
    main()
