#!/usr/bin/env python3
"""
weed_locator RVIZ仿真启动文件

启动内容:
1. ik_service - 逆运动学求解服务（使用绝对路径URDF给placo）
2. fake_dynamixel_node - 仿真Dynamixel服务（不连接硬件）
3. robot_state_publisher - 发布TF到RVIZ（使用package://路径给RVIZ）

核心设计：双轨制URDF
- placo需要绝对文件系统路径
- RVIZ需要package://格式路径
所以分别使用不同的URDF配置

使用方法:
  ros2 launch weed_locator weed_locator_simulation.launch.py
"""

import os
import re
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


def convert_to_package_uri(urdf_content, package_name='weed_locator'):
    """
    将URDF中的绝对路径转换为package://格式
    
    原理：ROS2的package://协议会解析到包的share目录
    
    情况分析：
    - 源URDF中mesh路径指向: /home/.../install/weed_locator/share/weed_locator/config/SO101/assets/xxx.stl
    - 需要转换成: package://weed_locator/config/SO101/assets/xxx.stl
    - package://会解析到: {share_dir}/weed_locator/config/SO101/assets/xxx.stl
    """
    # 需要把 install/weed_locator/share/weed_locator/config/SO101 替换为 package://weed_locator/config/SO101
    old_prefix = '/home/whr/cc_ws/tros_ws/install/weed_locator/share/weed_locator/config/SO101'
    new_prefix = f'package://{package_name}/config/SO101'
    
    converted = urdf_content.replace(old_prefix, new_prefix)
    
    # 调试：打印转换了多少处
    count = urdf_content.count(old_prefix)
    print(f"[weed_locator_simulation] 转换了 {count} 处mesh路径")
    
    return converted


def generate_launch_description():
    # 1. 获取URDF路径（绝对路径，给placo用）
    # config现在安装到share目录了
    pkg_dir = get_package_share_directory('weed_locator')
    
    # 尝试从share目录读取（新路径）
    urdf_file = os.path.join(pkg_dir, 'config', 'SO101', 'so101_new_calib.urdf')
    
    # 如果不存在，回退到旧的lib目录路径（开发环境可能还有）
    if not os.path.exists(urdf_file):
        urdf_file = os.path.join(pkg_dir, 'lib', 'weed_locator', 'config', 'SO101', 'so101_new_calib.urdf')
    
    # 如果还不存在，使用绝对路径（安装目录）
    if not os.path.exists(urdf_file):
        urdf_file = '/home/whr/cc_ws/tros_ws/install/weed_locator/share/weed_locator/config/SO101/so101_new_calib.urdf'
    
    if not os.path.exists(urdf_file):
        urdf_file = '/home/whr/cc_ws/tros_ws/install/weed_locator/lib/weed_locator/config/SO101/so101_new_calib.urdf'
    
    print(f"[weed_locator_simulation] URDF文件: {urdf_file}")
    print(f"[weed_locator_simulation] 文件存在: {os.path.exists(urdf_file)}")
    
    # 读取原始URDF（绝对路径）
    with open(urdf_file, 'r') as f:
        robot_desc_absolute = f.read()
    
    # 转换URDF为package://格式（给RVIZ用）
    robot_desc_package = convert_to_package_uri(robot_desc_absolute)
    
    return LaunchDescription([
        # 1. IK求解服务 - 使用绝对路径URDF给placo
        Node(
            package='weed_locator',
            executable='ik_service',
            name='ik_solver',
            output='screen',
        ),
        
        # 2. 仿真Dynamixel节点（模拟硬件响应）
        Node(
            package='weed_locator',
            executable='fake_dynamixel_node',
            name='fake_dynamixel',
            output='screen',
            parameters=[{
                'joint_ids': [1, 2, 3, 4, 5, 6],
                'publish_rate': 30.0,
            }]
        ),
        
        # 3. Robot State Publisher - 使用package://格式给RVIZ
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_desc_package,  # 使用转换后的package://格式
                'publish_frequency': 30.0,
            }]
        ),
    ])
