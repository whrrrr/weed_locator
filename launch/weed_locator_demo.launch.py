#!/usr/bin/env python3
"""
weed_locator 演示启动文件

启动顺序:
1. dynamixel_node - 机械臂控制
2. weed_fusion_node - 双目目标检测 (已有)
3. hand_eye_calibration - 手眼标定
4. target_tracking_node - 目标追踪控制
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('weed_locator')
    
    ld = LaunchDescription([
        # 1. 机械臂控制节点
        Node(
            package='weed_locator',
            executable='dynamixel_node',
            name='dynamixel_controller',
            output='screen',
            parameters=[{
                'port': '/dev/ttyACM0',
                'baudrate': 1000000,
                'joint_ids': [1, 2, 3, 4, 5, 6],
                'publish_rate': 10.0,
            }],
            remappings=[
                ('/joint_states', '/joint_states'),
            ]
        ),
        
        # 2. 双目目标融合节点 (已有)
        Node(
            package='weed_locator',
            executable='weed_fusion_node',
            name='weed_fusion',
            output='screen',
            parameters=[{
                'depth_scale': 0.001,
                'min_depth': 0.1,
                'max_depth': 10.0,
            }]
        ),
        
        # 3. 手眼标定节点
        Node(
            package='weed_locator',
            executable='hand_eye_calibration',
            name='hand_eye_calibration',
            output='screen',
            parameters=[{
                'calibration_points_count': 8,
                'save_path': '~/.ros/calibration/hand_eye.yaml',
                'use_existing_calibration': True,
            }]
        ),
        
        # 4. 目标追踪控制节点
        Node(
            package='weed_locator',
            executable='target_tracking_node',
            name='target_tracking',
            output='screen',
            parameters=[{
                'step_size': 0.1,
                'max_iterations': 100,
                'position_tolerance': 0.01,
            }]
        ),
    ])
    
    return ld


if __name__ == '__main__':
    generate_launch_description()
