#!/usr/bin/env python3
"""Launch the official LeRobot SO101 ROS bridge with robot_state_publisher."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node


def convert_to_package_uri(urdf_content, package_name='weed_locator'):
    pkg_dir = get_package_share_directory(package_name)
    so101_dir = os.path.join(pkg_dir, 'config', 'SO101')
    return urdf_content.replace(so101_dir, f'package://{package_name}/config/SO101')


def generate_launch_description():
    pkg_dir = get_package_share_directory('weed_locator')
    urdf_file = os.path.join(pkg_dir, 'config', 'SO101', 'so101_new_calib.urdf')

    with open(urdf_file, 'r') as f:
        robot_description = convert_to_package_uri(f.read())

    port = LaunchConfiguration('port')
    robot_id = LaunchConfiguration('robot_id')
    calibration_dir = LaunchConfiguration('calibration_dir')
    publish_rate = LaunchConfiguration('publish_rate')
    connect_on_start = LaunchConfiguration('connect_on_start')
    calibrate_on_connect = LaunchConfiguration('calibrate_on_connect')
    command_units = LaunchConfiguration('command_units')
    wrist_roll_sign = LaunchConfiguration('wrist_roll_sign')
    wrist_roll_offset_deg = LaunchConfiguration('wrist_roll_offset_deg')
    elbow_p_coefficient = LaunchConfiguration('elbow_p_coefficient')
    use_gripper = LaunchConfiguration('use_gripper')

    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('robot_id', default_value='my_awesome_follower_arm'),
        DeclareLaunchArgument(
            'calibration_dir',
            default_value='/home/whr/.cache/huggingface/lerobot/calibration/robots/so_follower',
        ),
        DeclareLaunchArgument('publish_rate', default_value='30.0'),
        DeclareLaunchArgument('connect_on_start', default_value='True'),
        DeclareLaunchArgument('calibrate_on_connect', default_value='False'),
        DeclareLaunchArgument('command_units', default_value='degrees'),
        DeclareLaunchArgument('wrist_roll_sign', default_value='1.0'),
        DeclareLaunchArgument('wrist_roll_offset_deg', default_value='0.0'),
        DeclareLaunchArgument('elbow_p_coefficient', default_value='16'),
        DeclareLaunchArgument('use_gripper', default_value='True'),
        Node(
            package='weed_locator',
            executable='lerobot_so101_bridge',
            name='lerobot_so101_bridge',
            output='screen',
            parameters=[{
                'port': port,
                'robot_id': robot_id,
                'calibration_dir': calibration_dir,
                'publish_rate': ParameterValue(publish_rate, value_type=float),
                'connect_on_start': ParameterValue(connect_on_start, value_type=bool),
                'calibrate_on_connect': ParameterValue(calibrate_on_connect, value_type=bool),
                'use_degrees': True,
                'max_relative_target': 10.0,
                'command_units': command_units,
                'wrist_roll_sign': ParameterValue(wrist_roll_sign, value_type=float),
                'wrist_roll_offset_deg': ParameterValue(wrist_roll_offset_deg, value_type=float),
                'elbow_p_coefficient': ParameterValue(elbow_p_coefficient, value_type=int),
                'use_gripper': ParameterValue(use_gripper, value_type=bool),
            }],
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'publish_frequency': ParameterValue(publish_rate, value_type=float),
            }],
        ),
    ])
