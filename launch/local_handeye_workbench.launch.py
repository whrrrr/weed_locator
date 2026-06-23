#!/usr/bin/env python3
"""One-command launch for the independent Delta hand-eye workbench."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import FrontendLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    start_camera = LaunchConfiguration('start_camera')
    start_delta_bridge = LaunchConfiguration('start_delta_bridge')
    model_path = PathJoinSubstitution([
        FindPackageShare('weed_locator'), 'models', 'xiangqi_best.pt'
    ])
    return LaunchDescription([
        DeclareLaunchArgument('start_camera', default_value='true'),
        DeclareLaunchArgument('start_delta_bridge', default_value='true'),
        DeclareLaunchArgument(
            'port',
            default_value='/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0',
        ),
        IncludeLaunchDescription(
            FrontendLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare('astra_camera'), 'launch', 'gemini.launch.xml'
            ])),
            launch_arguments={
                'camera_name': 'camera', 'color_width': '640', 'color_height': '480', 'color_fps': '60',
                'depth_width': '640', 'depth_height': '400', 'depth_fps': '30',
                'depth_registration': 'false', 'enable_point_cloud': 'true', 'enable_colored_point_cloud': 'false',
                'enable_color': 'true', 'enable_depth': 'true', 'enable_ir': 'false',
            }.items(),
            condition=IfCondition(start_camera),
        ),
        Node(package='weed_locator', executable='depth_to_color_registration', name='depth_to_color_registration', output='screen', parameters=[{
            'depth_topic': '/camera/depth/image_raw', 'depth_camera_info_topic': '/camera/depth/camera_info',
            'color_topic': '/camera/color/image_raw', 'color_camera_info_topic': '/camera/color/camera_info',
            'output_depth_topic': '/camera/depth_registered/image_raw',
            'output_camera_info_topic': '/camera/depth_registered/camera_info',
            'camera_params_service': '/camera/get_camera_params', 'output_rate_hz': 15.0,
        }]),
        Node(package='weed_locator', executable='delta_gcode_bridge', name='delta_gcode_bridge', output='screen',
             condition=IfCondition(start_delta_bridge), parameters=[{'port': LaunchConfiguration('port'), 'baudrate': 115200, 'default_feedrate': 80.0}]),
        Node(package='weed_locator', executable='chess_detector', name='chess_detector', output='screen', parameters=[{
            'image_topic': '/camera/color/image_raw', 'model_path': model_path,
            'chess_class_name': 'xiangqi', 'confidence_threshold': 0.01, 'processing_interval_sec': 0.033,
            'device': 'auto', 'input_size': 640, 'use_half': True, 'target_selection': 'nearest_image_center',
            'hold_last_detection_sec': 0.0, 'enhance_image': True, 'contrast_alpha': 1.0,
            'brightness_beta': 55.0, 'gamma': 0.8, 'clahe_enabled': False, 'draw_calibration_lines': False,
        }]),
        Node(package='weed_locator', executable='delta_handeye_workbench', name='delta_handeye_workbench', output='screen'),
    ])
