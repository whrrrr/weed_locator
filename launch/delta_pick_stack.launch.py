#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    port = LaunchConfiguration(
        'port',
        default='/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0',
    )
    baudrate = LaunchConfiguration('baudrate', default='115200')
    video_device = LaunchConfiguration('video_device', default='/dev/video0')
    image_topic = LaunchConfiguration('image_topic', default='/camera/image_raw')
    model_path = LaunchConfiguration(
        'model_path',
        default=PathJoinSubstitution([
            FindPackageShare('weed_locator'), 'models', 'xiangqi_best.pt'
        ]),
    )
    chess_class_name = LaunchConfiguration('chess_class_name', default='xiangqi')
    confidence_threshold = LaunchConfiguration('confidence_threshold', default='0.15')
    processing_interval_sec = LaunchConfiguration('processing_interval_sec', default='0.2')
    device = LaunchConfiguration('device', default='auto')
    image_width = LaunchConfiguration('image_width', default='640.0')
    top_line_y = LaunchConfiguration('top_line_y', default='120.0')
    bottom_line_y = LaunchConfiguration('bottom_line_y', default='385.0')
    belt_width_mm = LaunchConfiguration('belt_width_mm', default='100.0')
    high_delay_sec = LaunchConfiguration('high_delay_sec', default='22.5')

    return LaunchDescription([
        DeclareLaunchArgument('port', default_value=port),
        DeclareLaunchArgument('baudrate', default_value=baudrate),
        DeclareLaunchArgument('video_device', default_value=video_device),
        DeclareLaunchArgument('image_topic', default_value=image_topic),
        DeclareLaunchArgument('model_path', default_value=model_path),
        DeclareLaunchArgument('chess_class_name', default_value=chess_class_name),
        DeclareLaunchArgument('confidence_threshold', default_value=confidence_threshold),
        DeclareLaunchArgument('processing_interval_sec', default_value=processing_interval_sec),
        DeclareLaunchArgument('device', default_value=device),
        DeclareLaunchArgument('image_width', default_value=image_width),
        DeclareLaunchArgument('top_line_y', default_value=top_line_y),
        DeclareLaunchArgument('bottom_line_y', default_value=bottom_line_y),
        DeclareLaunchArgument('belt_width_mm', default_value=belt_width_mm),
        DeclareLaunchArgument('high_delay_sec', default_value=high_delay_sec),
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            name='camera',
            output='screen',
            parameters=[{
                'video_device': video_device,
            }],
            remappings=[
                ('/image_raw', image_topic),
            ],
        ),
        Node(
            package='weed_locator',
            executable='chess_detector',
            name='chess_detector',
            output='screen',
            parameters=[{
                'model_path': model_path,
                'image_topic': image_topic,
                'chess_class_name': chess_class_name,
                'confidence_threshold': confidence_threshold,
                'processing_interval_sec': processing_interval_sec,
                'device': device,
                'top_line_y': top_line_y,
                'bottom_line_y': bottom_line_y,
                'belt_width_mm': belt_width_mm,
                'high_delay_sec': high_delay_sec,
                'trigger_use_abs_x': True,
            }],
        ),
        Node(
            package='weed_locator',
            executable='delta_gcode_bridge',
            name='delta_gcode_bridge',
            output='screen',
            parameters=[{
                'port': port,
                'baudrate': baudrate,
            }],
        ),
        Node(
            package='weed_locator',
            executable='workspace_checker',
            name='workspace_checker',
            output='screen',
        ),
        Node(
            package='weed_locator',
            executable='target_commander',
            name='target_commander',
            output='screen',
            parameters=[{
                'auto_home_first': False,
                'pc_relative_mode': True,
                'drop_y': 80.0,
                'go_work_origin_before_pick': True,
                'work_origin_x': 0.0,
                'work_origin_y': -55.0,
                'work_origin_z': -195.0,
            }],
        ),
        Node(
            package='weed_locator',
            executable='conveyor_controller',
            name='conveyor_controller',
            output='screen',
        ),
        Node(
            package='weed_locator',
            executable='conveyor_pick_scheduler',
            name='conveyor_pick_scheduler',
            output='screen',
            parameters=[{
                'image_width': image_width,
                'top_line_y': top_line_y,
                'bottom_line_y': bottom_line_y,
                'belt_width_mm': belt_width_mm,
            }],
        ),
    ])
