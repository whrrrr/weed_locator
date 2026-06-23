#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    camera_device = LaunchConfiguration('camera_device')
    port = LaunchConfiguration('port')
    baudrate = LaunchConfiguration('baudrate')
    calibration_dir = LaunchConfiguration('calibration_dir')
    start_delta_bridge = LaunchConfiguration('start_delta_bridge')
    start_calibration = LaunchConfiguration('start_calibration')

    save_path = PathJoinSubstitution([calibration_dir, 'delta_hand_eye.yaml'])
    validation_path = PathJoinSubstitution([calibration_dir, 'delta_hand_eye_filtered.yaml'])
    waypoint_path = PathJoinSubstitution([calibration_dir, 'delta_safe_9_waypoints.yaml'])
    boundary_path = PathJoinSubstitution([calibration_dir, 'delta_workspace_slices.yaml'])
    manual_rotation_path = PathJoinSubstitution([calibration_dir, 'delta_manual_board_rotations.yaml'])

    return LaunchDescription([
        DeclareLaunchArgument('camera_device', default_value='/dev/video2'),
        DeclareLaunchArgument('port', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('baudrate', default_value='115200'),
        DeclareLaunchArgument(
            'calibration_dir',
            default_value='/home/wyy/gpt_dev_ws/calibration_targets',
        ),
        DeclareLaunchArgument('start_delta_bridge', default_value='true'),
        DeclareLaunchArgument('start_calibration', default_value='true'),
        Node(
            package='weed_locator',
            executable='opencv_camera_publisher',
            name='opencv_camera_publisher',
            output='screen',
            parameters=[{
                'device': camera_device,
                'width': 640,
                'height': 480,
                'fps': 30.0,
                'image_topic': '/camera/color/image_raw',
                'camera_info_topic': '/camera/color/camera_info',
                'fx': 600.0,
                'fy': 600.0,
                'cx': 320.0,
                'cy': 240.0,
            }],
        ),
        Node(
            package='weed_locator',
            executable='delta_gcode_bridge',
            name='delta_gcode_bridge',
            output='screen',
            condition=IfCondition(start_delta_bridge),
            parameters=[{
                'port': port,
                'baudrate': baudrate,
                'default_feedrate': 80.0,
            }],
        ),
        Node(
            package='weed_locator',
            executable='delta_charuco_calibration',
            name='delta_charuco_calibration',
            output='screen',
            emulate_tty=True,
            condition=IfCondition(start_calibration),
            parameters=[{
                'image_topic': '/camera/color/image_raw',
                'camera_info_topic': '/camera/color/camera_info',
                'delta_move_topic': '/delta_arm/move_to',
                'delta_home_topic': '/delta_arm/home',
                'save_path': save_path,
                'validation_path': validation_path,
                'waypoint_path': waypoint_path,
                'boundary_path': boundary_path,
                'manual_rotation_path': manual_rotation_path,
                'debug_image_path': '/tmp/delta_charuco_debug.png',
                'squares_x': 8,
                'squares_y': 5,
                'square_length_m': 0.020,
                'marker_length_m': 0.014,
                'dictionary': 'DICT_4X4_50',
                'home_x_mm': 0.0,
                'home_y_mm': 0.0,
                'home_z_mm': 0.0,
                'feedrate': 80.0,
                'hold_after_auto_run': True,
                'auto_start_index': 1,
                'auto_end_index': 8,
                'manual_rotation_waypoint_index': 9,
                'motion_safety_enabled': True,
                'safe_xy_z_mm': -210.0,
                'workspace_min_x_mm': -90.0,
                'workspace_max_x_mm': 90.0,
                'workspace_min_y_mm': -90.0,
                'workspace_max_y_mm': 90.0,
                'workspace_min_z_mm': -320.0,
                'workspace_max_z_mm': 0.0,
                'jog_step_xy_mm': 5.0,
                'jog_step_z_mm': 5.0,
                'debug_image_topic': '/delta_charuco/debug_image',
                'publish_debug_image': True,
            }],
        ),
    ])
