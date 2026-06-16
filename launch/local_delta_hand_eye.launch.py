#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import FrontendLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    port = LaunchConfiguration('port')
    baudrate = LaunchConfiguration('baudrate')
    calibration_dir = LaunchConfiguration('calibration_dir')
    start_camera = LaunchConfiguration('start_camera')
    start_delta_bridge = LaunchConfiguration('start_delta_bridge')
    start_calibration = LaunchConfiguration('start_calibration')
    show_debug_image = LaunchConfiguration('show_debug_image')

    save_path = PathJoinSubstitution([calibration_dir, 'delta_hand_eye.yaml'])
    validation_path = PathJoinSubstitution([calibration_dir, 'delta_hand_eye_filtered.yaml'])
    waypoint_path = PathJoinSubstitution([calibration_dir, 'delta_safe_9_waypoints.yaml'])
    boundary_path = PathJoinSubstitution([calibration_dir, 'delta_workspace_slices.yaml'])
    manual_rotation_path = PathJoinSubstitution([calibration_dir, 'delta_manual_board_rotations.yaml'])
    bad_zone_path = PathJoinSubstitution([calibration_dir, 'delta_bad_discovery_zones.yaml'])
    corner_observation_path = PathJoinSubstitution([calibration_dir, 'delta_corner_observations.yaml'])
    discovery_progress_path = PathJoinSubstitution([calibration_dir, 'delta_discovery_progress.yaml'])
    discovery_dashboard_path = PathJoinSubstitution([calibration_dir, 'delta_discovery_dashboard.txt'])

    astra_launch = IncludeLaunchDescription(
        FrontendLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('astra_camera'),
                'launch',
                'gemini.launch.xml',
            ])
        ),
        launch_arguments={
            'camera_name': 'camera',
            'depth_registration': 'true',
            'enable_point_cloud': 'true',
            'enable_colored_point_cloud': 'true',
            'enable_color': 'true',
            'enable_depth': 'true',
            'enable_ir': 'false',
        }.items(),
        condition=IfCondition(start_camera),
    )

    delta_bridge = Node(
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
    )

    calibration = Node(
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
            'bad_zone_path': bad_zone_path,
            'corner_observation_path': corner_observation_path,
            'discovery_progress_path': discovery_progress_path,
            'discovery_dashboard_path': discovery_dashboard_path,
            'debug_image_path': '/tmp/delta_charuco_debug.png',
            'squares_x': 8,
            'squares_y': 5,
            'square_length_m': 0.028,
            'marker_length_m': 0.020,
            'dictionary': 'DICT_4X4_50',
            'min_charuco_corners': 14,
            'max_reprojection_error_px': 1.0,
            'calibration_outlier_threshold_mm': 20.0,
            'home_x_mm': 0.0,
            'home_y_mm': 0.0,
            'home_z_mm': 0.0,
            'feedrate': 80.0,
            'home_before_discovery': True,
            'home_between_samples': True,
            'home_settle_sec': 4.0,
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
            'stable_detection_frames': 5,
            'stable_detection_tolerance_mm': 1.0,
            'post_move_detect_timeout_sec': 5.0,
            'auto_discover_on_start': False,
            'discover_target_waypoints': 25,
            'discover_max_probes': 260,
            'discover_min_corners': 14,
            'discover_grid_step_xy_mm': 10.0,
            'discover_z_mm': -230.0,
            'discover_z_levels_down': 1,
            'discover_z_step_mm': 20.0,
            'discover_bounds_margin_mm': 0.0,
            'discover_save_samples': True,
            'discover_compute_after': True,
            'discover_bad_zone_radius_mm': 5.0,
            'discover_bad_zone_max_corners': 11,
            'discover_adaptive_radius_mm': 25.0,
        }],
    )

    debug_view = Node(
        package='image_tools',
        executable='showimage',
        name='show_delta_charuco_debug',
        output='screen',
        condition=IfCondition(show_debug_image),
        arguments=['--ros-args', '-r', '/image:=/delta_charuco/debug_image'],
    )

    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('baudrate', default_value='115200'),
        DeclareLaunchArgument(
            'calibration_dir',
            default_value='/home/wyy/gpt_dev_ws/calibration_targets',
        ),
        DeclareLaunchArgument('start_camera', default_value='true'),
        DeclareLaunchArgument('start_delta_bridge', default_value='true'),
        DeclareLaunchArgument('start_calibration', default_value='true'),
        DeclareLaunchArgument('show_debug_image', default_value='true'),
        astra_launch,
        delta_bridge,
        calibration,
        debug_view,
    ])
