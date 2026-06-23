#!/usr/bin/env python3
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
    port = LaunchConfiguration('port')
    baudrate = LaunchConfiguration('baudrate')
    color_width = LaunchConfiguration('color_width')
    color_height = LaunchConfiguration('color_height')
    color_fps = LaunchConfiguration('color_fps')
    depth_width = LaunchConfiguration('depth_width')
    depth_height = LaunchConfiguration('depth_height')
    depth_fps = LaunchConfiguration('depth_fps')
    image_topic = LaunchConfiguration('image_topic')
    model_path = LaunchConfiguration('model_path')
    chess_class_name = LaunchConfiguration('chess_class_name')
    confidence_threshold = LaunchConfiguration('confidence_threshold')
    processing_interval_sec = LaunchConfiguration('processing_interval_sec')
    device = LaunchConfiguration('device')
    input_size = LaunchConfiguration('input_size')
    use_half = LaunchConfiguration('use_half')
    chess_real_diameter_mm = LaunchConfiguration('chess_real_diameter_mm')
    target_selection = LaunchConfiguration('target_selection')
    hold_last_detection_sec = LaunchConfiguration('hold_last_detection_sec')
    enhance_image = LaunchConfiguration('enhance_image')
    contrast_alpha = LaunchConfiguration('contrast_alpha')
    brightness_beta = LaunchConfiguration('brightness_beta')
    gamma = LaunchConfiguration('gamma')
    clahe_enabled = LaunchConfiguration('clahe_enabled')
    depth_topic = LaunchConfiguration('depth_topic')
    depth_camera_info_topic = LaunchConfiguration('depth_camera_info_topic')
    handeye_path = LaunchConfiguration('handeye_path')
    dual_model_source = LaunchConfiguration('dual_model_source')
    depth_search_window_px = LaunchConfiguration('depth_search_window_px')
    target_z_override_mm = LaunchConfiguration('target_z_override_mm')
    target_x_offset_mm = LaunchConfiguration('target_x_offset_mm')
    target_y_offset_mm = LaunchConfiguration('target_y_offset_mm')
    target_z_offset_mm = LaunchConfiguration('target_z_offset_mm')
    safe_xy_z_mm = LaunchConfiguration('safe_xy_z_mm')
    approach_feedrate = LaunchConfiguration('approach_feedrate')
    min_x_mm = LaunchConfiguration('min_x_mm')
    max_x_mm = LaunchConfiguration('max_x_mm')
    min_y_mm = LaunchConfiguration('min_y_mm')
    max_y_mm = LaunchConfiguration('max_y_mm')
    min_z_mm = LaunchConfiguration('min_z_mm')
    max_z_mm = LaunchConfiguration('max_z_mm')

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
            'color_width': color_width,
            'color_height': color_height,
            'color_fps': color_fps,
            'depth_width': depth_width,
            'depth_height': depth_height,
            'depth_fps': depth_fps,
            'depth_registration': 'false',
            'enable_point_cloud': 'true',
            'enable_colored_point_cloud': 'false',
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

    depth_registration = Node(
        package='weed_locator',
        executable='depth_to_color_registration',
        name='depth_to_color_registration',
        output='screen',
        parameters=[{
            'depth_topic': '/camera/depth/image_raw',
            'depth_camera_info_topic': '/camera/depth/camera_info',
            'color_topic': '/camera/color/image_raw',
            'color_camera_info_topic': '/camera/color/camera_info',
            'output_depth_topic': '/camera/depth_registered/image_raw',
            'output_camera_info_topic': '/camera/depth_registered/camera_info',
            'camera_params_service': '/camera/get_camera_params',
            'output_rate_hz': 10.0,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('start_camera', default_value='true'),
        DeclareLaunchArgument('start_delta_bridge', default_value='true'),
        DeclareLaunchArgument('port', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('baudrate', default_value='115200'),
        DeclareLaunchArgument('color_width', default_value='640'),
        DeclareLaunchArgument('color_height', default_value='480'),
        DeclareLaunchArgument('color_fps', default_value='60'),
        DeclareLaunchArgument('depth_width', default_value='640'),
        DeclareLaunchArgument('depth_height', default_value='400'),
        DeclareLaunchArgument('depth_fps', default_value='30'),
        DeclareLaunchArgument('image_topic', default_value='/camera/color/image_raw'),
        DeclareLaunchArgument('model_path', default_value='/home/wyy/gpt_dev_ws/models/xiangqi_best.pt'),
        DeclareLaunchArgument('chess_class_name', default_value='xiangqi'),
        DeclareLaunchArgument('confidence_threshold', default_value='0.01'),
        DeclareLaunchArgument('processing_interval_sec', default_value='0.033'),
        DeclareLaunchArgument('device', default_value='auto'),
        DeclareLaunchArgument('input_size', default_value='640'),
        DeclareLaunchArgument('use_half', default_value='true'),
        DeclareLaunchArgument('chess_real_diameter_mm', default_value='32.0'),
        DeclareLaunchArgument('target_selection', default_value='nearest_image_center'),
        DeclareLaunchArgument('hold_last_detection_sec', default_value='1.5'),
        DeclareLaunchArgument('enhance_image', default_value='true'),
        DeclareLaunchArgument('contrast_alpha', default_value='1.00'),
        DeclareLaunchArgument('brightness_beta', default_value='55.0'),
        DeclareLaunchArgument('gamma', default_value='0.80'),
        DeclareLaunchArgument('clahe_enabled', default_value='false'),
        DeclareLaunchArgument('depth_topic', default_value='/camera/depth_registered/image_raw'),
        DeclareLaunchArgument('depth_camera_info_topic', default_value='/camera/depth_registered/camera_info'),
        DeclareLaunchArgument('handeye_path', default_value='/home/wyy/gpt_dev_ws/calibration_targets/delta_hand_eye.yaml'),
        DeclareLaunchArgument('dual_model_source', default_value='depth'),
        DeclareLaunchArgument('depth_search_window_px', default_value='11'),
        DeclareLaunchArgument('target_z_override_mm', default_value='-230.0'),
        DeclareLaunchArgument('target_x_offset_mm', default_value='0.0'),
        DeclareLaunchArgument('target_y_offset_mm', default_value='0.0'),
        DeclareLaunchArgument('target_z_offset_mm', default_value='0.0'),
        DeclareLaunchArgument('safe_xy_z_mm', default_value='-210.0'),
        DeclareLaunchArgument('approach_feedrate', default_value='80.0'),
        DeclareLaunchArgument('min_x_mm', default_value='-90.0'),
        DeclareLaunchArgument('max_x_mm', default_value='90.0'),
        DeclareLaunchArgument('min_y_mm', default_value='-60.0'),
        DeclareLaunchArgument('max_y_mm', default_value='100.0'),
        DeclareLaunchArgument('min_z_mm', default_value='-320.0'),
        DeclareLaunchArgument('max_z_mm', default_value='0.0'),
        astra_launch,
        depth_registration,
        delta_bridge,
        Node(
            package='weed_locator',
            executable='chess_detector',
            name='chess_detector',
            output='screen',
            parameters=[{
                'image_topic': image_topic,
                'model_path': model_path,
                'chess_class_name': chess_class_name,
                'confidence_threshold': confidence_threshold,
                'processing_interval_sec': processing_interval_sec,
                'device': device,
                'input_size': input_size,
                'use_half': use_half,
                'chess_real_diameter_mm': chess_real_diameter_mm,
                'target_selection': target_selection,
                'hold_last_detection_sec': hold_last_detection_sec,
                'enhance_image': enhance_image,
                'contrast_alpha': contrast_alpha,
                'brightness_beta': brightness_beta,
                'gamma': gamma,
                'clahe_enabled': clahe_enabled,
                'publish_pixel_center_topic': '/weed_detector/pixel_center',
                'selected_pixel_center_topic': '/chess/selected_pixel_center',
                'camera_point_topic': '/chess/camera_point',
                'detections_json_topic': '/chess/detections_json',
                'draw_calibration_lines': False,
            }],
        ),
        Node(
            package='weed_locator',
            executable='chess_handeye_target',
            name='chess_handeye_target',
            output='screen',
            parameters=[{
                'handeye_path': handeye_path,
                'dual_model_source': dual_model_source,
                'selected_pixel_topic': '/chess/selected_pixel_center',
                'detections_json_topic': '/chess/detections_json',
                'depth_topic': depth_topic,
                'depth_camera_info_topic': depth_camera_info_topic,
                'depth_search_window_px': depth_search_window_px,
                'target_z_override_mm': target_z_override_mm,
                'target_x_offset_mm': target_x_offset_mm,
                'target_y_offset_mm': target_y_offset_mm,
                'target_z_offset_mm': target_z_offset_mm,
                'command_topic': '/chess/handeye_command',
                'camera_point_topic': '/chess/camera_point',
                'delta_target_topic': '/chess/delta_target',
                'move_status_topic': '/chess/move_status',
                'move_topic': '/delta_arm/move_to',
                'raw_gcode_topic': '/delta_arm/gcode_raw',
                'staged_move_on_go': True,
                'safe_xy_z_mm': safe_xy_z_mm,
                'approach_feedrate': approach_feedrate,
                'workspace_check_enabled': False,
                'min_x_mm': min_x_mm,
                'max_x_mm': max_x_mm,
                'min_y_mm': min_y_mm,
                'max_y_mm': max_y_mm,
                'min_z_mm': min_z_mm,
                'max_z_mm': max_z_mm,
                'use_polygon_workspace': False,
                'polygon_margin_mm': 8.0,
            }],
        ),
        Node(
            package='image_tools',
            executable='showimage',
            name='show_chess_detection',
            output='screen',
            arguments=['--ros-args', '-r', '/image:=/chess/detection_image'],
        ),
    ])
