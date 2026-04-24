import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    image_width = LaunchConfiguration('image_width', default='1088')
    image_height = LaunchConfiguration('image_height', default='1280')
    
    # 1. mipi_cam 相机
    mipi_cam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('mipi_cam'), 'launch/mipi_cam.launch.py')
        ),
        launch_arguments={
            'mipi_image_width': image_width,
            'mipi_image_height': image_height,
            'mipi_io_method': 'shared_mem',
            'mipi_video_device': 'GS132GS',
        }.items()
    )
    
    # 2. dnn_node_example 检测
    dnn_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('dnn_node_example'), 'launch/dnn_node_example.launch.py')
        ),
        launch_arguments={
            'dnn_example_image_width': image_width,
            'dnn_example_image_height': image_height,
        }.items()
    )
    
    # 3. StereoNet 深度（新增！）
    stereonet_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('hobot_stereonet'), 'launch/hobot_stereonet_ai_weeds.launch.py')
        )
    )
    
    # 4. 你的融合节点
    weed_fusion_node = Node(
        package='weed_locator',
        executable='weed_fusion_node',
        output='screen'
    )
    
    return LaunchDescription([
        DeclareLaunchArgument('image_width', default_value='1088'),
        DeclareLaunchArgument('image_height', default_value='1280'),
        mipi_cam_launch,
        dnn_launch,
        stereonet_launch,  # 加上深度
        weed_fusion_node,
    ])