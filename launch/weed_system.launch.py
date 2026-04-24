#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    img_width = LaunchConfiguration('img_width', default='1088')
    img_height = LaunchConfiguration('img_height', default='1280')
    
    return LaunchDescription([
        DeclareLaunchArgument('img_width', default_value='1088'),
        DeclareLaunchArgument('img_height', default_value='1280'),
        
        ExecuteProcess(
            cmd=['bash', '/root/run_stereo.sh', 
                 '--mipi_image_width', img_width,
                 '--mipi_image_height', img_height,
                 '--stereonet_pub_web', 'False'],
            output='screen',
            shell=True
        ),
        
        Node(
            package='dnn_node_example',
            executable='example',
            name='dnn_example_node',
            output='screen',
            remappings=[('/image', '/image_left_raw')],
            parameters=[{
                'config_file': '/opt/tros/humble/lib/dnn_node_example/config/fcosworkconfig.json',
                'feed_type': 1,
                'image_type': 0,
                'is_shared_mem_sub': 0,
                'ros_img_topic_name': '/image_left_raw',
                'image_width': img_width,
                'image_height': img_height,
            }]
        ),
        
        Node(
            package='weed_locator',
            executable='weed_fusion_node',
            name='weed_fusion_node',
            output='screen',
            parameters=[{
                'depth_scale': 0.001,
                'min_depth': 0.1,
                'max_depth': 10.0,
            }]
        ),
    ])
