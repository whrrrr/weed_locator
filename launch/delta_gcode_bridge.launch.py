#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    port = LaunchConfiguration(
        'port',
        default='/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0',
    )
    baudrate = LaunchConfiguration('baudrate', default='115200')
    default_feedrate = LaunchConfiguration('default_feedrate', default='80.0')

    return LaunchDescription([
        DeclareLaunchArgument('port', default_value=port),
        DeclareLaunchArgument('baudrate', default_value=baudrate),
        DeclareLaunchArgument('default_feedrate', default_value=default_feedrate),
        Node(
            package='weed_locator',
            executable='delta_gcode_bridge',
            name='delta_gcode_bridge',
            output='screen',
            parameters=[{
                'port': port,
                'baudrate': baudrate,
                'default_feedrate': default_feedrate,
            }],
        ),
    ])
