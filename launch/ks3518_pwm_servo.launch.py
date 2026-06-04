#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pin = LaunchConfiguration('pin')
    initial_angle = LaunchConfiguration('initial_angle')

    return LaunchDescription([
        DeclareLaunchArgument('pin', default_value='33'),
        DeclareLaunchArgument('initial_angle', default_value='90.0'),
        Node(
            package='weed_locator',
            executable='ks3518_pwm_servo_node',
            name='ks3518_pwm_servo',
            output='screen',
            parameters=[{
                'pin': ParameterValue(pin, value_type=int),
                'frequency_hz': 50.0,
                'min_angle_deg': 0.0,
                'max_angle_deg': 180.0,
                'min_pulse_ms': 0.5,
                'max_pulse_ms': 2.5,
                'angle': ParameterValue(initial_angle, value_type=float),
                'angle_topic': '/ks3518/angle',
            }],
        ),
    ])
