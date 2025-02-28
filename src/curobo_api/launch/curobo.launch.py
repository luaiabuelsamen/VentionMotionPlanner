import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot_config_file', default_value='ur5e.yml', description='Path to robot config file'),
        DeclareLaunchArgument('world_config_file', default_value='collision_table.yml', description='Path to world config file'),
        DeclareLaunchArgument('tensor_device', default_value='cuda:0', description='Device for tensor computations'),
        DeclareLaunchArgument('n_obstacle_cuboids', default_value='20', description='Number of obstacle cuboids'),
        DeclareLaunchArgument('n_obstacle_mesh', default_value='2', description='Number of obstacle meshes'),

        Node(
            package='curobo_api',
            executable='curobo_service',
            name='curobo_service',
            output='screen',
            parameters=[{
                'robot_config_file': LaunchConfiguration('robot_config_file'),
                'world_config_file': LaunchConfiguration('world_config_file'),
                'tensor_device': LaunchConfiguration('tensor_device'),
                'n_obstacle_cuboids': LaunchConfiguration('n_obstacle_cuboids'),
                'n_obstacle_mesh': LaunchConfiguration('n_obstacle_mesh'),
            }]
        )
    ])
