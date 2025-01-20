import launch
from launch.substitutions import Command, LaunchConfiguration
import launch_ros
import os
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    pkgPath = launch_ros.substitutions.FindPackageShare(package = 'gantry').find('gantry')
    urdfModelPath = os.path.join(pkgPath, 'urdf/gantry.urdf.xacro')

    params = {
    'robot_description': ParameterValue(Command(['xacro ', urdfModelPath]), value_type=str)
    }


    robot_state_publisher_node = launch_ros.actions.Node(
        package = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output = 'screen',
        parameters = [params])
    
    joint_state_publisher_node = launch_ros.actions.Node(
        package = 'joint_state_publisher',
        executable = 'joint_state_publisher',
        name = 'joint_state_publisher',
        parameters = [params],
        condition = launch.conditions.IfCondition(LaunchConfiguration('gui')))
    
    joint_state_publisher_gui_node = launch_ros.actions.Node(
        package = 'joint_state_publisher_gui',
        executable = 'joint_state_publisher_gui',
        name = 'joint_state_publisher_gui',
        parameters = [params],
        condition = launch.conditions.IfCondition(LaunchConfiguration('gui')))
    
    rviz_node = launch_ros.actions.Node(
        package = 'rviz2',
        executable = 'rviz2',
        name = 'rviz2',
        output = 'screen')
    
    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(name='gui', default_value='True', 
                                             description='joint_state_publisher_gui'),
        launch.actions.DeclareLaunchArgument(name='model', default_value=urdfModelPath, 
                                             description='Path to urdf file'),

        robot_state_publisher_node,
        #joint_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node
    ])