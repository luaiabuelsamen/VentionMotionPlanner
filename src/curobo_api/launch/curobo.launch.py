import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
import yaml
from ament_index_python.packages import get_package_share_directory
from urdfpy import URDF
import shutil

target_yaml_dir = os.path.expanduser('~/curobo/src/curobo/content/configs/robot')
assets_base_path = os.path.expanduser('~/curobo/src/curobo/content/assets')

def configure_nodes(context, *args, **kwargs):
    demo = LaunchConfiguration('demo').perform(context)
    robot_config_file = LaunchConfiguration('robot_config_file').perform(context)
    world_config_file = LaunchConfiguration('world_config_file').perform(context)
    mujoco_xml_path = './src/mujoco_curobo/assets/ur5e/scene_ur5e_2f140_obj_gantry.xml'
    mujoco_meshes = []
    mujoco_update_world = []
    nodes = []
    joint_names = []
    meshes_array = ['None']
    
    if demo and demo != '':
        package_share_directory = get_package_share_directory('curobo_api')
        demos_file_path = os.path.join(package_share_directory, 'config', 'demos.yml')
        if not os.path.exists(demos_file_path):
            demos_file_path = os.path.join('.', 'config', 'demos.yml')
        if os.path.exists(demos_file_path):
            with open(demos_file_path, 'r') as file:
                demos_config = yaml.safe_load(file)
            if demo in demos_config:
                demo_params = demos_config[demo]
                if 'curobo' in demo_params:
                    curobo_params = demo_params['curobo']
                    if 'robot_config_file' in curobo_params:
                        robot_config_file = curobo_params['robot_config_file']
                        yml_file_path = os.path.join('.', f'config/curobo/self_collision/{robot_config_file}.yml')
                        urdf_file_path = os.path.join('.', f'config/curobo/self_collision/{robot_config_file}.urdf')
                        
                        if os.path.exists(yml_file_path):
                            shutil.copy(yml_file_path, target_yaml_dir)
                        else:
                            yml_file_path = os.path.join(target_yaml_dir, f'{robot_config_file}.yml')
                        
                        with open(yml_file_path, 'r') as yml:
                            config = yaml.safe_load(yml)
                            urdf_relative_path = config.get('robot_cfg', {}).get('kinematics', {}).get('urdf_path', '')
                            lock_joints = config['robot_cfg']['kinematics']['lock_joints']
                            joint_names = config['robot_cfg']['kinematics']['cspace']['joint_names']
                            if lock_joints:
                                joint_names = [joint for joint in joint_names if joint not in lock_joints]
                            urdf_full_path = os.path.join(assets_base_path, urdf_relative_path)
                            if os.path.exists(urdf_file_path):
                                shutil.copy(urdf_file_path, urdf_full_path)
                        
                    if 'world_config_file' in curobo_params:
                        world_config_file = curobo_params['world_config_file']
                if 'mujoco' in demo_params:
                    mujoco_params = demo_params['mujoco']
                    if 'xml_path' in mujoco_params:
                        mujoco_xml_path = mujoco_params['xml_path']
                    if 'meshes' in mujoco_params:
                        mujoco_meshes = mujoco_params['meshes']
                        for key, value in mujoco_meshes.items():
                            meshes_array.append(key)
                            meshes_array.append(value)
                    if 'update_world' in mujoco_params:
                        mujoco_update_world = mujoco_params['update_world']
                    nodes.append(
                        Node(
                            package='mujoco_api',
                            executable='mujoco_service',
                            name='mujoco_joint_state_publisher',
                            output='screen',
                            parameters=[{
                                'xml_path': mujoco_xml_path,
                                'meshes': meshes_array,
                                'update_world': mujoco_update_world,
                                'publish_rate': 50.0,
                                'default_positions': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                'joint_names': joint_names
                            }]
                        )
                    )
            else:
                raise Exception(f'Demo {demo} doesnt exist in yml')
        else:
            print(f"Warning: demos.yml file not found at {demos_file_path}")

    nodes.append(
        Node(
            package='curobo_api',
            executable='curobo_service',
            name='curobo_service',
            output='screen',
            parameters=[{
                'robot_config_file': robot_config_file + '.yml',
                'world_config_file': world_config_file,
            }]
        )
    )
    
    return nodes

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'demo', 
            default_value='',
            description='Demo configuration to use from demos.yml'
        ),
        DeclareLaunchArgument(
            'robot_config_file', 
            default_value='ur5e_robotiq_2f_140_x.yml',
            description='Path to robot config file'
        ),
        DeclareLaunchArgument(
            'world_config_file', 
            default_value='collision_table.yml',
            description='Path to world config file'
        ),
        OpaqueFunction(function=configure_nodes)
    ])
import collections