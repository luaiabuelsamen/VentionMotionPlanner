import yaml
import os
import numpy as np

import mujoco
import time
from mujoco_parser import MuJoCoParserClass

import torch
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.robot import JointState as RoboJointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml

# Init Mujoco and the exporter
np.set_printoptions(precision=2, suppress=True, linewidth=100)

# Path to the XML model
xml_path = 'assets/ur5e/scene_ur5e.xml'
env = MuJoCoParserClass(name='UR5e', rel_xml_path=xml_path)

# Init Curobo
robot_config_file = "ur5e.yml"
world_config_inital = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), "collision_table.yml")))
n_obstacle_cuboids = 20
n_obstacle_mesh = 2
motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_config_file,
    interpolation_dt=0.01,
    world_model=world_config_inital,
    collision_checker_type=CollisionCheckerType.MESH,
    collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
)
motion_gen = MotionGen(motion_gen_config)
motion_gen.warmup()

# Initialize MuJoCo viewer
env.model.body('base').pos = np.array([0, 0, 0])
env.init_viewer(viewer_title='UR5e with RG2 gripper', viewer_width=1200, viewer_height=800,
                viewer_hide_menus=True)
env.update_viewer(azimuth=66.08, distance=3.0, elevation=-50, lookat=[0.4, 0.18, 0.71],
                  VIS_TRANSPARENT=False, VIS_CONTACTPOINT=False,
                  contactwidth=0.05, contactheight=0.05, contactrgba=np.array([1, 0, 0, 1]),
                  VIS_JOINT=True, jointlength=0.25, jointwidth=0.05, jointrgba=[0.2, 0.6, 0.8, 0.6])
env.reset()

def update_curobo(new_wrld) -> None:
    print(new_wrld)
    world_config = WorldConfig.from_dict(new_wrld)
    world_config.add_obstacle(world_config_inital.cuboid[0])
    motion_gen.update_world(world_config)

def save_world_state(model, data, ignore_set):
    stage = {}
    for i in range(model.nbody):
        body_name_start = model.name_bodyadr[i]
        body_name = model.names[body_name_start:].split(b'\x00', 1)[0].decode('utf-8')
        
        if body_name and body_name not in ignore_set:
            body_pos = data.xpos[i]
            body_quat = data.xquat[i]

            for j in range(model.ngeom):
                if model.geom_bodyid[j] == i:
                    geom_type = model.geom_type[j]
                    geom_size = model.geom_size[j]
                    
                    geom_name_start = model.name_geomadr[j]
                    geom_name = model.names[geom_name_start:].split(b'\x00', 1)[0].decode('utf-8')
                    pose = list(body_pos) + list(body_quat)
                    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                        sphere_rad = geom_size[0]
                        stage.setdefault('sphere', {})[body_name] = {
                            "radius": sphere_rad,
                            "pose": pose,
                            "color": [0.5, 0.5, 0.5, 1.0]
                        }
                    elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                        # Here is where you adjust the position to the edge
                        cube_size = [2 * s for s in geom_size[:3]]  # Convert half-size to full-size
                        
                        # Adjust the position to the edge (subtract half the size in z)
                        pose[2] += cube_size[2] / 2   # Correct for edge alignment along z-axis
                        
                        stage.setdefault('cuboid', {})[body_name] = {
                            "dims": cube_size,
                            "pose": pose,
                            "color": [0.8, 0.3, 0.3, 1.0]
                        }
                    elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                        mesh_file_start = model.name_meshadr[model.geom_dataid[j]]
                        mesh_file = model.names[mesh_file_start:].split(b'\x00', 1)[0].decode('utf-8')
                        stage.setdefault('mesh', {})[geom_name] = {
                            "pose": pose,
                            "file_path": mesh_file,
                            "color": [0.7, 0.7, 0.7, 1.0]
                        }
    return stage

ignore = set(['base',
'shoulder_link',
'upper_arm_link',
'forearm_link',
'wrist_1_link',
'wrist_2_link',
'wrist_3_link',
'table'])
stage = save_world_state(env.model, env.data, ignore)
update_curobo(stage)


# plan the motion
goal_position = [1.57, -1.57/2, 0.0, 0.0, 0.0, 0.0]
goal_position = [0.4, -0.4, 0.4, 1.0, 0.0, 0.0, 0.0]
current_position = np.array([env.data.qpos[env.model.joint(joint).qposadr[0]] for joint in env.rev_joint_names[:6]])
start_state = RoboJointState.from_position(torch.tensor(np.array([current_position]), device="cuda:0", dtype=torch.float32))
# goal_pose = RoboJointState.from_position(torch.tensor([goal_position], device="cuda:0", dtype=torch.float32))
goal_pose = Pose.from_list(goal_position)
result = motion_gen.plan_single(start_state, goal_pose, MotionGenPlanConfig(max_attempts=1))
if result.success.item():
    resulting_plan = result.get_interpolated_plan()
    joint_positions = resulting_plan.position.tolist()
else:
    print("Motion planning failed.")
    exit()
resulting_plan = result.get_interpolated_plan()
positions = np.array(resulting_plan.position.tolist())

tick = 0
try:
    while (env.get_sim_time() < 100.0) and env.is_viewer_alive():
        tick += 1
        if tick >= len(positions):
            env.step(ctrl=positions[-1,:], ctrl_idxs=env.idxs_forward)
        else:    
            env.step(ctrl=positions[tick,:], ctrl_idxs=env.idxs_forward)
        env.render()

except KeyboardInterrupt:
    print("Simulation interrupted.")

finally:
    print("Exporter process finished.")