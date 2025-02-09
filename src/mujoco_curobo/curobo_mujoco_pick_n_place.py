import yaml
import os
import numpy as np
import mujoco
import time
import torch
from mujoco_parser import MuJoCoParserClass
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.robot import JointState as RoboJointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from util import sample_xyzs, rpy2r, get_interp_const_vel_traj

# Initialize MuJoCo environment
xml_path = 'assets/ur5e/scene_ur5e_rg2_d435i_obj.xml'
env = MuJoCoParserClass(name='UR5e with RG2 gripper', rel_xml_path=xml_path, VERBOSE=True)

def setup_curobo():
    """ Setup Curobo with the UR5e and Robotiq 2F-140 gripper """
    robot_config_file = "ur5e_robotiq_2f_140.yml"
    world_config = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), "collision_table.yml")))
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_config_file,
        interpolation_dt=0.01,
        world_model=world_config,
        collision_checker_type=CollisionCheckerType.MESH
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()
    return motion_gen

motion_gen = setup_curobo()

# Object setup for pick-and-place
obj_names = [body_name for body_name in env.body_names if body_name.startswith("obj_")]
n_obj = len(obj_names)
xyzs = sample_xyzs(n_sample=n_obj, x_range=[0.4, 0.8], y_range=[-0.2, 0.2], z_range=[0.025, 0.025], min_dist=0.1)

for obj_idx, obj_name in enumerate(obj_names):
    jntadr = env.model.body(obj_name).jntadr[0]
    env.model.joint(jntadr).qpos0[:3] = xyzs[obj_idx, :]

platform_xyz = np.random.uniform([0.6, -0.3, 0.01], [1.0, 0.3, 0.01])
env.model.body('red_platform').pos = platform_xyz
env.model.body('base').pos = np.array([0.18, 0, 0])

# Initial robot pose
q_init_upright = np.array([0, -np.pi/2, 0, 0, np.pi/2, 0])
env.reset()
env.forward(q=q_init_upright, joint_idxs=env.idxs_forward)
R_trgt = rpy2r(np.radians([-180, 0, 90]))

# Pick-and-place motion planning
pick_position = env.get_p_body(obj_names[0])
pick_position[2] += 0.002  # Adjusted for feasibility
pre_grasp_position = pick_position + np.array([0.0, 0.0, 0.05])  # Lift higher for grasp  # Reduce height change

# Plan motion using Curobo
current_position = np.array([env.data.qpos[env.model.joint(joint).qposadr[0]] for joint in env.rev_joint_names[:6]])
start_state = RoboJointState.from_position(torch.tensor(np.array([current_position]), device="cuda:0", dtype=torch.float32))
goal_pose = Pose(
    position=torch.tensor(pre_grasp_position, device="cuda:0", dtype=torch.float32),
    quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda:0", dtype=torch.float32)  # Identity quaternion
)

print("Start State:", start_state.position.cpu().numpy())
print(f"Pick position: {pick_position}")
print(f"Pre-grasp position: {pre_grasp_position}")
dist = np.linalg.norm(pre_grasp_position - current_position[:3])
print(f"Distance to goal: {dist} meters")
max_reach = 1.0  # Approximate UR5e reachability limit
if dist > max_reach:
    print(f"⚠️ Warning: Goal is out of reach! Distance = {dist:.2f}m")
print("Goal Pose:", goal_pose)

result = motion_gen.plan_single(start_state, goal_pose, MotionGenPlanConfig(max_attempts=20))  # Increased attempts

if result.success.item():
    resulting_plan = result.get_interpolated_plan()
    joint_positions = np.hstack([np.array(resulting_plan.position.tolist()), np.full((len(resulting_plan.position), 1), 0.0)])  # Ensure gripper is open at start
    
    # Ensure joint_positions has 7 values (append default gripper value)
    if joint_positions.shape[1] == 6:
        joint_positions = np.hstack([joint_positions, np.full((joint_positions.shape[0], 1), 0.5)])
else:
    print("Motion planning failed. Error code:", result.status)
    print("Motion planning failed. Visualizing current state.")
    joint_positions = np.tile(np.append(current_position, 0.5), (100, 1))  # Maintain last known state

# Debugging: Check shapes before execution
print(f"Joint positions shape: {joint_positions.shape}")
print(f"Control indices shape: {len([0, 1, 2, 3, 4, 5, 6])}")

env.init_viewer(viewer_title='UR5e with RG2 gripper', viewer_width=1200, viewer_height=800, viewer_hide_menus=True)
tick = 0
# Close gripper after reaching grasp position
grasp_index = len(joint_positions) // 2  # Grasp at midpoint
while tick < joint_positions.shape[0]:
    if tick == grasp_index:
        joint_positions[:, 6] = 1.0  # Close gripper
    env.step(ctrl=joint_positions[tick, :], ctrl_idxs=[0, 1, 2, 3, 4, 5, 6])
    env.render()
    tick += 1
    env.step(ctrl=joint_positions[tick, :], ctrl_idxs=[0, 1, 2, 3, 4, 5, 6])
    env.render()
    tick += 1

print("Pick-and-place motion visualization completed.")
