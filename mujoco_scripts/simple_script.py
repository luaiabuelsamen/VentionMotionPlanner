import yaml
import os
import numpy as np
import mujoco
import time
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mujoco_parser import MuJoCoParserClass
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.robot import JointState as RoboJointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from util import save_world_state
from curobo.geom.types import Mesh
from curobo.types.base import TensorDeviceType
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.robot import RobotConfig

# Data collection containers
ee_trajectory_data = []     # Store end-effector positions
cycle_times = []            # Store cycle completion times
planning_times_pick = []    # Store planning times for pick motion
planning_times_place = []   # Store planning times for place motion

robot_config_file = "ur5e_robotiq_2f_140_x.yml"
current_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
world_config_inital = WorldConfig.from_dict({
    "cuboid": {
        "floor": {
            "dims": [2.2, 2.2, 0.2], # x, y, z,
            "pose": [0.0, 0.0, -0.14, 1, 0, 0, 0.0] 
        }
    }
})
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

xml_path = '../assets/ur5e/scene_ur5e_obj_gantry_suction.xml'
env = MuJoCoParserClass(name='UR5e with RG2 gripper', rel_xml_path=xml_path, VERBOSE=True)

include = [
    'bs_link',
    'linear_rail',
    'col_1',
    'col_2',
]

mesh_paths = [
    {
      'bs_link': 'assets/ur5e/mesh/gantry/bs_link.STL',
      'linear_rail': 'assets/ur5e/mesh/gantry/linear_rail.STL'
    }
]

new_wrld, meshes = save_world_state(env.model, env.data, include_set=include, mesh_paths=mesh_paths)
world_config = WorldConfig.from_dict(new_wrld)
for mesh in meshes:
    cur_mesh = Mesh(file_path=meshes[mesh]['file_path'], name=mesh, pose=meshes[mesh]['pose'])
    cur_mesh.file_path = meshes[mesh]['file_path']
    new_wrld.add_obstacle(cur_mesh)
world_config.add_obstacle(world_config_inital.cuboid[0])
motion_gen.update_world(world_config)

device = "cuda:0"
tensor_args = TensorDeviceType(device=device, dtype=torch.float32)

goal_positions = [
    [-0.5, -0.7, 0.065, 0.0, 0.0, -1.0, 0.0],  # Pose A (pick)
    [0.5, 0.7, 0.25, 0.0, 0.0, -1.0, 0.0]      # Pose B (place)
]

env.init_viewer(viewer_title='UR5e with RG2 gripper', viewer_width=1200, viewer_height=800,
                viewer_hide_menus=True)
env.update_viewer(azimuth=66.08, distance=3.0, elevation=-50, lookat=[0.4, 0.18, 0.71],
                  VIS_TRANSPARENT=False, VIS_CONTACTPOINT=False,
                  contactwidth=0.05, contactheight=0.05, contactrgba=np.array([1, 0, 0, 1]),
                  VIS_JOINT=True, jointlength=0.25, jointwidth=0.05, jointrgba=[0.2, 0.6, 0.8, 0.6])

robot_cfg_file = load_yaml(join_path(get_robot_configs_path(), "ur5e_robotiq_2f_140_x.yml"))
robot_cfg = RobotConfig.from_dict(robot_cfg_file, tensor_args)
robot_model = CudaRobotModel(robot_cfg.kinematics)

goal_idx = 1
current_state = RoboJointState.from_position(torch.tensor([current_position], device=device))

num_cycles = 20  # Number of pick-place cycles to run
cycle_count = 0

while cycle_count < num_cycles and env.is_viewer_alive():
    
    current_position = []
    for i, ctrl_idx in enumerate(env.ctrl_joint_idxs[:-1]):
        pos = env.data.qpos[env.ctrl_qpos_idxs[i]]
        current_position.append(float(pos))
    current_state = RoboJointState.from_position(torch.tensor([current_position], device=device))
    
    goal_position = goal_positions[goal_idx]
    goal_pose = Pose(
        position=tensor_args.to_device([goal_position[:3]]),
        quaternion=tensor_args.to_device([goal_position[3:]])
    )

    result = motion_gen.plan_single(
        current_state,
        goal_pose,
        MotionGenPlanConfig(max_attempts=10000)
    )

    plan_time = result.total_time
    if goal_idx == 0:
        planning_times_pick.append(plan_time)
        current_goal_name = "Pick"
    else:
        planning_times_place.append(plan_time)
        current_goal_name = "Place"
    
    print(f"Planning time for {current_goal_name}: {plan_time:.4f} seconds")
    
    if result.success.item():
        plan = result.get_interpolated_plan()
        joint_positions = plan.position.tolist()

        for joint in joint_positions:
            joint_state = torch.tensor(joint, device=tensor_args.device)
            ee_pose = robot_model.get_state(joint_state).ee_position.cpu().numpy()[0]
            ee_trajectory_data.append({
                'x': ee_pose[0],
                'y': ee_pose[1],
                'z': ee_pose[2],
                'cycle': cycle_count,
                'goal': current_goal_name
            })
            
            env.step(ctrl=joint, ctrl_idxs=[0, 1, 2, 3, 4, 5, 6])
            env.render()

        motion_time = 0.01 * len(joint_positions)
        print(f"{current_goal_name} motion completed in {motion_time:.4f} seconds")

        goal_idx = 1 - goal_idx

        if goal_idx == 1:
            cycle_times.append(motion_time + planning_times_pick[-1] + planning_times_place[-1])
            print(f"Cycle {cycle_count + 1} completed")
            cycle_count += 1
            motion_time = 0
    else:
        print(f"Failed planning with status: {result.status}")
        print(f"Current state: {current_state}, Goal position: {goal_position}")
        break
    
    env.render()

if hasattr(env, 'close_viewer'):
    env.close_viewer()

if not ee_trajectory_data:
    print("No trajectory data collected. Exiting without plotting.")
    exit()

traj_data = np.array([(d['x'], d['y'], d['z']) for d in ee_trajectory_data])
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for cycle in range(cycle_count):
    cycle_data = [d for d in ee_trajectory_data if d['cycle'] == cycle]
    
    pick_data = [d for d in cycle_data if d['goal'] == "Pick"]
    if pick_data:
        pick_x = [d['x'] for d in pick_data]
        pick_y = [d['y'] for d in pick_data]
        pick_z = [d['z'] for d in pick_data]
        ax.plot(pick_x, pick_y, pick_z, 'r-', linewidth=2, alpha=0.7, 
                label=f"Cycle {cycle+1} Pick" if cycle == 0 else "")
    
    # Plot place trajectory
    place_data = [d for d in cycle_data if d['goal'] == "Place"]
    if place_data:
        place_x = [d['x'] for d in place_data]
        place_y = [d['y'] for d in place_data]
        place_z = [d['z'] for d in place_data]
        ax.plot(place_x, place_y, place_z, 'b-', linewidth=2, alpha=0.7, 
                label=f"Cycle {cycle+1} Place" if cycle == 0 else "")

# Plot the goal positions
ax.scatter([goal_positions[0][0]], [goal_positions[0][1]], [goal_positions[0][2]], 
           color='red', s=100, marker='o', label='Pick Position')
ax.scatter([goal_positions[1][0]], [goal_positions[1][1]], [goal_positions[1][2]], 
           color='blue', s=100, marker='o', label='Place Position')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('End-Effector Trajectory During Pick and Place Operations')
ax.legend()

plt.savefig('ee_trajectory_3d.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate and print average cycle time
if cycle_times:
    avg_cycle_time = np.mean(cycle_times)
    std_cycle_time = np.std(cycle_times)
    print(f"\nPerformance Statistics:")
    print(f"Average Cycle Time: {avg_cycle_time:.4f} ± {std_cycle_time:.4f} seconds")
    
    # Calculate average planning times for pick and place
    if planning_times_pick:
        avg_pick_plan = np.mean(planning_times_pick)
        std_pick_plan = np.std(planning_times_pick)
        print(f"Average Pick Planning Time: {avg_pick_plan:.4f} ± {std_pick_plan:.4f} seconds")
    
    if planning_times_place:
        avg_place_plan = np.mean(planning_times_place)
        std_place_plan = np.std(planning_times_place)
        print(f"Average Place Planning Time: {avg_place_plan:.4f} ± {std_place_plan:.4f} seconds")
    
    # Calculate average total planning time
    all_planning_times = planning_times_pick + planning_times_place
    if all_planning_times:
        avg_planning_time = np.mean(all_planning_times)
        std_planning_time = np.std(all_planning_times)
        print(f"Average Planning Time: {avg_planning_time:.4f} ± {std_planning_time:.4f} seconds")
        print(f"Average Execution Time: {avg_cycle_time - avg_planning_time:.4f} seconds")

    # Plot cycle times
    plt.figure(figsize=(10, 6))
    cycles = range(1, len(cycle_times) + 1)
    plt.bar(cycles, cycle_times, color='skyblue')
    plt.axhline(y=avg_cycle_time, color='r', linestyle='-', label=f'Average: {avg_cycle_time:.2f}s')
    plt.xlabel('Cycle Number')
    plt.ylabel('Time (seconds)')
    plt.title('Pick and Place Cycle Times')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('cycle_times.png', dpi=300)
    plt.close()

    # Create separate plots for pick and place planning times
    if planning_times_pick and planning_times_place:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pick planning times
        pick_cycles = range(1, len(planning_times_pick) + 1)
        ax1.bar(pick_cycles, planning_times_pick, color='coral')
        ax1.axhline(y=avg_pick_plan, color='r', linestyle='-', label=f'Avg: {avg_pick_plan:.2f}s')
        ax1.set_xlabel('Cycle Number')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Pick Planning Times')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Place planning times
        place_cycles = range(1, len(planning_times_place) + 1)
        ax2.bar(place_cycles, planning_times_place, color='skyblue')
        ax2.axhline(y=avg_place_plan, color='r', linestyle='-', label=f'Avg: {avg_place_plan:.2f}s')
        ax2.set_xlabel('Cycle Number')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Place Planning Times')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('planning_times.png', dpi=300)
        plt.close()

# 2D projections of the trajectory
fig, axs = plt.subplots(2, 2, figsize=(16, 14))

# XY Projection
axs[0, 0].set_title('XY Projection')
axs[0, 0].set_xlabel('X (m)')
axs[0, 0].set_ylabel('Y (m)')
axs[0, 0].grid(True)

# XZ Projection
axs[0, 1].set_title('XZ Projection')
axs[0, 1].set_xlabel('X (m)')
axs[0, 1].set_ylabel('Z (m)')
axs[0, 1].grid(True)

# YZ Projection
axs[1, 0].set_title('YZ Projection')
axs[1, 0].set_xlabel('Y (m)')
axs[1, 0].set_ylabel('Z (m)')
axs[1, 0].grid(True)

# 3D view in the last subplot
ax3d = fig.add_subplot(2, 2, 4, projection='3d')
ax3d.set_title('3D Trajectory')
ax3d.set_xlabel('X (m)')
ax3d.set_ylabel('Y (m)')
ax3d.set_zlabel('Z (m)')

for cycle in range(min(cycle_count, 5)):  # Limit to 5 cycles for clarity
    cycle_data = [d for d in ee_trajectory_data if d['cycle'] == cycle]
    
    # Process pick and place separately for coloring
    for goal_type, color, marker in [("Pick", 'red', '-'), ("Place", 'blue', '-')]:
        goal_data = [d for d in cycle_data if d['goal'] == goal_type]
        if goal_data:
            x = [d['x'] for d in goal_data]
            y = [d['y'] for d in goal_data]
            z = [d['z'] for d in goal_data]
            
            # Plot projections
            axs[0, 0].plot(x, y, color=color, linestyle=marker, alpha=0.7)
            axs[0, 1].plot(x, z, color=color, linestyle=marker, alpha=0.7)
            axs[1, 0].plot(y, z, color=color, linestyle=marker, alpha=0.7)
            
            # Plot 3D
            ax3d.plot(x, y, z, color=color, linestyle=marker, alpha=0.7)

# Mark goal positions on all plots
for i, (goal, color, label) in enumerate(zip(goal_positions, ['red', 'blue'], ['Pick', 'Place'])):
    axs[0, 0].scatter(goal[0], goal[1], color=color, s=100, marker='o', label=f'{label} Position')
    axs[0, 1].scatter(goal[0], goal[2], color=color, s=100, marker='o', label=f'{label} Position')
    axs[1, 0].scatter(goal[1], goal[2], color=color, s=100, marker='o', label=f'{label} Position')
    ax3d.scatter(goal[0], goal[1], goal[2], color=color, s=100, marker='o', label=f'{label} Position')

# Add legends
axs[0, 0].legend()
axs[0, 1].legend()
axs[1, 0].legend()
ax3d.legend()

plt.tight_layout()
plt.savefig('trajectory_projections.png', dpi=300)
plt.close()

print("Analysis complete! Plots saved to disk.")