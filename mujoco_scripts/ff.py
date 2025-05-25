import numpy as np
import mujoco
import time
import torch
import math
import threading
from mujoco_parser import MuJoCoParserClass
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Mesh
from curobo.types.robot import JointState as RoboJointState
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.types.base import TensorDeviceType
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.robot import RobotConfig
from curobo.rollout.rollout_base import Goal
from util import save_world_state
from curobo.geom.sphere_fit import SphereFitType

# Initialize device and tensor args
device = "cuda:0"
tensor_args = TensorDeviceType(device=device, dtype=torch.float32)

# Robot configuration
robot_config_file = "ur5e_x_suction.yml"
robot_cfg_file = load_yaml(join_path(get_robot_configs_path(), robot_config_file))
robot_cfg = RobotConfig.from_dict(robot_cfg_file, tensor_args)

joint_names = [
    "base_x",
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

world_config_initial = WorldConfig.from_dict({
    "cuboid": {
        "floor": {
            "dims": [2.2, 2.2, 0.2],
            "pose": [0.0, 0.0, -0.14, 1, 0, 0, 0.0]
        }
    }
})

# Initialize MotionGen and MPC
n_obstacle_cuboids = 20
n_obstacle_mesh = 2

# Setting use_cuda_graph to False to avoid CUDA graph capture conflicts
motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_config_file,
    interpolation_dt=0.01,
    world_model=world_config_initial,
    collision_checker_type=CollisionCheckerType.MESH,
    use_cuda_graph=False,  # Disable CUDA graph
    collision_activation_distance=0.0001,
    collision_max_outside_distance=0.0001,
    collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
)
motion_gen = MotionGen(motion_gen_config)
motion_gen.warmup()

mpc_config = MpcSolverConfig.load_from_robot_config(
    robot_config_file,
    world_config_initial,
    store_rollouts=True,
    use_cuda_graph=False,  # Disable CUDA graph 
    step_dt=0.02,
)
mpc = MpcSolver(mpc_config)

# Initialize MuJoCo environment
xml_path = '../assets/ur5e/scene_ur5e_obj_gantry_suction.xml'
env = MuJoCoParserClass(name='UR5e with RG2 gripper', rel_xml_path=xml_path, VERBOSE=True)

# Objects to track
include = [
    # Robot components that need tracking
    'bs_link', 
    'linear_rail',
    
    # Conveyor system
    'conveyor_belt',
    'roller_1',
    'roller_2',
    
    # Moving boxes on conveyor
    'box_1', 
    'box_2', 
    'box_3', 
    'box_4', 
    'box_5',
    
    # Pallet and storage area
    'pallet',
]

mesh_paths = [{
    'bs_link': 'assets/ur5e/mesh/gantry/bs_link.STL',
    'linear_rail': 'assets/ur5e/mesh/gantry/linear_rail.STL'
}]

# Global flag to control the conveyor thread
conveyor_running = True
# Setup robot model for FK
robot_model = CudaRobotModel(robot_cfg.kinematics)

# Initialize viewer
env.init_viewer(viewer_title='UR5e MPC Demo', viewer_width=1200, viewer_height=800,
                viewer_hide_menus=True)
env.update_viewer(azimuth=66.08, distance=3.0, elevation=-50, lookat=[0.4, 0.18, 0.71],
                  VIS_TRANSPARENT=False, VIS_CONTACTPOINT=False,
                  VIS_JOINT=True)

stack_state = {
    'available_boxes': ['box_1', 'box_2', 'box_3', 'box_4', 'box_5'],
    'current_box': 'box_1',
    'placed_positions': [], # List of (x,y) tuples that are already occupied on pallet
    'move': False
}


def get_goal(goal_idx, env):
    # Default orientation (gripper facing downward)
    orientation = [0.0, 0.0, -1.0, 0.0]
    
    if goal_idx == 0:
        selected_box = stack_state['current_box']
        box_pos = env.model.body(stack_state['current_box']).pos.copy()
        stack_state['available_boxes'].remove(selected_box)
        pick_pos = [box_pos[0], box_pos[1], box_pos[2] + 0.25]
        stack_state['move'] = False
        stack_state['current_box'] = stack_state['available_boxes'][0]
        return pick_pos + orientation
    
    else:
        pallet_pos = env.model.body('pallet').pos.copy()
        grid_size = 0.15  # Distance between placement positions
        pallet_positions = []
        
        # Create a 3x3 grid of positions on the pallet
        for i in range(3):
            for j in range(3):
                x = pallet_pos[0] - grid_size + (i * grid_size)
                y = pallet_pos[1] - grid_size + (j * grid_size)
                # Pallet top surface z + distance for box height + grip height
                z = pallet_pos[2] + 0.05 + 0.3 
                pallet_positions.append((x, y, z))
        
        # Filter out positions that are already occupied
        available_positions = [pos for pos in pallet_positions 
                              if (pos[0], pos[1]) not in stack_state['placed_positions']]
        
        if not available_positions:
            # If pallet is full, clear it and start again
            stack_state['placed_positions'] = []
            available_positions = pallet_positions
        
        # Choose the first available position
        place_pos = available_positions[0]
        
        # Add to placed positions
        stack_state['placed_positions'].append((place_pos[0], place_pos[1]))
        stack_state['move'] = True
        return list(place_pos) + orientation

def update_world():
    new_wrld, meshes = save_world_state(env.model, env.data, include_set=include, mesh_paths=mesh_paths)
    world_config = WorldConfig.from_dict(new_wrld)
    
    for mesh in meshes:
        cur_mesh = Mesh(file_path=meshes[mesh]['file_path'], name=mesh, pose=meshes[mesh]['pose'])
        cur_mesh.file_path = meshes[mesh]['file_path']
        world_config.add_obstacle(cur_mesh)
    
    world_config.add_obstacle(world_config_initial.cuboid[0])
    
    # Synchronize CUDA operations before updating world
    torch.cuda.synchronize()
    motion_gen.update_world(world_config)
    torch.cuda.synchronize()
    mpc.update_world(world_config)
    torch.cuda.synchronize()

goal_idx = 0
num_cycles = 10
last_update_time = time.time()
start_time = time.time()
update_world()

for cycle in range(num_cycles):
    if not env.is_viewer_alive():
        break
        
    print(f"Starting Cycle {cycle+1}")
    
    # Get current robot position
    current_position = []
    for i, ctrl_idx in enumerate(env.ctrl_joint_idxs[:7]):
        pos = env.data.qpos[env.ctrl_qpos_idxs[i]]
        current_position.append(float(pos))
        
    # Set up goal
    goal_position = get_goal(goal_idx, env)
    print(goal_position)
    goal_pose = Pose(
        position=tensor_args.to_device([goal_position[:3]]),
        quaternion=tensor_args.to_device([goal_position[3:]])
    )
    
    start_state = RoboJointState.from_position(
        torch.tensor([current_position], device=device), 
        joint_names=joint_names  # Pass joint names here
    )
    
    goal = Goal(
        current_state=start_state,
        goal_pose=goal_pose
    )
    


    if False:
        # Setup the MPC solver
        goal_buffer = mpc.setup_solve_single(goal, 1)
        mpc.update_goal(goal_buffer)
        
        converged = False
        tstep = 0
        
        print(f"Running MPC to goal {goal_idx+1}")
        
        while not converged and env.is_viewer_alive():
            
            # Update current state
            current_position = []
            for i, ctrl_idx in enumerate(env.ctrl_joint_idxs[:7]):
                pos = env.data.qpos[env.ctrl_qpos_idxs[i]]
                current_position.append(float(pos))
            
            # Create joint state with joint names
            current_state = RoboJointState.from_position(
                torch.tensor([current_position], device=device),
                joint_names=joint_names  # Pass joint names here
            )
            
            # Run MPC step with error handling
            torch.cuda.synchronize()  # Ensure previous CUDA operations are complete
            result = mpc.step(current_state, 1)
            torch.cuda.synchronize()  # Ensure MPC step is complete
            
            # Execute action
            joint_position = result.action.position.cpu().numpy()[0]
            env.step(ctrl=joint_position, ctrl_idxs=[0, 1, 2, 3, 4, 5, 6])
            # if stack_state['move'] and env.model.body(stack_state['current_box']).pos[0] <= -0.5:
            #     env.step(ctrl=[1, 1, 1, 1, 1], ctrl_idxs=[7, 8, 9, 10, 11])
            # else:
            #     env.step(ctrl=[0, 0, 0, 0, 0], ctrl_idxs=[7, 8, 9, 10, 11])
            # Check convergence
            if result.metrics.pose_error.item() < 0.35:

                converged = True
                print(f"MPC converged in {tstep} steps")
                
            tstep += 1
            env.render()

    else:
        result = motion_gen.plan_single(start_state, goal_pose, MotionGenPlanConfig(max_attempts=10000))
        if result.success.item():
            joints = result.get_interpolated_plan().position.tolist()
            print(len(joints))
            for joint_position in joints:
                env.step(ctrl=joint_position, ctrl_idxs=[0, 1, 2, 3, 4, 5, 6])
                env.render()

                current_position = []
                for i, ctrl_idx in enumerate(env.ctrl_joint_idxs[:7]):
                    pos = env.data.qpos[env.ctrl_qpos_idxs[i]]
                    current_position.append(float(pos))
                
                # Create joint state with joint names
                current_state = RoboJointState.from_position(
                    torch.tensor([current_position], device=device),
                    joint_names=joint_names  # Pass joint names here
                )
        else:
            print(result)
            
    # Switch goal for next cycle
    goal_idx = 1 - goal_idx
    for i in range(100):
        env.step(ctrl=[goal_idx], ctrl_idxs=[7])
        env.render()
    if goal_idx == 1:
        motion_gen.attach_objects_to_robot(
            current_state,
            [stack_state["current_box"]],
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            world_objects_pose_offset=Pose.from_list([0, 0, 0.1, 1, 0, 0, 0], tensor_args),
        )
    else:
        motion_gen.detach_object_from_robot()