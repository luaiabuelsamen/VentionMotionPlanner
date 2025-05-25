import numpy as np
import mujoco
import time
import torch
import math
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

# Initialize device and tensor args
device = "cuda:0"
tensor_args = TensorDeviceType(device=device, dtype=torch.float32)

# Robot configuration
robot_config_file = "ur5e_robotiq_2f_140_x.yml"
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
    step_dt=0.03,
)
mpc = MpcSolver(mpc_config)

# Initialize MuJoCo environment
xml_path = '../assets/ur5e/scene_ur5e_obj_gantry_suction.xml'
env = MuJoCoParserClass(name='UR5e with RG2 gripper', rel_xml_path=xml_path, VERBOSE=True)

# Print available MuJoCo sites and bodies (helpful for debugging)
print(f"_site:{env.model.nsite}")
print(f"site_names:{[env.model.site(i).name for i in range(env.model.nsite)]}")
print(f"body_names:{[env.model.body(i).name for i in range(env.model.nbody)]}")

# Objects to track
include = ['bs_link', 'linear_rail', 'col_1', 'col_2', 'col_3', 'col_4', 'cube_1', 'cube_2']
mesh_paths = [{
    'bs_link': 'assets/ur5e/mesh/gantry/bs_link.STL',
    'linear_rail': 'assets/ur5e/mesh/gantry/linear_rail.STL'
}]

# Function to move objects in a pattern
def move_objects(t):
    x = 0.2 * math.cos(t/4)
    env.model.body('col_2').pos = np.array([x, -0.5, 0.23])

# Update world function - now called inline in main loop
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

# Setup robot model for FK
robot_model = CudaRobotModel(robot_cfg.kinematics)

# Initialize viewer
env.init_viewer(viewer_title='UR5e MPC Demo', viewer_width=1200, viewer_height=800,
                viewer_hide_menus=True)
env.update_viewer(azimuth=66.08, distance=3.0, elevation=-50, lookat=[0.4, 0.18, 0.71],
                  VIS_TRANSPARENT=False, VIS_CONTACTPOINT=False,
                  VIS_JOINT=True)

# Initial world update
update_world()

# Define goal positions
goal_positions = [
    [-0.5, -0.7, 0.25, 0.0, 0.0, -1.0, 0.0],  # Pose A (pick)
    [0.5, 0.7, 0.25, 0.0, 0.0, -1.0, 0.0]     # Pose B (place)
]

try:
    goal_idx = 0
    num_cycles = 5
    last_update_time = time.time()
    update_interval = 0.2  # Update world every 0.2 seconds
    start_time = time.time()
    
    for cycle in range(num_cycles):
        if not env.is_viewer_alive():
            break
            
        print(f"Starting Cycle {cycle+1}")
        
        # Get current robot position
        current_position = []
        for i, ctrl_idx in enumerate(env.ctrl_joint_idxs[:-1]):
            pos = env.data.qpos[env.ctrl_qpos_idxs[i]]
            current_position.append(float(pos))
            
        # Set up goal
        goal_position = goal_positions[goal_idx]
        goal_pose = Pose(
            position=tensor_args.to_device([goal_position[:3]]),
            quaternion=tensor_args.to_device([goal_position[3:]])
        )
        
        # Run MPC - Create a joint state with joint names
        start_state = RoboJointState.from_position(
            torch.tensor([current_position], device=device), 
            joint_names=joint_names  # Pass joint names here
        )
        
        # Create a goal for the MPC
        goal = Goal(
            current_state=start_state,
            goal_pose=goal_pose
        )
        
        # Setup the MPC solver
        goal_buffer = mpc.setup_solve_single(goal, 1)
        mpc.update_goal(goal_buffer)
        
        converged = False
        tstep = 0
        
        print(f"Running MPC to goal {goal_idx+1}")
        
        while not converged and tstep < 300 and env.is_viewer_alive():
            # Calculate time for object movement
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Move objects
            move_objects(elapsed)
            
            # Update world occasionally in the main thread
            if current_time - last_update_time > update_interval:
                update_world()
                last_update_time = current_time
            
            # Update current state
            current_position = []
            for i, ctrl_idx in enumerate(env.ctrl_joint_idxs[:-1]):
                pos = env.data.qpos[env.ctrl_qpos_idxs[i]]
                current_position.append(float(pos))
            
            # Create joint state with joint names
            current_state = RoboJointState.from_position(
                torch.tensor([current_position], device=device),
                joint_names=joint_names  # Pass joint names here
            )
            
            # Run MPC step with error handling
            try:
                torch.cuda.synchronize()  # Ensure previous CUDA operations are complete
                result = mpc.step(current_state, 1)
                torch.cuda.synchronize()  # Ensure MPC step is complete
                
                # Execute action
                joint_position = result.action.position.cpu().numpy()[0]
                env.step(ctrl=joint_position, ctrl_idxs=[0, 1, 2, 3, 4, 5, 6])
                
                # Check convergence
                if result.metrics.pose_error.item() < 0.05:
                    converged = True
                    print(f"MPC converged in {tstep} steps")
            except Exception as e:
                print(f"Error in MPC step: {e}")
                break
            
            tstep += 1
            env.render()
        
        # Switch goal for next cycle
        goal_idx = 1 - goal_idx
        
except KeyboardInterrupt:
    print("Interrupted by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    print("Finished")