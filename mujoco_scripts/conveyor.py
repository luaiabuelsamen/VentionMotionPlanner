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
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, PoseCostMetric
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.robot import JointState as RoboJointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from util import save_world_state
from curobo.geom.types import Mesh
from curobo.types.base import TensorDeviceType
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.robot import RobotConfig
from curobo.geom.sphere_fit import SphereFitType

# Data collection containers
ee_trajectory_data = []     # Store end-effector positions
cycle_times = []            # Store cycle completion times
planning_times_pick = []    # Store planning times for pick motion
planning_times_place = []   # Store planning times for place motion

robot_config_file = "ur5e_x.yml"
current_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
world_config_inital = WorldConfig.from_dict({
    "cuboid": {
        "floor": {
            "dims": [2.2, 2.2, 0.1], # x, y, z,
            "pose": [0.0, 0.0, -0.13, 1, 0, 0, 0.0] 
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
    # Robot components that need tracking
    'bs_link', 
    'linear_rail',
    
    # Conveyor system
    'conveyor_belt',
    # 'roller_1',
    # 'roller_2',
    
    # Moving boxes on conveyor
    'box_1', 
    'box_2', 
    'box_3', 
    'box_4', 
    'box_5',
    
    # Pallet and storage area
    'pallet',
]

mesh_paths = [
    {
      'bs_link': 'assets/ur5e/mesh/gantry/bs_link.STL',
      'linear_rail': 'assets/ur5e/mesh/gantry/linear_rail.STL'
    }
]

new_wrld, meshes = save_world_state(env.model, env.data, include_set=include, mesh_paths=mesh_paths)
world_config_initial = WorldConfig.from_dict(new_wrld)

device = "cuda:0"
tensor_args = TensorDeviceType(device=device, dtype=torch.float32)

pick_position = [-0.5, -0.7, 0.4, 0.0, 0.0, -1.0, 0.0]  # Fixed pick position
place_base_position = [0.5, 0.9, 0.27, 0.0, 0.0, -1.0, 0.0]  # Base place position

box_joint_indices = [7, 8, 9, 10, 11]  # Joint indices for 5 boxes
box_count = 5
box_shift_speed = 1.1 # You can tune this

picked_boxes = set()

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
cycle_count = 0
num_cycles = 5  # Pick all 5 boxes


def update_world(ignore = None):

    new_wrld, meshes = save_world_state(env.model, env.data, include_set=[obj for obj in include if obj != ignore], mesh_paths=mesh_paths)
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

picked = None

update_world(ignore = picked)
while cycle_count < num_cycles and env.is_viewer_alive():
    current_position = []
    for i in range(7):
        pos = env.data.qpos[env.ctrl_qpos_idxs[i]]
        current_position.append(float(pos))
    current_state = RoboJointState.from_position(torch.tensor([current_position], device=device))
    if goal_idx == 0:
        goal_position = pick_position
    else:
        # Offset along a grid: 2 rows x 3 columns (modify as needed)
        row = cycle_count // 3
        col = cycle_count % 3
        dx = 0.3 * col  # spacing in X
        dy = -0.3 * row  # spacing in Y

        goal_position = [
            place_base_position[0] + dx,
            place_base_position[1] + dy,
            place_base_position[2],
            place_base_position[3],
            place_base_position[4],
            place_base_position[5],
            place_base_position[6],
        ]
    goal_pose = Pose(
        position=tensor_args.to_device([goal_position[:3]]),
        quaternion=tensor_args.to_device([goal_position[3:]])
    )
    

    pose_cost = PoseCostMetric(
        hold_partial_pose= True,
        hold_vec_weight=tensor_args.to_device([1, 1, 1, 0, 0, 0]), # this should set cost on EE
    )
    result = motion_gen.plan_single(
        current_state,
        goal_pose,
        MotionGenPlanConfig(max_attempts=10000, 
                            # pose_cost_metric= pose_cost)
        )
    )

    plan_time = result.total_time
    current_goal_name = "Pick" if goal_idx == 0 else "Place"

    print(f"Planning time for {current_goal_name}: {plan_time:.4f} seconds")
    if goal_idx == 0:
        picked_boxes.add(cycle_count)
        
    if result.success.item():
        plan = result.get_interpolated_plan()
        joint_positions = plan.position.tolist()

        for joint in joint_positions:
            env.step(ctrl=joint, ctrl_idxs=[0, 1, 2, 3, 4, 5, 6])
            env.render()

            if goal_idx == 1:
                if len(picked_boxes):
                    if env.get_p_body(f'box_{list(picked_boxes)[-1] + 2}')[0] <= -0.5:
                        env.step(ctrl=[box_shift_speed] * (5-len(picked_boxes)) , ctrl_idxs=[i for i in range(8 + len(picked_boxes), 13)])
                update_world(ignore = picked)
        if goal_idx == 1:
            if len(picked_boxes):
                while env.get_p_body(f'box_{list(picked_boxes)[-1] + 2}')[0] <= -0.5:
                    env.step(ctrl=[box_shift_speed] * (5-len(picked_boxes)) , ctrl_idxs=[i for i in range(8 + len(picked_boxes), 13)])
                    env.render()
                    
                env.step(ctrl=[0] * (5-len(picked_boxes)) , ctrl_idxs=[i for i in range(8 + len(picked_boxes), 13)])
                env.render()
            print("detatching", f"box_{cycle_count + 1}")
            motion_gen.detach_object_from_robot()
        else:
            print("attatching", f"box_{cycle_count + 1}")
            picked = f"box_{cycle_count + 1}"
            motion_gen.attach_objects_to_robot(
                current_state,
                [f"box_{cycle_count + 1}"],
                sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
                world_objects_pose_offset=Pose.from_list([0, 0, 0.1, 1, 0, 0, 0], tensor_args),
            )

        
        goal_idx = 1 - goal_idx

        motion_time = 0.01 * len(joint_positions)
        print(f"{current_goal_name} motion completed in {motion_time:.4f} seconds")


        if goal_idx == 1:
            cycle_count += 1
            print(f"Cycle {cycle_count} completed")

        for i in range(150):
            env.step(ctrl=[goal_idx], ctrl_idxs=[7])
            env.render()
    else:
        print(f"Failed planning with status: {result.status}")
        print(f"Current state: {current_state}, Goal position: {goal_position}")
        break

    env.render()
