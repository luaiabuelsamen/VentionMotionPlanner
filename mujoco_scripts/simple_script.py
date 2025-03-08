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
from util import save_world_state
from curobo.geom.types import Mesh

robot_config_file = "ur5e_robotiq_2f_140_x.yml"
current_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# goal_position = [0.99, 1.57, -1.57/2, 0.0, 0.0, 0.0, 0.0]
world_config_inital = WorldConfig.from_dict({
    "cuboid": {
        "floor": {
            "dims": [2.2, 2.2, 0.2], # x, y, z,
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

xml_path = '../assets/ur5e/scene_ur5e_2f140_obj_gantry.xml'
env = MuJoCoParserClass(name='UR5e with RG2 gripper', rel_xml_path=xml_path, VERBOSE=True)


#DYNAMICALLY AVOID MESH USING MUJOCO
#WORK IN PROGRESS
new_wrld, meshes = save_world_state(env.model, env.data, include_set=["bs_link"])
print(new_wrld)
# assert False
world_config = WorldConfig.from_dict(new_wrld)
# for mesh in meshes:
#     world_config.add_obstacle(mesh)
stl_file_path = "/home/jetson3/ros2_ws/src/mujoco_curobo/assets/ur5e/mesh/gantry/bs_link.STL"
mesh = Mesh(file_path=stl_file_path, name="example_mesh", pose=[0.0, 0.0, -0.12, 1.0, 0.0, 0.0, 0.0])
mesh.file_path = stl_file_path
world_config.add_obstacle(mesh)
world_config.add_obstacle(world_config_inital.cuboid[0])
# motion_gen.update_world(world_config)

# start_state = RoboJointState.from_position(torch.tensor([current_position], device="cuda:0"))
# goal_pose = RoboJointState.from_position(torch.tensor([goal_position], device="cuda:0"))
# result = motion_gen.plan_single_js(
#     start_state, goal_pose, MotionGenPlanConfig(max_attempts=1)
# )
# if not result.success.item():
#     print(result)
#     assert False
# resulting_plan = result.get_interpolated_plan()
# joint_positions = resulting_plan.position.tolist()
joint_positions = []
env.init_viewer(viewer_title='UR5e with RG2 gripper', viewer_width=1200, viewer_height=800,
                viewer_hide_menus=True)
env.update_viewer(azimuth=66.08, distance=3.0, elevation=-50, lookat=[0.4, 0.18, 0.71],
                  VIS_TRANSPARENT=False, VIS_CONTACTPOINT=False,
                  contactwidth=0.05, contactheight=0.05, contactrgba=np.array([1, 0, 0, 1]),
                  VIS_JOINT=True, jointlength=0.25, jointwidth=0.05, jointrgba=[0.2, 0.6, 0.8, 0.6])

traj = True
while (env.get_sim_time() < 100.0) and env.is_viewer_alive():
    if traj:
        for pos in joint_positions:
            traj = False
            env.step(ctrl=pos, ctrl_idxs=[0, 1, 2, 3, 4, 5, 6])
    env.render()