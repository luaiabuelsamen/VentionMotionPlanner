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

robot_config_file = "ur5e_robotiq_2f_140_x.yml"
current_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
goal_position = [0.49, 1.57, -1.57/2, 0.0, 0.0, 0.0, 0.0]
motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_config_file,
    interpolation_dt=0.01,
)
motion_gen = MotionGen(motion_gen_config)
motion_gen.warmup()
start_state = RoboJointState.from_position(torch.tensor([current_position], device="cuda:0"))
goal_pose = RoboJointState.from_position(torch.tensor([goal_position], device="cuda:0"))
result = motion_gen.plan_single_js(
    start_state, goal_pose, MotionGenPlanConfig(max_attempts=1)
)
result.success.item()
resulting_plan = result.get_interpolated_plan()
joint_positions = resulting_plan.position.tolist()
for pos in joint_positions:
    print(pos[0])