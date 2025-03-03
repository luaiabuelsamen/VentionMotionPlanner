# Third Party
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

def demo_motion_gen():
    PLOT = True
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"
    world_config = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
    robot_file = "ur5e_robotiq_2f_140_gantry.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_config,
        tensor_args,
        interpolation_dt=0.01,
        # trajopt_dt=0.15,
        # velocity_scale=0.1,
        use_cuda_graph=True,
        # finetune_dt_scale=2.5,
        interpolation_steps=10000,
    )

    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()
    retract_cfg = motion_gen.get_retract_config()
    state = motion_gen.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )

    start_state = JointState.from_position(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=tensor_args.device))
    goal_state = JointState.from_position(torch.tensor([[0.5, 0.0, -2.2, 1.0, -1.383, -1.57, 0.0]], device=tensor_args.device))

    result = motion_gen.plan_single_js(
        start_state,
        goal_state,
        MotionGenPlanConfig(max_attempts=1, time_dilation_factor=0.5),
    )
    new_result = result.clone()
    new_result.retime_trajectory(0.5, create_interpolation_buffer=True)
    print(new_result.optimized_dt, new_result.motion_time, result.motion_time)
    print(
    "Trajectory Generated: ",
    result.success,
    result.solve_time,
    result.status,
    result.optimized_dt)

    if result.success.item():
        traj = result.get_interpolated_plan()
    return traj