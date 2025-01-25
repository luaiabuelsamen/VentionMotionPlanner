import yaml
import os
import numpy as np

from pxr import Usd
import mujoco
import time
from mujoco_parser import MuJoCoParserClass
from mujoco.usd import exporter

import torch
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.util.usd_helper import UsdHelper
from curobo.types.robot import JointState as RoboJointState

# Init Mujoco and the exporter
np.set_printoptions(precision=2, suppress=True, linewidth=100)

# Path to the XML model
xml_path = 'assets/ur5e/scene_ur5e.xml'

# Initialize MuJoCo environment
env = MuJoCoParserClass(name='UR5e', rel_xml_path=xml_path)
# Init Curobo
robot_config_file = "ur5e.yml"
world_config_yml = yaml.load(open(os.path.join(os.path.dirname(__file__), 'collision_test.yml')), Loader=yaml.FullLoader)
motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_config_file,
    interpolation_dt=0.01,
    world_model=world_config_yml,
    collision_checker_type=CollisionCheckerType.MESH
)
motion_gen = MotionGen(motion_gen_config)
motion_gen.warmup()
usd_help = UsdHelper()

# plan the motion
# goal_position = [1.57, -1.57/2, 0.0, 0.0, 0.0, 0.0]
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

# Initialize MuJoCo viewer
env.model.body('base').pos = np.array([0, 0, 0])
env.init_viewer(viewer_title='UR5e with RG2 gripper', viewer_width=1200, viewer_height=800,
                viewer_hide_menus=True)
env.update_viewer(azimuth=66.08, distance=3.0, elevation=-50, lookat=[0.4, 0.18, 0.71],
                  VIS_TRANSPARENT=False, VIS_CONTACTPOINT=False,
                  contactwidth=0.05, contactheight=0.05, contactrgba=np.array([1, 0, 0, 1]),
                  VIS_JOINT=True, jointlength=0.25, jointwidth=0.05, jointrgba=[0.2, 0.6, 0.8, 0.6])
env.reset()

def update_curobo(ignore_substring: str, robot_prim_path: str) -> None:
    obstacles = usd_help.get_obstacles_from_stage(
        ignore_substring=ignore_substring, reference_prim_path=robot_prim_path
    ).get_collision_check_world()
    motion_gen.update_world(obstacles)

# output_dir = 'output_usd'
# exporter_instance = exporter.USDExporter(
#     None,
#     output_directory_name=output_dir,
#     output_directory_root='./',
#     verbose=False,
# )

tick = 0
def save_stage( iteration = [1]):
    [os.remove(os.path.join(output_dir + "/frames/", f)) for f in os.listdir(output_dir + "/frames/") if f.endswith(".usd")]
    data = mujoco.MjData(env.model)
    exporter_instance.update_scene(data)
    exporter_instance.save_scene('usd')
    stage = Usd.Stage.Open(f"./{output_dir}/" + f"frames/frame{iteration[0]}.usd")
    iteration[0] += 1
    return stage

try:
    while (env.get_sim_time() < 100.0) and env.is_viewer_alive():
        # usd_stage = save_stage()
        # usd_help.load_stage(usd_stage)
        # update_curobo(ignore_substring="obstacle", robot_prim_path="robot")
        tick += 1
        if tick >= len(positions):
            env.step(ctrl=positions[-1,:], ctrl_idxs=env.idxs_forward)
        else:    
            env.step(ctrl=positions[tick,:], ctrl_idxs=env.idxs_forward)
        env.render()
        time.sleep(0.05)


except KeyboardInterrupt:
    print("Simulation interrupted.")

finally:
    print("Exporter process finished.")