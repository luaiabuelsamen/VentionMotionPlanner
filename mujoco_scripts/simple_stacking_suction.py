import yaml
import os
import numpy as np
import mujoco
import time
from mujoco_parser import MuJoCoParserClass
from util import save_world_state
import torch
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.robot import JointState as RoboJointState
from curobo.types.state import JointState
from curobo.util_file import get_world_configs_path, join_path, load_yaml
from curobo.types.base import TensorDeviceType
from curobo.geom.sphere_fit import SphereFitType
from collections import deque
import xml.etree.ElementTree as ET


class StateMachine:
    def __init__(self):
        self.states = ["approach", "pick", "lift", "move", "lower", "place"]
        self.current_state_index = 0

    def next_state(self):
        self.current_state_index = (self.current_state_index + 1) % len(self.states)
        return self.states[self.current_state_index]
    
    @property
    def current_state(self):
        return self.states[self.current_state_index]

class UR5eMotionPlanner:
    def __init__(self, xml_path: str, robot_config_file: str, world_config_file: str, tensor_device: str = "cuda:0"):
        np.set_printoptions(precision=2, suppress=True, linewidth=100)
        self.xml_path = xml_path  # Store the XML path

        # Initialize MuJoCo and Curobo configurations
        self.env = MuJoCoParserClass(name='UR5e', rel_xml_path=xml_path)
        self.init_curobo(robot_config_file, world_config_file)
        self.sm = StateMachine()    

        # Set up initial positions of cubes and environment
        self.obj_names = [body_name for body_name in self.env.body_names if body_name.startswith("cube")]
        self.posns = deque([])
        self.n_obj = len(self.obj_names)
        for obj_name in self.obj_names:
            jntadr = self.env.model.body(obj_name).jntadr[0]
            self.posns.append((self.env.model.joint(jntadr).qpos0[:3], obj_name))
        
        # Initialize robot and environment
        self.platform_xyz = np.random.uniform([-0.5, -0.5, 0.01], [-0.5, -0.5, 0.01])
        # self.env.model.body('rail').pos = np.array([0, 0, 0.1])
        self.q_init_upright = np.array([0, -np.pi / 2, 0, 0, np.pi / 2, 0])
        self.env.reset()
        self.env.forward(q=self.q_init_upright, joint_idxs=self.env.idxs_forward)
        
        # Initialize planning variables
        self.cur_plan = None
        self.place = [0.50, -0.50, 0.05]
        self.current_position = RoboJointState.from_position(
            torch.tensor(np.array([self.q_init_upright]), device=tensor_device, dtype=torch.float32))
        self.cur_box_name = None
        
        # Vacuum gripper parameters
        self.gripper_offset = 0.1  # Length of vacuum gripper from wrist_3
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        print("MuJoCo model loaded successfully.")

    def init_curobo(self, robot_config_file:str, world_config_file:str, tensor_device: str = "cuda:0", 
                    n_obstacle_cuboids: int = 20, n_obstacle_mesh: int = 2):
        self.tensor_args = TensorDeviceType(device=tensor_device, dtype=torch.float32)

        self.robot_config_file = robot_config_file
        self.world_config_inital = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_config_file)))
        self.n_obstacle_cuboids = n_obstacle_cuboids
        self.n_obstacle_mesh = n_obstacle_mesh
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_config_file,
            interpolation_dt=0.01,
            world_model=self.world_config_inital,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={"obb": self.n_obstacle_cuboids, "mesh": self.n_obstacle_mesh},
        )
        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.warmup()

    def update_curobo(self, new_wrld) -> None:
        world_config = WorldConfig.from_dict(new_wrld)
        world_config.add_obstacle(self.world_config_inital.cuboid[0])
        self.motion_gen.update_world(world_config)

    def plan_motion(self, goal_position, approach_offset=0.0):
        goal_position[2] += 0.05
        # Add gripper offset and any approach offset for pre-grasp positioning
        offset = np.array([0.0, 0.0, -0.015])
        goal_position = np.array(goal_position) + offset
        print("Planned XYZ waypoint:", goal_position.tolist())


        goal_pose = Pose(
            position=self.tensor_args.to_device([goal_position]),
            quaternion=self.tensor_args.to_device([[0, 0, -1, 0]])  
        )
        current_position = RoboJointState.from_position(
            torch.tensor([list(self.get_curpos())], device="cuda:0", dtype=torch.float32))
        current_position.unsqueeze(0)
        result = self.motion_gen.plan_single(current_position, goal_pose, MotionGenPlanConfig(max_attempts=10000))
        
        if result.success.item():
            resulting_plan = result.get_interpolated_plan()
            self.cur_plan = np.array(resulting_plan.position.tolist())
        else:
            print(f"Motion planning failed for position: at {goal_position}")
            print(result)
            exit()

    def get_curpos(self):
        return np.array([self.env.data.qpos[self.env.model.joint(joint).qposadr[0]] for joint in self.env.rev_joint_names[:7]])

    def step_plan(self):
        if not isinstance(self.cur_plan, np.ndarray):
            if self.sm.current_state == "approach":
                if self.posns:
                    print("Planning approach")
                    self.pick_pos, self.cur_box_name = self.posns.popleft()
                    self.plan_motion(self.pick_pos, approach_offset=0.2)  # Add offset for approach
                    self.sm.next_state()
                    
            elif self.sm.current_state == "pick":
                print("Planning pick")
                self.plan_motion(self.pick_pos)  # Move to exact position
                # Extract cube number from name (e.g., "cube_1" -> "cube1")
                # Attach object to robot in Curobo
                cube_name = self.cur_box_name.replace("_", "")
                #sim_js = self.get_curpos()
                #self.attach_obj()
                #APPEND 1
                self.sm.next_state()
                
            elif self.sm.current_state == "lift":
                print("Planning lift")
                lift_pos = self.pick_pos + np.array([0, 0, 0.1])
                self.plan_motion(lift_pos)
                self.sm.next_state()
                
            elif self.sm.current_state == "move":
                print("Planning move to place position")
                self.plan_motion(self.place, approach_offset=0.05)
                self.sm.next_state()
                
            elif self.sm.current_state == "lower":
                print("Planning lower")
                self.plan_motion(self.place)
                self.sm.next_state()
                
            elif self.sm.current_state == "place":
                print("Placing")
                # Extract cube number from name (e.g., "cube_1" -> "cube1")
                cube_name = self.cur_box_name.replace("_", "")

                #APPEND 0
                #self.detach_obj()
                self.place[2] += 0.06  # Increment height for next cube
                curpos = self.get_curpos()
                self.cur_plan = np.tile(curpos, (100, 1))  # Hold position briefly
                self.sm.next_state()

    def run_simulation(self):
        # Initialize MuJoCo viewer
        self.env.init_viewer(viewer_title='UR5e with vacuum gripper', viewer_width=1200, viewer_height=800,
                             viewer_hide_menus=True)
        self.env.update_viewer(azimuth=66.08, distance=3.0, elevation=-50, lookat=[0.4, 0.18, 0.71],
                               VIS_TRANSPARENT=False, VIS_CONTACTPOINT=False,
                               contactwidth=0.05, contactheight=0.05, contactrgba=np.array([1, 0, 0, 1]),
                               VIS_JOINT=True, jointlength=0.25, jointwidth=0.05, jointrgba=[0.2, 0.6, 0.8, 0.6])
        
        tick = 0
        while (self.env.get_sim_time() < 100.0) and self.env.is_viewer_alive():
            stage = save_world_state(self.env.model, self.env.data, include_set=self.obj_names)
            # self.update_curobo(stage)
            
            if isinstance(self.cur_plan, np.ndarray):
                self.env.step(ctrl=self.cur_plan[tick, :6], ctrl_idxs=self.env.idxs_forward)
                tick += 1
                if tick >= len(self.cur_plan):
                    tick = 0
                    self.cur_plan = None
                    print("Plan finished")
            
            self.step_plan()
            self.env.render()

    def attach_obj(self, sim_js: JointState, cube_name: str) -> None:
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js),
        )

        self.motion_gen.attach_objects_to_robot(
            cu_js,
            [cube_name],
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            world_objects_pose_offset=Pose.from_list([0, 0, 0.01, 1, 0, 0, 0], self.tensor_args),
        )

    def detach_obj(self) -> None:
        self.motion_gen.detach_object_from_robot()


# Usage example:
if __name__ == "__main__":
    xml_path = '/home/jetson3/ros2_ws/assets/ur5e/scene_ur5e_2f140_obj_gantry.xml'
    robot_config_file = "ur5e.yml"
    world_config_file = "collision_table.yml"
    
    motion_planner = UR5eMotionPlanner(xml_path=xml_path, 
                                      robot_config_file=robot_config_file, 
                                      world_config_file=world_config_file)
    motion_planner.run_simulation()