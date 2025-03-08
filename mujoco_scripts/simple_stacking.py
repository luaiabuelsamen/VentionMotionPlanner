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


class StateMachine:
    def __init__(self):
        self.states = ["open", "pick", "close", "place"]
        self.current_state_index = 0

    def next_state(self):
        self.current_state_index = (self.current_state_index + 1) % len(self.states)
        print(self.states[self.current_state_index])
        return self.states[self.current_state_index]
    def wait(self):
        self.current_state_index = 4
    @property
    def current_state(self):
        return self.states[self.current_state_index]

class UR5eMotionPlanner:
    def __init__(self, xml_path: str, robot_config_file: str, world_config_file: str, tensor_device: str = "cuda:0"):
        np.set_printoptions(precision=2, suppress=True, linewidth=100)

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
        print(self.posns)
        self.platform_xyz = np.random.uniform([-0.5, -0.5, 0.01], [-0.5, -0.5, 0.01])
        self.env.model.body('rail').pos = np.array([0, 0, 0])
        self.q_init_upright = np.array([0, -np.pi / 2, 0, 0, np.pi / 2, 0])
        self.env.reset()
        self.env.forward(q=self.q_init_upright, joint_idxs=self.env.idxs_forward)
        self.cur_plan = None
        self.place = [-0.50, 0.0, 0.2]
        
        # Initialize the robot position
        self.current_position = RoboJointState.from_position(torch.tensor(np.array([self.q_init_upright]), 
                                                                        device=tensor_device, dtype=torch.float32))
        self.cur_box_name = None
    
    def init_curobo(self, robot_config_file:str, world_config_file:str, tensor_device: str = "cuda:0", 
                    n_obstacle_cuboids: int = 20, n_obstacle_mesh: int = 2, ):
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
            use_cuda_graph=True,
            self_collision_check=True,
        )
        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.warmup()
        
    def update_curobo(self, new_wrld) -> None:
        world_config = WorldConfig.from_dict(new_wrld)
        # world_config.add_obstacle(self.world_config_inital.cuboid[0])
        for obstacle in self.world_config_inital.cuboid:
            world_config.add_obstacle(obstacle)
        self.motion_gen.update_world(world_config)
    
    def update_curobo(self, new_world_state) -> None:
        """Update cuRobo's world model with the new environment state."""
        world_config = WorldConfig.from_dict(new_world_state)
        self.motion_gen.update_world(world_config)

    def plan_motion(self, goal_position):
        print(f"plan generated for {goal_position}")
        goal_pose = Pose(
            position=self.tensor_args.to_device([goal_position]),
            quaternion=self.tensor_args.to_device([[0, 0, -1, 0]])  
        )
        current_position = RoboJointState.from_position(
            torch.tensor([list(self.get_curpos())], device="cuda:0", dtype=torch.float32))
        current_position.unsqueeze(0)
        result = self.motion_gen.plan_single(current_position, goal_pose, MotionGenPlanConfig(max_attempts=5000))
        if result.success.item():
            resulting_plan = result.get_interpolated_plan()
            self.cur_plan = np.array(resulting_plan.position.tolist())
            return True
        else:
            print(f"Motion planning failed for position: at {goal_position}")
            print(f"Error: {result.message}")
            return False

    def get_curpos(self):
        return np.array([self.env.data.qpos[self.env.model.joint(joint).qposadr[0]] for joint in self.env.rev_joint_names[:6]])
    
    def step_plan(self):
        # print(f"Currently in {self.sm.current_state}")
        world_state, _ = save_world_state(self.env.model, self.env.data, include_set=self.obj_names)
        self.update_curobo(world_state)
        if not isinstance(self.cur_plan, np.ndarray):
            if self.sm.current_state == "open":
                curpos = self.get_curpos()
                curpos = np.append(curpos, 0)
                self.cur_plan =  np.tile(curpos, (200, 1))
                self.cur_plan[:50, -1] = 1
                self.cur_plan[50:, -1] = np.linspace(1, 0, 150)
                self.sm.next_state()
                if self.cur_box_name:
                     self.detach_obj()
                     self.cur_box_name = None
            elif self.sm.current_state == "close":
                curpos = self.get_curpos()
                curpos = np.append(curpos, 1)
                self.cur_plan =  np.tile(curpos, (200, 1))
                self.cur_plan[:50, -1] = 0
                self.cur_plan[50:, -1] = np.linspace(0, 1, 150)
    
                # self.sm.wait()
                self.sm.next_state()
                # self.attach_obj(curpos, self.cur_box_name)
            elif self.sm.current_state == "pick":
                if self.posns:
                    pick, self.cur_box_name = self.posns.popleft()
                    pick[2] += 0.09
                    self.plan_motion(pick)
                    print(self.cur_plan)
                    self.sm.next_state()
            elif self.sm.current_state == "place":
                self.plan_motion(self.place)
                self.place[2] += 0.15
                self.sm.next_state()

    def run_simulation(self):
        # Initialize MuJoCo viewer
        self.env.init_viewer(viewer_title='UR5e with RG2 gripper', viewer_width=1200, viewer_height=800,
                             viewer_hide_menus=True)
        self.env.update_viewer(azimuth=66.08, distance=3.0, elevation=-50, lookat=[0.4, 0.18, 0.71],
                               VIS_TRANSPARENT=False, VIS_CONTACTPOINT=False,
                               contactwidth=0.05, contactheight=0.05, contactrgba=np.array([1, 0, 0, 1]),
                               VIS_JOINT=True, jointlength=0.25, jointwidth=0.05, jointrgba=[0.2, 0.6, 0.8, 0.6])
        tick = 0
        while (self.env.get_sim_time() < 100.0) and self.env.is_viewer_alive():
            stage, _ = save_world_state(self.env.model, self.env.data, include_set=self.obj_names)
            self.update_curobo(stage)
            # print(np.array([self.env.data.qpos[6]]))
            if isinstance(self.cur_plan, np.ndarray):
                if len(self.cur_plan[0]) != 6:
                    self.env.step(ctrl=self.cur_plan[tick, :], ctrl_idxs=range(0, 7))
                else:
                    self.env.step(ctrl=self.cur_plan[tick, :6], ctrl_idxs=self.env.idxs_forward)
                tick += 1
                if tick >= len(self.cur_plan):
                    tick = 0
                    self.cur_plan = None
                    print("Plan finished")
            self.step_plan()
            self.env.render()

    def attach_obj(
        self,
        sim_js,
        cube_name):
        print(f"Attaching {cube_name} to robot")

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
    xml_path = '../assets/ur5e/scene_ur5e_2f140_obj_suction.xml'
    robot_config_file = "ur5e.yml"
    world_config_file = "collision_table.yml"
    
    motion_planner = UR5eMotionPlanner(xml_path=xml_path, 
                                      robot_config_file=robot_config_file, 
                                      world_config_file=world_config_file)
    motion_planner.run_simulation()
