import numpy as np
import mujoco
import time
import torch
from typing import List, Optional, Dict, Tuple

from mujoco_parser import MuJoCoParserClass
from util import save_world_state

from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, PoseCostMetric
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.robot import JointState as RoboJointState
from curobo.types.state import JointState
from curobo.util_file import get_world_configs_path, get_robot_configs_path, join_path, load_yaml
from curobo.types.base import TensorDeviceType


class BasicMotionPlanner:
    def __init__(self, 
                 xml_path: str, 
                 robot_config_file: str, 
                 world_config_file: str, 
                 tensor_device: str = "cuda:0"):
        """
        Initialize a basic motion planner with MuJoCo visualization.
        
        Args:
            xml_path: Path to the MuJoCo XML file
            robot_config_file: CuRobo robot configuration file name
            world_config_file: CuRobo world configuration file name
            tensor_device: Device to use for tensor computations
        """
        np.set_printoptions(precision=2, suppress=True, linewidth=100)
        self.xml_path = xml_path

        # Initialize MuJoCo environment
        self.env = MuJoCoParserClass(name='Robot', rel_xml_path=xml_path)
        
        # Initialize CuRobo configurations
        self.init_curobo(robot_config_file, world_config_file, tensor_device)
        
        # Initialize robot configuration
        self.robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_config_file))["robot_cfg"]
        self.j_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        self.default_config = self.robot_cfg["kinematics"]["cspace"]["retract_config"]
        
        # Check if the joint names in the robot config match those in MuJoCo
        print("CuRobo joint names:", self.j_names)
        print("MuJoCo joint names:", self.env.rev_joint_names)
        
        # Initialize target positions to be more reachable and avoid self-collision
        self.target_positions = [
            np.array([0.55, -0.4, 0.5]),   # Target 1 (matches original code)
            np.array([0.55, 0.4, 0.5])     # Target 2 (matches original code)
        ]
        # Use the same orientation as in the original code
        self.target_orientation = np.array([0.5, -0.5, 0.5, 0.5])
        self.current_target_idx = 0
        
        # Initialize motion planning variables
        self.current_plan = None
        self.plan_step_idx = 0
        self.cmd_step_idx = 0  # Additional counter to slow down execution
        
        # Initialize the environment
        self.env.reset()
        self.set_robot_config(self.default_config)
        
        # Tensor args for CuRobo
        self.tensor_args = TensorDeviceType(device=tensor_device, dtype=torch.float32)

    def init_curobo(self, 
                   robot_config_file: str, 
                   world_config_file: str, 
                   tensor_device: str = "cuda:0", 
                   n_obstacle_cuboids: int = 10, 
                   n_obstacle_mesh: int = 10):
        """
        Initialize CuRobo motion planning components.
        
        Args:
            robot_config_file: Name of the robot configuration file
            world_config_file: Name of the world configuration file
            tensor_device: Device to use for tensor computations
            n_obstacle_cuboids: Number of obstacle cuboids to cache
            n_obstacle_mesh: Number of obstacle meshes to cache
        """
        self.tensor_args = TensorDeviceType(device=tensor_device, dtype=torch.float32)
        
        # Load world configuration
        self.world_config_initial = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_config_file))
        )
        
        # Lower the table slightly to match original code
        if len(self.world_config_initial.cuboid) > 0:
            self.world_config_initial.cuboid[0].pose[2] -= 0.01
        
        # Configure motion generation
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_config_file))["robot_cfg"]
        
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            self.world_config_initial,
            tensor_args=self.tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
            interpolation_dt=0.02,
            ee_link_name=robot_cfg.get("kinematics", {}).get("ee_link", "tool0"),
        )
        
        # Initialize motion generator
        self.motion_gen = MotionGen(self.motion_gen_config)
        print("Warming up motion generator...")
        self.motion_gen.warmup(warmup_js_trajopt=False)
        print("Motion generator ready.")

    def set_robot_config(self, config):
        """
        Set the robot to a specific configuration.
        
        Args:
            config: Joint configuration to set
        """
        self.env.forward(q=config, joint_idxs=self.env.idxs_forward)

    def get_current_joint_state(self):
        """
        Get the current joint state of the robot.
        
        Returns:
            Current joint state as a CuRobo JointState
        """
        try:
            # Get current joint positions and velocities
            current_position = []
            current_velocity = []
            
            for joint in self.env.rev_joint_names[:len(self.j_names)]:
                joint_idx = self.env.model.joint(joint).qposadr[0]
                vel_idx = self.env.model.joint(joint).dofadr[0]
                
                joint_pos = self.env.data.qpos[joint_idx]
                joint_vel = self.env.data.qvel[vel_idx]
                
                current_position.append(joint_pos)
                current_velocity.append(joint_vel)
            
            # Convert to numpy arrays
            current_position = np.array(current_position)
            current_velocity = np.array(current_velocity)
            
            # Create a JointState with zero acceleration and jerk
            joint_state = JointState(
                position=self.tensor_args.to_device(current_position),
                velocity=self.tensor_args.to_device(current_velocity),
                acceleration=self.tensor_args.to_device(current_velocity) * 0.0,
                jerk=self.tensor_args.to_device(current_velocity) * 0.0,
                joint_names=self.j_names,
            )
            
            return joint_state
        except Exception as e:
            print(f"Error getting joint state: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a zero joint state as a fallback
            zero_position = np.zeros(len(self.j_names))
            return JointState(
                position=self.tensor_args.to_device(zero_position),
                velocity=self.tensor_args.to_device(zero_position),
                acceleration=self.tensor_args.to_device(zero_position),
                jerk=self.tensor_args.to_device(zero_position),
                joint_names=self.j_names,
            )

    def plan_motion_to_target(self):
        """
        Plan a motion to the current target.
        
        Returns:
            True if planning was successful, False otherwise
        """
        try:
            # Get current robot state
            cu_js = self.get_current_joint_state()
            
            # Get the current target position and orientation
            target_position = self.target_positions[self.current_target_idx]
            target_orientation = self.target_orientation
            
            print(f"Planning motion to target: {target_position}")
            
            # Create the IK goal
            ik_goal = Pose(
                position=self.tensor_args.to_device(target_position),
                quaternion=self.tensor_args.to_device(target_orientation)
            )
            
            # Configure planning
            plan_config = MotionGenPlanConfig(
                enable_graph=True,
                enable_graph_attempt=4,
                max_attempts=5,
                enable_finetune_trajopt=True,
                time_dilation_factor=0.5,
            )

            if np.all(target_position == np.array([0.55, 0.4, 0.5])):
                print("I am here")
                pose_cost_metric = PoseCostMetric(
                        hold_partial_pose= True,
                        hold_vec_weight = self.motion_gen.tensor_args.to_device([1, 1, 1, 0, 0, 0]),
                        )
            else:    
                pose_cost_metric = PoseCostMetric(
                            hold_partial_pose=True,
                            hold_vec_weight = self.motion_gen.tensor_args.to_device([0, 0, 0, 0, 0, 0]),
                            )
            
            plan_config.pose_cost_metric = pose_cost_metric
            # Plan the motion
            result = self.motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
            
            # Check if planning was successful
            succ = result.success.item()
            if succ:
                print("Planning successful!")
                # Get the interpolated plan and full joint state
                self.current_plan = result.get_interpolated_plan()
                self.current_plan = self.motion_gen.get_full_js(self.current_plan)
                
                # Print debug information
                print(f"Plan length: {len(self.current_plan.position)}")
                print(f"Plan joint names: {self.current_plan.joint_names}")
                print(f"MuJoCo joint names: {self.env.rev_joint_names}")
                
                # Get only joint names that are in both MuJoCo and CuRobo
                common_js_names = []
                for x in self.env.rev_joint_names:
                    if x in self.current_plan.joint_names:
                        common_js_names.append(x)
                
                print(f"Common joint names: {common_js_names}")
                
                # Order the plan by the common joint names
                self.current_plan = self.current_plan.get_ordered_joint_state(common_js_names)
                
                # Reset counters for execution
                self.plan_step_idx = 0
                self.cmd_step_idx = 0
                
                # Move to the next target for the next planning cycle
                self.current_target_idx = (self.current_target_idx + 1) % len(self.target_positions)
                
                return True
            else:
                print("Motion planning failed")
                return False
                
        except Exception as e:
            print(f"Exception during motion planning: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def step_simulation(self):
        """
        Step the simulation forward with the current plan or generate a new plan.
        
        Returns:
            True if the simulation should continue, False otherwise
        """
        try:
            if self.current_plan is not None:
                # Execute the current plan at a slower pace (like in original code)
                if self.plan_step_idx < len(self.current_plan.position):
                    cmd_state = self.current_plan[self.plan_step_idx]
                    
                    # Apply the command to the robot
                    joint_positions = cmd_state.position.cpu().numpy()
                    self.env.step(ctrl=joint_positions, ctrl_idxs=self.env.idxs_forward)
                    
                    # Increment the command step counter
                    self.cmd_step_idx += 1
                    
                    # Only move to the next plan step every 2 simulation steps (like in original)
                    if self.cmd_step_idx == 2:
                        self.plan_step_idx += 1
                        self.cmd_step_idx = 0
                else:
                    # Plan completed
                    print("Plan execution completed")
                    self.current_plan = None
            else:
                # Generate a new plan
                success = self.plan_motion_to_target()
                if not success:
                    # If planning failed, hold the current position
                    print("Planning failed, holding current position")
                    current_position = np.array([self.env.data.qpos[self.env.model.joint(joint).qposadr[0]] 
                                               for joint in self.env.rev_joint_names[:len(self.j_names)]])
                    self.env.step(ctrl=current_position, ctrl_idxs=self.env.idxs_forward)
                    
                    # Add a small delay to avoid continuous replanning attempts
                    time.sleep(0.5)
                    
            # Always render the current state
            self.env.render()
            
            # Return True to continue the simulation
            return True
            
        except Exception as e:
            print(f"Exception during simulation step: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Try to recover by holding position
            try:
                current_position = np.array([self.env.data.qpos[self.env.model.joint(joint).qposadr[0]] 
                                           for joint in self.env.rev_joint_names[:len(self.j_names)]])
                self.env.step(ctrl=current_position, ctrl_idxs=self.env.idxs_forward)
                self.env.render()
            except:
                pass
                
            # Still return True to continue the simulation
            return True

    def run_simulation(self):
        """
        Run the full simulation loop with visualization.
        """
        # Initialize MuJoCo viewer
        self.env.init_viewer(viewer_title='Basic Motion Planning', viewer_width=1200, viewer_height=800,
                            viewer_hide_menus=False)
        
        # Set viewer parameters for better visibility
        self.env.update_viewer(
            azimuth=160,           # Adjusted for better view
            distance=1.8,          # Closer view
            elevation=-20,         # Slightly higher view
            lookat=[0.4, 0.0, 0.3], # Looking at the workspace center
            VIS_TRANSPARENT=False, 
            VIS_CONTACTPOINT=True,
            contactwidth=0.05, 
            contactheight=0.05, 
            contactrgba=np.array([1, 0, 0, 1]),
            VIS_JOINT=True, 
            jointlength=0.1, 
            jointwidth=0.05, 
            jointrgba=[0.2, 0.6, 0.8, 0.6]
        )
        
        print("Starting simulation...")
        
        # Main simulation loop
        while self.env.is_viewer_alive() and self.env.get_sim_time() < 100.0:
            # Step the simulation
            self.step_simulation()
            
            # Small delay to slow down the simulation for visualization
            time.sleep(0.01)
            
        print("Simulation finished.")


if __name__ == "__main__":
    # Path to MuJoCo XML file
    xml_path = '../assets/ur5e/scene_ur5e_2f140_obj_suction.xml'
  
    # CuRobo configuration files
    robot_config_file = "ur5e.yml"  # Update this to match your robot
    world_config_file = "collision_table.yml"
    
    # Create and run the motion planner
    motion_planner = BasicMotionPlanner(
        xml_path=xml_path,
        robot_config_file=robot_config_file,
        world_config_file=world_config_file
    )
    
    motion_planner.run_simulation()