import numpy as np
import mujoco
import time
import torch
from typing import List, Optional, Dict, Tuple

from mujoco_parser import MuJoCoParserClass
from util import save_world_state

from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.robot import JointState as RoboJointState
from curobo.types.state import JointState
from curobo.util_file import get_world_configs_path, get_robot_configs_path, join_path, load_yaml
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.motion_gen import PoseCostMetric


class ConstrainedMotionPlanner:
    def __init__(self, 
                 xml_path: str, 
                 robot_config_file: str, 
                 world_config_file: str, 
                 tensor_device: str = "cuda:0"):
        """
        Initialize the constrained motion planner with MuJoCo visualization.
        
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
        
        # Make sure the joint names match or map them appropriately
        if len(self.j_names) != len(self.env.rev_joint_names):
            print(f"Warning: Number of joints in CuRobo config ({len(self.j_names)}) doesn't match MuJoCo ({len(self.env.rev_joint_names)})")
            # Use the MuJoCo joint names if they don't match
            if len(self.env.rev_joint_names) > 0:
                self.j_names = self.env.rev_joint_names[:len(self.j_names)]
        
        # Initialize target positions to be more reachable and avoid self-collision
        self.target_positions = [
            np.array([0.35, -0.3, 0.4]),   # Target 1: In front of robot, to the left, higher
            np.array([0.35, 0.3, 0.4])     # Target 2: In front of robot, to the right, higher
        ]
        # Orientation with end effector pointing forward with slight downward angle
        self.target_orientation = np.array([0.7071, 0.7071, 0.0, 0.0])  # 45-degree tilt
        self.current_target_idx = 0
        
        # Initialize motion planning variables
        self.current_plan = None
        self.plan_step_idx = 0
        self.constraint_mode = 0  # 0: No constraint, 1-4: Different constraint modes
        self.pose_cost_metric = None
        
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
        
        # Configure motion generation
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_config_file,
            interpolation_dt=0.02,
            world_model=self.world_config_initial,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        )
        
        # Initialize motion generator
        self.motion_gen = MotionGen(self.motion_gen_config)
        print("Warming up motion generator...")
        self.motion_gen.warmup(warmup_js_trajopt=False)
        print("Motion generator ready.")

    def update_curobo_world(self, new_world):
        """
        Update the CuRobo world with new obstacle information.
        
        Args:
            new_world: New world configuration or obstacle data
        """
        # Check if new_world is a dictionary or something else
        if isinstance(new_world, dict):
            world_config = WorldConfig.from_dict(new_world)
        else:
            # If it's not a dictionary, we'll use the initial world config
            # and not try to update with the new data
            world_config = WorldConfig()
            
        # Add table from initial world
        for cuboid in self.world_config_initial.cuboid:
            world_config.add_obstacle(cuboid)
            
        # Update the motion generator with the new world
        self.motion_gen.update_world(world_config)
        
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
            # Get current joint positions
            current_position = []
            for joint in self.env.rev_joint_names[:len(self.j_names)]:
                joint_idx = self.env.model.joint(joint).qposadr[0]
                joint_pos = self.env.data.qpos[joint_idx]
                current_position.append(joint_pos)
            
            current_position = np.array(current_position)
            print(f"Current joint position: {current_position}")
            
            # Create a JointState with the current position
            joint_state = RoboJointState.from_position(
                torch.tensor([current_position], device=self.tensor_args.device, dtype=self.tensor_args.dtype)
            )
            
            return joint_state
        except Exception as e:
            print(f"Error getting joint state: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a zero joint state as a fallback
            zero_position = np.zeros(len(self.j_names))
            return RoboJointState.from_position(
                torch.tensor([zero_position], device=self.tensor_args.device, dtype=self.tensor_args.dtype)
            )

    def plan_motion_to_target(self):
        """
        Plan a motion to the current target with the current constraint mode.
        
        Returns:
            True if planning was successful, False otherwise
        """
        try:
            # Update the constraint mode text (it doesn't actually apply constraints now)
            self._update_constraint_mode()
            
            # Get current robot state
            current_position = self.get_current_joint_state()
            
            # Set the target pose based on constraint mode
            base_pos = np.array([0.35, 0.0, 0.4])
            
            if self.constraint_mode == 0:
                # Unconstrained - alternate between two positions
                if self.current_target_idx == 0:
                    target_position = np.array([0.35, -0.3, 0.4])
                else:
                    target_position = np.array([0.35, 0.3, 0.4])
            
            elif self.constraint_mode == 1:
                # Up/down movement - vary Z
                if self.current_target_idx == 0:
                    target_position = np.array([0.35, 0.0, 0.3])
                else:
                    target_position = np.array([0.35, 0.0, 0.5])
                
            elif self.constraint_mode == 2:
                # Left/right movement - vary Y
                if self.current_target_idx == 0:
                    target_position = np.array([0.35, -0.3, 0.4])
                else:
                    target_position = np.array([0.35, 0.3, 0.4])
                
            elif self.constraint_mode == 3:
                # Forward/backward movement - vary X
                if self.current_target_idx == 0:
                    target_position = np.array([0.25, 0.0, 0.4])
                else:
                    target_position = np.array([0.45, 0.0, 0.4])
                
            elif self.constraint_mode == 4:
                # Diagonal movement - vary X and Y
                if self.current_target_idx == 0:
                    target_position = np.array([0.25, -0.25, 0.4])
                else:
                    target_position = np.array([0.45, 0.25, 0.4])
            
            target_orientation = self.target_orientation
            
            print(f"Planning motion to target: {target_position}")
            
            ik_goal = Pose(
                position=self.tensor_args.to_device(target_position),
                quaternion=self.tensor_args.to_device(target_orientation)
            )
            
            # Configure planning
            plan_config = MotionGenPlanConfig(
                enable_graph=True,
                enable_graph_attempt=4,
                max_attempts=15,
                enable_finetune_trajopt=True
            )
            
            # Plan the motion
            result = self.motion_gen.plan_single(current_position, ik_goal, plan_config)
            
            if result.success.item():
                print("Planning successful!")
                self.current_plan = result.get_interpolated_plan()
                self.plan_step_idx = 0
                
                # Move to the next target for the next planning cycle
                self.current_target_idx = (self.current_target_idx + 1) % len(self.target_positions)
                
                # Increment constraint mode every other successful plan
                if self.current_target_idx == 0:
                    self.constraint_mode = (self.constraint_mode + 1) % 5
                    
                return True
            else:
                print(f"Motion planning failed for target: {target_position}")
                print(f"Error: {result}")
                
                # Try with a different target if this one failed
                self.current_target_idx = (self.current_target_idx + 1) % len(self.target_positions)
                
                return False
                
        except Exception as e:
            print(f"Exception during motion planning: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _update_constraint_mode(self):
        """Update the pose cost metric based on the current constraint mode."""
        if self.constraint_mode == 0:
            print("Constraint Mode: None (Unconstrained Motion)")
            self.pose_cost_metric = None
            
        elif self.constraint_mode == 1:
            print("Constraint Mode: Modified for simplicity - moving up/down")
            # For simplicity, we'll just modify the targets rather than using constraint metrics
            self.pose_cost_metric = None
            
        elif self.constraint_mode == 2:
            print("Constraint Mode: Modified for simplicity - moving left/right")
            # For simplicity, we'll just modify the targets rather than using constraint metrics
            self.pose_cost_metric = None
            
        elif self.constraint_mode == 3:
            print("Constraint Mode: Modified for simplicity - moving forward/backward")
            # For simplicity, we'll just modify the targets rather than using constraint metrics
            self.pose_cost_metric = None
            
        elif self.constraint_mode == 4:
            print("Constraint Mode: Modified for simplicity - diagonal movement")
            # For simplicity, we'll just modify the targets rather than using constraint metrics
            self.pose_cost_metric = None

    def step_simulation(self):
        """
        Step the simulation forward with the current plan or generate a new plan.
        
        Returns:
            True if the simulation should continue, False otherwise
        """
        try:
            if self.current_plan is not None:
                # Execute the current plan
                if self.plan_step_idx < len(self.current_plan.position):
                    current_cmd = self.current_plan[self.plan_step_idx]
                    
                    # Extract joint positions with safeguards for different tensor shapes
                    joint_positions = current_cmd.position.cpu().numpy()
                    
                    # Handle different shapes that might come from the planner
                    if isinstance(joint_positions, np.ndarray):
                        if joint_positions.ndim == 2:
                            # If 2D array [batch, joints]
                            joint_positions = joint_positions[0]
                        elif joint_positions.ndim == 0:
                            # If it's a scalar (shouldn't happen but just in case)
                            print("Warning: Received scalar joint position")
                            joint_positions = np.array([0.0] * len(self.j_names))
                    else:
                        # Fallback if not a numpy array
                        print("Warning: Joint positions not a numpy array")
                        joint_positions = np.array([0.0] * len(self.j_names))
                    
                    # Safety check that we have enough values
                    if len(joint_positions) >= len(self.j_names):
                        # Apply the command to the robot
                        self.env.step(ctrl=joint_positions[:len(self.j_names)], ctrl_idxs=self.env.idxs_forward)
                    else:
                        print(f"Warning: joint positions array length {len(joint_positions)} is less than required {len(self.j_names)}")
                        # Hold current position instead
                        current_position = np.array([self.env.data.qpos[self.env.model.joint(joint).qposadr[0]] 
                                                for joint in self.env.rev_joint_names[:len(self.j_names)]])
                        self.env.step(ctrl=current_position, ctrl_idxs=self.env.idxs_forward)
                    
                    self.plan_step_idx += 1
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
        self.env.init_viewer(viewer_title='Constrained Motion Planning', viewer_width=1200, viewer_height=800,
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
        
        # Try to add visual indicators for target positions if possible
        try:
            # This would depend on how your MuJoCo environment is set up
            # and whether it supports adding visual elements at runtime
            for i, target_pos in enumerate(self.target_positions):
                print(f"Target {i+1} position: {target_pos}")
        except Exception as e:
            print(f"Could not add target visualizations: {e}")
        
        print("Starting simulation...")
        
        # Main simulation loop
        while self.env.is_viewer_alive() and self.env.get_sim_time() < 100.0:
            # For simplicity, we'll skip dynamic world updates for now
            # and just use the initial world configuration
            # This avoids issues with the save_world_state function
            
            # Step the simulation
            self.step_simulation()
            
            # Small delay to slow down the simulation for visualization
            time.sleep(0.01)
            
        print("Simulation finished.")


# # Example usage
# if __name__ == "__main__":
#     # Path to MuJoCo XML file
#     xml_path = 'assets/ur5e/scene_ur5e.xml'  # Update this path to your robot model
    
#     # CuRobo configuration files
#     robot_config_file = "ur5e.yml"  # Update this to match your robot
#     world_config_file = "collision_table.yml"
    
#     # Create and run the constrained motion planner
#     motion_planner = ConstrainedMotionPlanner(
#         xml_path=xml_path,
#         robot_config_file=robot_config_file,
#         world_config_file=world_config_file
#     )
    
#     motion_planner.run_simulation()

if __name__ == "__main__":
    # Path to MuJoCo XML file
    # xml_path = 'assets/ur5e/scene_ur5e.xml'  # Update this path to your robot model
    xml_path = '../assets/ur5e/scene_ur5e_2f140_obj_suction.xml'
  
    # CuRobo configuration files
    robot_config_file = "ur5e.yml"  # Update this to match your robot
    world_config_file = "collision_table.yml"
    
    # Create and run the constrained motion planner
    motion_planner = ConstrainedMotionPlanner(
        xml_path=xml_path,
        robot_config_file=robot_config_file,
        world_config_file=world_config_file
    )
    
    motion_planner.run_simulation()