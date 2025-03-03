#importing relevant dependencies for ROS2 to work
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import torch
import numpy as np
import time

# Importing relevant dependencies for CuRobo to work
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.math import Pose
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState as CuRoboJointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

class UR5ePickPlaceNode(Node):
    def __init__(self):
        super().__init__('ur5e_pick_place_node')
        # Set up publisher to publish the joint_states in the JointState topic for RViz2
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        
        # Robot joint names (this is for the Robot with gripper and the gantry base)
        self.joint_names = [
            "base_x",
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "finger_joint"
        ]
        
        # Initialize motion planner
        # (This is a function that takes into account the world and the robot file)
        self.motion_gen = self.setup_motion_planner()
        
        # Generate pick and place trajectories
        self.get_logger().info('Generating pick and place trajectories...')
        self.home_to_pick_traj, self.pick_to_place_traj = self.generate_trajectories()
        
        # Execute trajectories
        self.get_logger().info('Starting trajectory execution...')
        self.execute_trajectories()

    def setup_motion_planner(self):
        # Set up tensor args and load configurations
        tensor_args = TensorDeviceType()
        world_file = "collision_table.yml"
        world_config = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
        robot_file = "ur5e_robotiq_2f_140_gantry.yml"
        
        # Configure motion generator
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_file,
            world_config,
            tensor_args,
            interpolation_dt=0.01,
            use_cuda_graph=True,
            interpolation_steps=10000,
        )
        
        # Create and warm up motion generator
        motion_gen = MotionGen(motion_gen_config)
        motion_gen.warmup()
        
        return motion_gen
    
    def quaternion_from_euler(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return [qw, qx, qy, qz]
    
    def generate_trajectories(self):
        tensor_args = TensorDeviceType()
        
        # Define joint positions
        home_position = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=tensor_args.device)
        # pick_position = torch.tensor([[-0.5, -0.5, -1.5, 1.0, -1.0, -1.57, 0.0]], device=tensor_args.device)
        # place_position = torch.tensor([[0.5, 0.5, -1.5, 1.5, -1.2, -1.57, 0.0]], device=tensor_args.device)
        
        # Create joint states
        home_state = CuRoboJointState.from_position(home_position)
        # pick_state = CuRoboJointState.from_position(pick_position)
        # place_state = CuRoboJointState.from_position(place_position)
        
        # Pick position
        pick_pos = torch.tensor([[-0.5, -0.5, -0.1]], device=tensor_args.device)
        # Downward-facing orientation (gripper pointing down)
        pick_quat = torch.tensor([self.quaternion_from_euler(np.pi, 0.0, 0.0)], device=tensor_args.device)

        # Place position
        place_pos = torch.tensor([[0.5, 0.5, -0.1]], device=tensor_args.device)
        # Same orientation as pick
        place_quat = torch.tensor([self.quaternion_from_euler(np.pi, 0.0, 0.0)], device=tensor_args.device)
        
        # Create pose objects
        pick_pose = Pose(position=pick_pos, quaternion=pick_quat)
        place_pose = Pose(position=place_pos, quaternion=place_quat)
        
        # Plan trajectory from home to pick
        # self.get_logger().info('Planning: Home to Pick')
        # result_home_to_pick = self.motion_gen.plan_single_js(
        #     home_state,
        #     pick_state,
        #     MotionGenPlanConfig(max_attempts=3, time_dilation_factor=0.5),
        # )

        # Plan trajectory from home to pick
        self.get_logger().info('Planning: Home to Pick')
        result_home_to_pick = self.motion_gen.plan_single(
            home_state,
            pick_pose,
            MotionGenPlanConfig(max_attempts=3, time_dilation_factor=0.5),
        )

        if not result_home_to_pick.success.item():
            self.get_logger().error(f'Failed to plan home to pick trajectory: {result_home_to_pick.status}')
            raise RuntimeError("Home to pick trajectory planning failed")
        
        # Get final state from pre-pick to pick trajectory
        pick_state = CuRoboJointState.from_position(
            result_home_to_pick.get_interpolated_plan().position[-1].reshape(1, -1)
        )
        
        # Plan trajectory from pick to place
        # self.get_logger().info('Planning: Pick to Place')
        # result_pick_to_place = self.motion_gen.plan_single_js(
        #     pick_state,
        #     place_state,
        #     MotionGenPlanConfig(max_attempts=3, time_dilation_factor=0.5),
        # )

        self.get_logger().info('Planning: Pick to Place')
        result_pick_to_place = self.motion_gen.plan_single(
            pick_state,
            place_pose,
            MotionGenPlanConfig(max_attempts=3, time_dilation_factor=0.5),
        )
        
        
        if not result_pick_to_place.success.item():
            self.get_logger().error(f'Failed to plan pick to place trajectory: {result_pick_to_place.status}')
            raise RuntimeError("Pick to place trajectory planning failed")
        
        # Get interpolated trajectories
        home_to_pick_traj = result_home_to_pick.get_interpolated_plan()
        pick_to_place_traj = result_pick_to_place.get_interpolated_plan()
        
        # Log trajectory details
        self.get_logger().info(f'Home to Pick: Duration={result_home_to_pick.motion_time.item():.2f}s, Points={len(home_to_pick_traj.position)}')
        self.get_logger().info(f'Pick to Place: Duration={result_pick_to_place.motion_time.item():.2f}s, Points={len(pick_to_place_traj.position)}')
        
        return home_to_pick_traj, pick_to_place_traj
    
    def execute_trajectories(self):
        # Execute home to pick trajectory
        self.get_logger().info('Executing: Home to Pick trajectory')
        self.execute_trajectory(self.home_to_pick_traj)
        
        # Pause at pick position (simulating gripper operation)
        self.get_logger().info('At pick position - pausing')
        time.sleep(3.0)
        
        # Execute pick to place trajectory
        self.get_logger().info('Executing: Pick to Place trajectory')
        self.execute_trajectory(self.pick_to_place_traj)
        
        self.get_logger().info('Pick and place operation completed successfully')
    
    def execute_trajectory(self, trajectory):
        positions = trajectory.position.cpu().numpy().reshape(-1, 7)
        
        # Create timer frequency to match trajectory timing
        dt = 0.12  # 10Hz for visualization
        
        for i, position in enumerate(positions):
            # Skip frames to reduce frequency if needed
            if i % 2 != 0:  # Only publish every 10th point
                continue
                
            # Create and publish joint state message
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joint_names
            
            # Add positions (add finger joint value)
            position_list = position.tolist()
            position_list.append(0.0)  # finger joint
            
            msg.position = position_list
            msg.velocity = [0.0] * 8
            msg.effort = [0.0] * 8
            
            self.publisher_.publish(msg)
            time.sleep(dt)


def main(args=None):
    rclpy.init(args=args)
    try:
        pick_and_place = UR5ePickPlaceNode()
        rclpy.spin_once(pick_and_place)
        pick_and_place.get_logger().info('Pick and place operation finished')
    except Exception as e:
        print(f"Error during pick and place operation: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()