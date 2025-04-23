#importing relevant dependencies for ROS2 to work
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import torch
import numpy as np
import time
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# Importing relevant dependencies for CuRobo to work
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.math import Pose
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState as CuRoboJointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.robot import RobotConfig

import json
from datetime import datetime
import os

def tensor_to_list(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().tolist()
    return tensor

def extract_result_data(result):
    return {
        "success": tensor_to_list(result.success)[0],
        "valid_query": result.valid_query,
        "optimized_dt": tensor_to_list(result.optimized_dt)[0],
        "position_error": tensor_to_list(result.position_error),
        "rotation_error": tensor_to_list(result.rotation_error),
        "solve_time": result.solve_time,
        "ik_time": result.ik_time,
        "trajopt_time": result.trajopt_time,
        "finetune_time": result.finetune_time,
        "total_time": result.total_time,
        "optimized_plan": {
            "joint_names": result.optimized_plan.joint_names,
            "position": tensor_to_list(result.optimized_plan.position),
            "velocity": tensor_to_list(result.optimized_plan.velocity),
            "acceleration": tensor_to_list(result.optimized_plan.acceleration),
            "jerk": tensor_to_list(result.optimized_plan.jerk),
        },
        "interpolated_plan": {
            "joint_names": result.interpolated_plan.joint_names,
            "position": tensor_to_list(result.interpolated_plan.position),
            "velocity": tensor_to_list(result.interpolated_plan.velocity),
            "acceleration": tensor_to_list(result.interpolated_plan.acceleration),
            "jerk": tensor_to_list(result.interpolated_plan.jerk),
        },
        "interpolation_dt": result.interpolation_dt,
        "path_buffer_last_tstep": tensor_to_list(result.path_buffer_last_tstep[0]),
        "goalset_index": tensor_to_list(result.goalset_index),
        "attempts": result.attempts,
        "trajopt_attempts": result.trajopt_attempts,
        "used_graph": result.used_graph
    }

def save_results(result_home_to_pick, result_pick_to_place):
    data = {
        "home_to_pick": extract_result_data(result_home_to_pick),
        "pick_to_place": extract_result_data(result_pick_to_place)
    }

    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"logs/results_{timestamp}.json"
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[INFO] Results saved to {filepath}")


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
        # Load robot model for FK
        self.robot_model = self.load_robot_model()
        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)
        # Initialize motion planner
        # (This is a function that takes into account the world and the robot file)
        self.motion_gen = self.setup_motion_planner()
        self.i = 0
        
        # Generate pick and place trajectories
        self.get_logger().info('Generating pick and place trajectories...')
        self.home_to_pick_traj, self.pick_to_place_traj = self.generate_trajectories()
        
        # Execute trajectories
        self.get_logger().info('Starting trajectory execution...')
        self.execute_trajectories()
        # Publisher for RViz markers


    def load_robot_model(self):
        tensor_args = TensorDeviceType()
        robot_cfg_file = load_yaml(join_path(get_robot_configs_path(), "ur5e_robotiq_2f_140_x.yml"))
        robot_cfg = RobotConfig.from_dict(robot_cfg_file, tensor_args)
        return  CudaRobotModel(robot_cfg.kinematics)

    def publish_marker(self, position, marker_id=0):
        marker = Marker()
        marker.header.frame_id = "bs_link"  # Adjust based on your URDF base
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "ur_tool0"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = Point(x=position[0], y=position[1], z=position[2])
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.marker_pub.publish(marker)

    def setup_motion_planner(self):
        # Set up tensor args and load configurations
        self.tensor_args = TensorDeviceType()
        # world_file = "collision_table.yml"
        # world_config = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
        robot_file = "ur5e_robotiq_2f_140_x.yml"
        
        # Configure motion generator
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_file,
            None,# world_config,
            self.tensor_args,
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
        pick_pos = torch.tensor([[-0.5, -0.5, 0.5]], device=tensor_args.device)
        # Downward-facing orientation (gripper pointing down)
        pick_quat = torch.tensor([self.quaternion_from_euler(np.pi, 0.0, 0.0)], device=tensor_args.device)

        # Place position
        place_pos = torch.tensor([[0.5, 0.5, -0.1]], device=tensor_args.device)
        # Same orientation as pick
        place_quat = torch.tensor([self.quaternion_from_euler(np.pi, 0.0, 0.0)], device=tensor_args.device)
        
        # Create pose objects
        pick_pose = Pose(position=pick_pos, quaternion=pick_quat)
        place_pose = Pose(position=place_pos, quaternion=place_quat)
        self.get_logger().info('Planning: Home to Pick')
        result_home_to_pick = self.motion_gen.plan_single(
            home_state,
            pick_pose,
            MotionGenPlanConfig(max_attempts=3, time_dilation_factor=0.5),
        )

        if not result_home_to_pick.success.item():
            self.get_logger().error(f'Failed to plan home to pick trajectory: {result_home_to_pick.status}')
            raise RuntimeError("Home to pick trajectory planning failed")
        
        pick_state = CuRoboJointState.from_position(
            result_home_to_pick.get_interpolated_plan().position[-1].reshape(1, -1)
        )

        self.get_logger().info('Planning: Pick to Place')
        result_pick_to_place = self.motion_gen.plan_single(
            pick_state,
            place_pose,
            MotionGenPlanConfig(max_attempts=3, time_dilation_factor=0.5),
        )
        
        
        if not result_pick_to_place.success.item():
            self.get_logger().error(f'Failed to plan pick to place trajectory: {result_pick_to_place.status}')
            raise RuntimeError("Pick to place trajectory planning failed")
        
        home_to_pick_traj = result_home_to_pick.get_interpolated_plan()
        pick_to_place_traj = result_pick_to_place.get_interpolated_plan()
        
        self.get_logger().info(f'Home to Pick: Duration={result_home_to_pick.motion_time.item():.2f}s, Points={len(home_to_pick_traj.position)}')
        self.get_logger().info(f'Pick to Place: Duration={result_pick_to_place.motion_time.item():.2f}s, Points={len(pick_to_place_traj.position)}')

        save_results(result_home_to_pick, result_pick_to_place)

        return home_to_pick_traj, pick_to_place_traj
    
    def execute_trajectories(self):
        # Execute home to pick trajectory
        self.get_logger().info('Executing: Home to Pick trajectory')
        self.execute_trajectory(self.home_to_pick_traj)
        self.get_logger().info('At pick position - pausing')
        self.get_logger().info('Executing: Pick to Place trajectory')
        self.execute_trajectory(self.pick_to_place_traj)
        self.get_logger().info('Pick and place operation completed successfully')
    
    def execute_trajectory(self, trajectory):
        positions = trajectory.position.cpu().numpy().reshape(-1, 7)
        
        dt = 0.01  # 10Hz for visualization
        for position in positions:
            # Create and publish joint state message
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joint_names
            # Forward kinematics
            joint_state = torch.tensor(position, device=self.tensor_args.device)
            ee_pose = self.robot_model.get_state(joint_state).ee_position.cpu().numpy()[0]
            self.publish_marker([float(ee_pose[0]), float(ee_pose[1]), float(ee_pose[2])], marker_id=self.i)


            # Add positions (add finger joint value)
            position_list = position.tolist()
            position_list.append(0.0)  # finger joint
            
            msg.position = position_list
            msg.velocity = [0.0] * 8
            msg.effort = [0.0] * 8
            
            self.publisher_.publish(msg)
            time.sleep(dt)
            self.i += 1


def main(args=None):
    rclpy.init(args=args)
    try:
        pick_and_place = UR5ePickPlaceNode()
        rclpy.spin_once(pick_and_place)
        pick_and_place.get_logger().info('Pick and place operation finished')
    except Exception as e:
        print(f"Error during pick and place operation: {e}")
    finally:
        rclpy()


if __name__ == '__main__':
    main()