#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Quaternion
from moveit_msgs.msg import DisplayTrajectory, RobotState
from moveit_msgs.action import MoveGroup
from std_msgs.msg import Header
import numpy as np
import time
import math
import tf2_ros
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class UR5ePickPlaceMoveitNode(Node):
    def __init__(self):
        super().__init__('ur5e_pick_place_moveit')
        
        # Set up publisher to publish the joint_states for RViz2
        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 10)
        
        # Set up MoveIt2 action client
        self.moveit_action_client = ActionClient(self, MoveGroup, 'move_action')
        
        # Set up publisher for MoveIt trajectory visualization
        self.display_trajectory_publisher = self.create_publisher(
            DisplayTrajectory, 'display_planned_path', 10)
        
        # Robot joint names (removed finger joint)
        self.joint_names = [
            "linear_rail",
            "ur_shoulder_pan_joint",
            "ur_shoulder_lift_joint",
            "ur_elbow_joint",
            "ur_wrist_1_joint",
            "ur_wrist_2_joint",
            "ur_wrist_3_joint"
        ]
        
        # Define planning group name
        self.planning_group = "ur5e_gantry"  # Adjust based on your MoveIt config
        
        # Set up TF2 listener for transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Wait for MoveIt action server to be available
        self.get_logger().info('Waiting for MoveIt action server...')
        if not self.moveit_action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('MoveIt action server not available')
            raise RuntimeError("MoveIt action server not available")
        
        # Execute pick and place operation
        self.get_logger().info('Starting pick and place operation...')
        self.execute_pick_and_place()

    def quaternion_from_euler(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return [qw, qx, qy, qz]
    
    def create_pose(self, x, y, z, roll, pitch, yaw):
        """Create geometry_msgs/Pose from position and Euler angles"""
        q = self.quaternion_from_euler(roll, pitch, yaw)
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = q[0]
        pose.orientation.x = q[1]
        pose.orientation.y = q[2]
        pose.orientation.z = q[3]
        return pose
    
    def create_moveit_goal(self, target_pose=None, joint_positions=None):
        """Create MoveGroup action goal for either joint or pose targets"""
        goal = MoveGroup.Goal()
        
        # Set standard parameters
        goal.request.workspace_parameters.header.frame_id = "base_link"
        goal.request.workspace_parameters.min_corner.x = -1.0
        goal.request.workspace_parameters.min_corner.y = -1.0
        goal.request.workspace_parameters.min_corner.z = -1.0
        goal.request.workspace_parameters.max_corner.x = 1.0
        goal.request.workspace_parameters.max_corner.y = 1.0
        goal.request.workspace_parameters.max_corner.z = 1.0
        
        goal.request.start_state.is_diff = True
        goal.request.group_name = self.planning_group
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.5
        goal.request.max_acceleration_scaling_factor = 0.5
        
        # Set target based on input
        if target_pose is not None:
            goal.request.goal_constraints = [self.create_pose_goal(target_pose)]
        elif joint_positions is not None:
            goal.request.goal_constraints = [self.create_joint_goal(joint_positions)]
        
        return goal
    
    def create_pose_goal(self, target_pose):
        """Create pose constraint for MoveIt"""
        from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
        from shape_msgs.msg import SolidPrimitive
        
        constraints = Constraints()
        constraints.name = "pose_goal"
        
        # Position constraint
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = "ur_base_link"
        position_constraint.link_name = "ur_wrist_3_link"  # End effector link
        position_constraint.target_point_offset.x = 0.0
        position_constraint.target_point_offset.y = 0.0
        position_constraint.target_point_offset.z = 0.0
        
        # Define a small bounding box around the target position
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [0.05, 0.05, 0.05]  # Small tolerance box
        
        bounding_volume = BoundingVolume()
        bounding_volume.primitives.append(primitive)
        
        # Set the center of the bounding box to the target position
        primitive_pose = Pose()
        primitive_pose.position = target_pose.position
        primitive_pose.orientation.w = 1.0
        bounding_volume.primitive_poses.append(primitive_pose)
        
        position_constraint.constraint_region = bounding_volume
        position_constraint.weight = 1.0
        
        # Orientation constraint
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = "ur_base_link"
        orientation_constraint.orientation = target_pose.orientation
        orientation_constraint.link_name = "ur_wrist_3_link"  # End effector link
        orientation_constraint.absolute_x_axis_tolerance = 0.1
        orientation_constraint.absolute_y_axis_tolerance = 0.1
        orientation_constraint.absolute_z_axis_tolerance = 0.1
        orientation_constraint.weight = 1.0
        
        # Add constraints
        constraints.position_constraints.append(position_constraint)
        constraints.orientation_constraints.append(orientation_constraint)
        
        return constraints
    
    def create_joint_goal(self, joint_positions):
        """Create joint constraint for MoveIt"""
        from moveit_msgs.msg import Constraints, JointConstraint
        
        constraints = Constraints()
        constraints.name = "joint_goal"
        
        # Create a constraint for each joint
        for i, joint_name in enumerate(self.joint_names):
            constraint = JointConstraint()
            constraint.joint_name = joint_name
            constraint.position = joint_positions[i]
            constraint.tolerance_above = 0.01
            constraint.tolerance_below = 0.01
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)
        
        return constraints
    
    def plan_and_execute(self, goal):
        """Send goal to MoveIt action server and wait for result"""
        self.get_logger().info('Sending goal to MoveIt...')
        
        # Send the goal
        send_goal_future = self.moveit_action_client.send_goal_async(goal)
        
        # Wait for goal acceptance
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by MoveIt')
            return False
        
        self.get_logger().info('Goal accepted by MoveIt, waiting for result...')
        
        # Wait for the result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        result = result_future.result().result
        
        if result.error_code.val != 1:  # MoveItErrorCodes.SUCCESS = 1
            self.get_logger().error(f'Planning failed with error code: {result.error_code.val}')
            return False
        
        self.get_logger().info('Motion plan succeeded')
        
        # Publish trajectory for visualization
        self.publish_trajectory_for_rviz(result.planned_trajectory)
        
        # Execute the trajectory (in a real system, the controller would handle this)
        self.execute_trajectory(result.planned_trajectory)
        
        return True
    
    def publish_trajectory_for_rviz(self, trajectory):
        """Publish trajectory for visualization in RViz"""
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory.append(trajectory)
        # Create a proper RobotState
        robot_state = RobotState()
        robot_state.joint_state.header = trajectory.joint_trajectory.header
        robot_state.joint_state.name = trajectory.joint_trajectory.joint_names
        
        # If the trajectory has points, use the first point's positions
        if trajectory.joint_trajectory.points:
            robot_state.joint_state.position = trajectory.joint_trajectory.points[0].positions
        
        display_trajectory.trajectory_start = robot_state
        
        self.display_trajectory_publisher.publish(display_trajectory)
    
    def execute_trajectory(self, trajectory):
        """Simulate execution of trajectory by publishing joint states"""
        points = trajectory.joint_trajectory.points
        joint_names = trajectory.joint_trajectory.joint_names
        
        # Ensure all joints in self.joint_names are present
        all_joint_names = list(joint_names)
        for joint in self.joint_names:
            if joint not in all_joint_names:
                all_joint_names.append(joint)
        
        # Determine time intervals
        dt = 0.05  # 20Hz for visualization
        
        self.get_logger().info(f'Executing trajectory with {len(points)} points')
        
        for i, point in enumerate(points):
            # Skip some points to reduce frequency if trajectory is dense
            if len(points) > 100 and i % 5 != 0:
                continue
                
            # Create joint state message
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = all_joint_names
            
            # Create position array for all joints
            positions = [0.0] * len(all_joint_names)
            
            # Fill in positions from trajectory point
            for j, joint_name in enumerate(joint_names):
                idx = all_joint_names.index(joint_name)
                positions[idx] = point.positions[j]
                
            msg.position = positions
            
            # Create velocity and effort arrays if available
            velocities = [0.0] * len(all_joint_names)
            efforts = [0.0] * len(all_joint_names)
            
            # Fill in velocities from trajectory point if available
            if len(point.velocities) > 0:
                for j, joint_name in enumerate(joint_names):
                    idx = all_joint_names.index(joint_name)
                    velocities[idx] = point.velocities[j]
                    
            # Fill in efforts (accelerations) from trajectory point if available
            if len(point.accelerations) > 0:
                for j, joint_name in enumerate(joint_names):
                    idx = all_joint_names.index(joint_name)
                    efforts[idx] = point.accelerations[j]
                    
            msg.velocity = velocities
            msg.effort = efforts
                
            self.joint_state_publisher.publish(msg)
            time.sleep(dt)
    
    def move_to_pose(self, x, y, z, roll, pitch, yaw):
        """Plan and execute motion to the specified pose"""
        target_pose = self.create_pose(x, y, z, roll, pitch, yaw)
        goal = self.create_moveit_goal(target_pose=target_pose)
        return self.plan_and_execute(goal)
    
    def move_to_joint_positions(self, joint_positions):
        """Plan and execute motion to the specified joint positions"""
        goal = self.create_moveit_goal(joint_positions=joint_positions)
        return self.plan_and_execute(goal)
    
    def execute_pick_and_place(self):
        """Execute complete pick and place operation"""
        try:
            # Define pick and place poses
            pick_x, pick_y, pick_z = -0.5, -0.5, -0.1
            place_x, place_y, place_z = 0.5, 0.5, -0.1
            
            # Define home joint positions (in radians)
            home_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
            # Move to home position
            self.get_logger().info('Moving to home position...')
            success = self.move_to_joint_positions(home_positions)
            if not success:
                self.get_logger().error('Failed to move to home position')
                return
            
            # Move directly to pick position
            self.get_logger().info('Moving to pick position...')
            success = self.move_to_pose(pick_x, pick_y, pick_z, math.pi, 0.0, 0.0)
            if not success:
                self.get_logger().error('Failed to move to pick position')
                return
            
            # Simulate gripper closing - pause
            self.get_logger().info('At pick position - closing gripper')
            time.sleep(1.0)
            
            # Move directly to place position
            self.get_logger().info('Moving to place position...')
            success = self.move_to_pose(place_x, place_y, place_z, math.pi, 0.0, 0.0)
            if not success:
                self.get_logger().error('Failed to move to place position')
                return
            
            # Simulate gripper opening
            self.get_logger().info('At place position - opening gripper')
            time.sleep(1.0)
            
            # Move back to home position
            self.get_logger().info('Moving back to home position...')
            success = self.move_to_joint_positions(home_positions)
            if not success:
                self.get_logger().error('Failed to move back to home position')
                return
            
            self.get_logger().info('Pick and place operation completed successfully')
            
        except Exception as e:
            self.get_logger().error(f'Error during pick and place operation: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    
    try:
        pick_and_place_moveit = UR5ePickPlaceMoveitNode()
        rclpy.spin(pick_and_place_moveit)
    except Exception as e:
        print(f"Error during pick and place operation: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()