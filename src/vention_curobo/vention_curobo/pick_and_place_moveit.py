#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from moveit_msgs.msg import DisplayTrajectory, RobotState
from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import GetPositionFK
import numpy as np
import math
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from std_msgs.msg import Header, ColorRGBA
from moveit_msgs.msg import MoveItErrorCodes
from visualization_msgs.msg import Marker, MarkerArray
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
from moveit_msgs.msg import CollisionObject
from moveit_msgs.msg import PlanningScene, PlanningSceneComponents, PlanningSceneWorld
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

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
            
        # Set up publisher for marker visualization in RViz with QoS profile for reliability
        self.marker_publisher = self.create_publisher(
            MarkerArray, '/visualization_marker_array', 10)
        
        # Initialize marker array for publishing
        self.marker_array = MarkerArray()
            
        # Store trajectory points for visualization
        self.trajectory_points = []
        self.marker_id = 0

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
        
        # Define the end-effector link name
        self.ee_link = "ur_tool0"  # Adjust based on your robot model
        
        # Set up TF2 listener for transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Wait for MoveIt action server to be available
        self.get_logger().info('Waiting for MoveIt action server...')
        if not self.moveit_action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('MoveIt action server not available')
            raise RuntimeError("MoveIt action server not available")
        
        # Create FK service client
        self.client = self.create_client(GetPositionFK, 'compute_fk')
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for FK service...')
        
        # Create a marker clearer timer that periodically publishes markers to ensure they're visible
        # self.marker_timer = self.create_timer(1.0, self.publish_markers_periodically)
        
        # Data collection containers (similar to CuRobo script)
        self.ee_trajectory_data = []      # Store end-effector positions with metadata
        self.cycle_times = []             # Store cycle completion times
        self.planning_times_pick = []     # Store planning times for pick motion
        self.planning_times_place = []    # Store planning times for place motion
        self.execution_times_pick = []    # Store execution times for pick motion
        self.execution_times_place = []   # Store execution times for place motion
        
        # Number of cycles to run
        self.num_cycles = 20
        self.current_cycle = 0
        self.current_goal = "Pick"  # Start with pick operation
        
        # Define pick and place poses - matching CuRobo script
        self.pick_pose = self.create_pose(-0.5, -0.7, 0.5, 0.0, math.pi, 0.0)
        self.place_pose = self.create_pose(0.5, 0.7, 0.5, 0.0, math.pi, 0.0)
        self.add_obstacles()

        # Execute pick and place operation
        self.get_logger().info('Starting pick and place operation...')
        self.initialize_data_collection()
        self.execute_pick_and_place_cycles()
        
    def initialize_data_collection(self):
        """Initialize data collection files and structures"""
        # Create directory for saving data and plots if it doesn't exist
        self.data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Clear the CSV file at the start
        self.csv_path = os.path.join(self.data_dir, 'end_effector_positions.csv')
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x', 'y', 'z', 'cycle', 'goal'])  # Write header
        
    def publish_markers_periodically(self):
        """Periodically publish all markers to ensure they stay visible in RViz"""
        if self.trajectory_points:
            self.publish_trajectory_markers()
            self.get_logger().debug('Published markers periodically')
        
    def publish_trajectory_markers(self):
        """Publish all stored trajectory points as markers for RViz visualization"""
        if not self.trajectory_points:
            self.get_logger().warn('No trajectory points to visualize')
            return
            
        # Create a new marker array
        marker_array = MarkerArray()
        
        # First, add a marker to delete all previous markers
        delete_marker = Marker()
        delete_marker.header.frame_id = "ur_base_link"  # Make sure this matches your RViz fixed frame
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.ns = "trajectory"
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # Create a line strip to connect all points
        line_marker = Marker()
        line_marker.header.frame_id = "ur_base_link"  # Make sure this matches your RViz fixed frame
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = "trajectory"
        line_marker.id = 1
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.pose.orientation.w = 1.0
        line_marker.scale.x = 0.01  # Line width
        line_marker.color.r = 0.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0
        line_marker.lifetime.sec = 0  # 0 means forever
        
        # Add all points to the line strip
        for point in self.trajectory_points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2]
            line_marker.points.append(p)
            
        marker_array.markers.append(line_marker)
        
        # Create sphere markers for each point
        for i, point in enumerate(self.trajectory_points):
            marker = Marker()
            marker.header.frame_id = "ur_base_link"  # Make sure this matches your RViz fixed frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "trajectory"
            marker.id = i + 2  # Start IDs after line marker
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = point[2]
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.02  # Sphere size
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            
            # Color changes gradually from red to blue based on position in trajectory
            marker.color.r = 1.0 - float(i) / max(1, len(self.trajectory_points) - 1)
            marker.color.b = float(i) / max(1, len(self.trajectory_points) - 1)
            marker.color.g = 0.2
            marker.color.a = 1.0
            marker.lifetime.sec = 0  # 0 means forever
            
            marker_array.markers.append(marker)
            
        # Save the marker array for future republishing
        self.marker_array = marker_array
        
        # Publish the marker array
        self.marker_publisher.publish(marker_array)
        self.get_logger().info(f'Published {len(self.trajectory_points)} trajectory points as markers')
    
    def add_point_marker(self, x, y, z):
        """Add a new point marker to RViz"""
        # Store point for later visualization
        self.trajectory_points.append((x, y, z))
        
        # Log the added point
        self.get_logger().debug(f'Added trajectory point: [{x:.4f}, {y:.4f}, {z:.4f}]')
        
        # After adding several points, update the full trajectory visualization
        if len(self.trajectory_points) % 5 == 0 or len(self.trajectory_points) == 1:
            self.publish_trajectory_markers()

    def add_obstacles(self):
        """Add collision objects to the planning scene that match the MuJoCo obstacles"""
        self.get_logger().info('Adding obstacles to planning scene...')
        
        # Create a publisher for the planning scene
        self.planning_scene_publisher = self.create_publisher(
            PlanningScene, '/monitored_planning_scene', 10)
        
        # Allow time for the publisher to connect
        # self.create_rate(1).sleep()
        
        # Create planning scene message
        planning_scene = PlanningScene()
        planning_scene.is_diff = True  # Use as a diff update
        planning_scene.robot_state.is_diff = True
        
        # Create the collision objects
        col1 = CollisionObject()
        col1.header.frame_id = "ur_base_link"
        col1.id = "column1"
        col1.operation = CollisionObject.ADD
        
        # Box primitive for first column
        box1 = SolidPrimitive()
        box1.type = SolidPrimitive.BOX
        box1.dimensions = [0.14, 0.14, 0.70]  # 2x size in each dimension
        
        # Set pose for first column
        pose1 = Pose()
        pose1.position.x = 0.0
        pose1.position.y = 0.50
        pose1.position.z = 0.35
        pose1.orientation.w = 1.0
        
        col1.primitives.append(box1)
        col1.primitive_poses.append(pose1)
        
        # Create the second column obstacle
        col2 = CollisionObject()
        col2.header.frame_id = "ur_base_link"
        col2.id = "column2"
        col2.operation = CollisionObject.ADD
        
        # Box primitive for second column
        box2 = SolidPrimitive()
        box2.type = SolidPrimitive.BOX
        box2.dimensions = [0.14, 0.14, 0.70]
        
        # Set pose for second column
        pose2 = Pose()
        pose2.position.x = 0.0
        pose2.position.y = -0.50
        pose2.position.z = 0.35
        pose2.orientation.w = 1.0
        
        col2.primitives.append(box2)
        col2.primitive_poses.append(pose2)
        
        # Add collision objects to the world
        planning_scene.world.collision_objects.append(col1)
        planning_scene.world.collision_objects.append(col2)
        
        # Publish the planning scene
        self.get_logger().info('Publishing collision objects to planning scene...')
        self.planning_scene_publisher.publish(planning_scene)
        
        # Wait to ensure the planning scene receives the objects
        # self.create_rate(2).sleep()
        
        self.get_logger().info('Obstacles added to planning scene')
        
        # Also visualize the obstacles in RViz
        self.visualize_obstacles()

    def visualize_obstacles(self):
        """Create visualization markers for the obstacles"""
        marker_array = MarkerArray()
        
        # First column marker
        marker1 = Marker()
        marker1.header.frame_id = "ur_base_link"
        marker1.header.stamp = self.get_clock().now().to_msg()
        marker1.ns = "obstacles"
        marker1.id = 1
        marker1.type = Marker.CUBE
        marker1.action = Marker.ADD
        
        marker1.pose.position.x = 0.0
        marker1.pose.position.y = 0.50
        marker1.pose.position.z = 0.35
        marker1.pose.orientation.w = 1.0
        
        marker1.scale.x = 0.14
        marker1.scale.y = 0.14
        marker1.scale.z = 0.70
        
        marker1.color.r = 0.3
        marker1.color.g = 0.5
        marker1.color.b = 0.8
        marker1.color.a = 0.7
        
        # Second column marker
        marker2 = Marker()
        marker2.header.frame_id = "ur_base_link"
        marker2.header.stamp = self.get_clock().now().to_msg()
        marker2.ns = "obstacles"
        marker2.id = 2
        marker2.type = Marker.CUBE
        marker2.action = Marker.ADD
        
        marker2.pose.position.x = 0.0
        marker2.pose.position.y = -0.50
        marker2.pose.position.z = 0.35
        marker2.pose.orientation.w = 1.0
        
        marker2.scale.x = 0.14
        marker2.scale.y = 0.14
        marker2.scale.z = 0.70
        
        marker2.color.r = 0.3
        marker2.color.g = 0.5
        marker2.color.b = 0.8
        marker2.color.a = 0.7
        
        marker_array.markers.append(marker1)
        marker_array.markers.append(marker2)
        
        self.marker_publisher.publish(marker_array)
        self.get_logger().info('Published obstacle visualization markers')

    def store_ee_position(self, x, y, z, cycle, goal):
        """Store end-effector position with metadata in our data structure"""
        self.ee_trajectory_data.append({
            'x': x,
            'y': y,
            'z': z,
            'cycle': cycle,
            'goal': goal
        })
        
        # Also write to CSV
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", cycle, goal])
            
    def compute_fk_for_trajectory(self, trajectory):
        """Compute forward kinematics for all points in the trajectory"""
        self.get_logger().info('Computing FK for all trajectory points...')
        
        # Get joint names and trajectory points
        joint_names = trajectory.joint_trajectory.joint_names
        points = trajectory.joint_trajectory.points
        
        self.get_logger().info(f'Computing FK for all {len(points)} trajectory points')
        
        for idx in range(len(points)):
            # Create the FK request
            request = GetPositionFK.Request()
            request.header = Header()
            request.header.stamp = self.get_clock().now().to_msg()
            request.header.frame_id = 'ur_base_link'
            request.fk_link_names = [self.ee_link]
            
            # Set joint positions from trajectory point
            joint_state = JointState()
            joint_state.header = request.header
            joint_state.name = joint_names
            joint_state.position = list(points[idx].positions)
            
            # Create robot state for FK calculation
            robot_state = RobotState()
            robot_state.joint_state = joint_state
            request.robot_state = robot_state
            
            # Send synchronous request (blocking call)
            future = self.client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            
            try:
                response = future.result()
                
                if response.error_code.val == MoveItErrorCodes.SUCCESS:
                    pose = response.pose_stamped[0].pose
                    x = pose.position.x
                    y = pose.position.y
                    z = pose.position.z
                    
                    # Format with precision to make the output cleaner
                    self.get_logger().debug(f'Point {idx}: Position: [{x:.4f}, {y:.4f}, {z:.4f}])')
                    
                    # Add marker for this point
                    # self.add_point_marker(x, y, z)
                    
                    # Store position data with metadata
                    self.store_ee_position(x, y, z, self.current_cycle, self.current_goal)
                else:
                    self.get_logger().error(f'FK calculation failed for point {idx} with error code: {response.error_code.val}')
            except Exception as e:
                self.get_logger().error(f'FK service call failed for point {idx}: {str(e)}')
                
        # After all points are processed, publish the complete trajectory
        # self.publish_trajectory_markers()

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
        """Create MoveGroup action goal for either joint or pose targets with lenient constraints"""
        goal = MoveGroup.Goal()
        
        # Set standard parameters with expanded workspace
        goal.request.workspace_parameters.header.frame_id = "ur_base_link"
        goal.request.workspace_parameters.min_corner.x = -3.0  # Expanded workspace
        goal.request.workspace_parameters.min_corner.y = -3.0
        goal.request.workspace_parameters.min_corner.z = -3.0
        goal.request.workspace_parameters.max_corner.x = 3.0
        goal.request.workspace_parameters.max_corner.y = 3.0
        goal.request.workspace_parameters.max_corner.z = 3.0
        
        goal.request.start_state.is_diff = True
        goal.request.group_name = self.planning_group
        goal.request.num_planning_attempts = 50  # More planning attempts
        goal.request.allowed_planning_time = 30.0  # More planning time
        goal.request.max_velocity_scaling_factor = 0.5  # Higher velocity scaling
        goal.request.max_acceleration_scaling_factor = 0.5  # Higher acceleration scaling

        # Specify the planner - RRTConnect is typically the most reliable for finding any solution
        goal.request.pipeline_id = "ompl"
        goal.request.planner_id = "RRTConnect"  # Most reliable general-purpose planner
        
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
        position_constraint.link_name = "ur_tool0"  # End effector link
        position_constraint.target_point_offset.x = 0.01
        position_constraint.target_point_offset.y = 0.01
        position_constraint.target_point_offset.z = 0.01
        
        # Define a small bounding box around the target position
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [0.01, 0.01, 0.01]  # Small tolerance box
        
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
        orientation_constraint.link_name = "ur_tool0"  # End effector link
        orientation_constraint.absolute_x_axis_tolerance = 0.01
        orientation_constraint.absolute_y_axis_tolerance = 0.01
        orientation_constraint.absolute_z_axis_tolerance = 0.01
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
            constraint.tolerance_above = 0.05
            constraint.tolerance_below = 0.05
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)
        
        return constraints
    
    def plan_and_execute(self, goal, is_pick=True):
        """Send goal to MoveIt action server and wait for result"""
        goal_type = "Pick" if is_pick else "Place"
        self.get_logger().info(f'Sending {goal_type} goal to MoveIt...')
        
        # Start timing the planning phase
        planning_start_time = self.get_clock().now()
        
        # Send the goal
        send_goal_future = self.moveit_action_client.send_goal_async(goal)
        
        # Wait for goal acceptance
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error(f'{goal_type} goal rejected by MoveIt')
            return False
        
        self.get_logger().info(f'{goal_type} goal accepted by MoveIt, waiting for result...')
        
        # Wait for the result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        # Stop timing when we get the result
        planning_end_time = self.get_clock().now()
        planning_time = (planning_end_time - planning_start_time).nanoseconds / 1e9  # Convert to seconds
        
        # Store planning time based on the type of motion
        if is_pick:
            self.planning_times_pick.append(planning_time)
        else:
            self.planning_times_place.append(planning_time)
    
        self.get_logger().info(f'{goal_type} planning time: {planning_time:.4f} seconds')
        
        result = result_future.result().result
        
        # Compute forward kinematics for the trajectory for visualization and data collection
        self.compute_fk_for_trajectory(result.planned_trajectory)
        
        if result.error_code.val != 1:  # MoveItErrorCodes.SUCCESS = 1
            self.get_logger().error(f'{goal_type} planning failed with error code: {result.error_code.val}')

        self.get_logger().info(f'{goal_type} motion plan succeeded')
        
        # Publish trajectory for visualization
        self.publish_trajectory_for_rviz(result.planned_trajectory)
        
        # Execute the trajectory and time it
        execution_start_time = self.get_clock().now()
        self.execute_trajectory(result.planned_trajectory)
        execution_end_time = self.get_clock().now()
        execution_time = (execution_end_time - execution_start_time).nanoseconds / 1e9  # Convert to seconds
        
        # Store execution time based on the type of motion
        if is_pick:
            self.execution_times_pick.append(execution_time)
        else:
            self.execution_times_place.append(execution_time)
            
        self.get_logger().info(f'{goal_type} execution time: {execution_time:.4f} seconds')
        
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
        """Execute trajectory by properly publishing joint states"""
        points = trajectory.joint_trajectory.points
        joint_names = trajectory.joint_trajectory.joint_names
        
        # Ensure all joints in self.joint_names are present
        all_joint_names = list(joint_names)
        
        self.get_logger().info(f'Executing trajectory with {len(points)} points')
        self.get_logger().debug(f'Joint names in trajectory: {joint_names}')
        
        # Create a Rate object for precise timing
        rate = self.create_rate(20)  # 20Hz for visualization
        
        for i, point in enumerate(points):
            # Skip some points to reduce frequency if trajectory is dense
            if len(points) > 100 and i % 5 != 0 and i < len(points) - 1:
                continue
                
            # Create joint state message
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "ur_base_link"  # Make sure this matches your robot's base frame
            msg.name = joint_names  # Use exactly the joint names from the trajectory
            msg.position = list(point.positions)
            
            # Add velocities if available
            if point.velocities:
                msg.velocity = list(point.velocities)
                
            # Add accelerations as efforts if available
            if point.accelerations:
                msg.effort = list(point.accelerations)
                
            # Publish the joint state
            self.joint_state_publisher.publish(msg)
            self.get_logger().debug(f'Published joint state for point {i}')
            
            # Use ROS 2 rate mechanism instead of time.sleep
            # rate.sleep()
        
        self.get_logger().info('Finished executing trajectory')
        
        # Publish the final complete visualization
        self.publish_trajectory_markers()
    
    def move_to_pose(self, pose, is_pick=True):
        """Plan and execute motion to the specified pose"""
        goal = self.create_moveit_goal(target_pose=pose)
        return self.plan_and_execute(goal, is_pick)
    
    def move_to_joint_positions(self, joint_positions, is_pick=True):
        """Plan and execute motion to the specified joint positions"""
        goal = self.create_moveit_goal(joint_positions=joint_positions)
        return self.plan_and_execute(goal, is_pick)
    
    def execute_pick_and_place_cycles(self):
        """Execute complete pick and place operation for multiple cycles"""
        try:
            self.get_logger().info(f'Starting {self.num_cycles} pick and place cycles...')
            
            # Add some test markers to verify visibility
            self.get_logger().info('Publishing test markers to verify visibility...')
            self.add_point_marker(0.0, 0.0, 0.5)  # Origin marker
            self.add_point_marker(0.5, 0.0, 0.5)  # X-axis marker
            self.add_point_marker(0.0, 0.5, 0.5)  # Y-axis marker
            # self.publish_trajectory_markers()
            
            while self.current_cycle < self.num_cycles:
                # Clear trajectory points for new cycle visualization
                self.trajectory_points = []
                
                self.get_logger().info(f'Executing cycle {self.current_cycle + 1} of {self.num_cycles}')
                
                # Move to pick position
                self.current_goal = "Pick"
                self.get_logger().info(f'Cycle {self.current_cycle + 1}: Moving to pick position...')
                success = self.move_to_pose(self.pick_pose, is_pick=True)
                if not success:
                    self.get_logger().error(f'Cycle {self.current_cycle + 1}: Failed to move to pick position')
                    break
                
                # Move to place position
                self.current_goal = "Place"
                self.get_logger().info(f'Cycle {self.current_cycle + 1}: Moving to place position...')
                success = self.move_to_pose(self.place_pose, is_pick=False)
                if not success:
                    self.get_logger().error(f'Cycle {self.current_cycle + 1}: Failed to move to place position')
                    break
                
                # Calculate total cycle time
                cycle_time = 0.0
                if len(self.planning_times_pick) > 0 and len(self.planning_times_place) > 0 and \
                   len(self.execution_times_pick) > 0 and len(self.execution_times_place) > 0:
                    cycle_time = (
                        self.planning_times_pick[-1] + self.planning_times_place[-1] + 
                        self.execution_times_pick[-1] + self.execution_times_place[-1]
                    )
                    self.cycle_times.append(cycle_time)
                    
                self.get_logger().info(f'Cycle {self.current_cycle + 1} completed in {cycle_time:.4f} seconds')
                
                # Increment cycle counter
                self.current_cycle += 1
                
            self.get_logger().info('All pick and place cycles completed')
            
            # Generate and save plots
            if self.current_cycle > 0:
                self.generate_plots()
            
        except Exception as e:
            self.get_logger().error(f'Error during pick and place operation: {str(e)}')
    
    def generate_plots(self):
        """Generate and save performance plots"""
        self.get_logger().info('Generating performance plots...')
        
        if not self.ee_trajectory_data:
            self.get_logger().error('No trajectory data collected. Cannot generate plots.')
            return
            
        try:
            # 3D trajectory plot
            self.plot_3d_trajectory()
            
            # Cycle times plot
            self.plot_cycle_times()
            
            # Planning times plot
            self.plot_planning_times()
            
            # 2D projections plot
            self.plot_2d_projections()
            
            # Print statistics
            self.print_performance_statistics()
            
            self.get_logger().info('All plots generated and saved successfully')
            
        except Exception as e:
            self.get_logger().error(f'Error generating plots: {str(e)}')
    
    def plot_3d_trajectory(self):
        """Plot 3D trajectory of end-effector movements"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for cycle in range(self.current_cycle):
            cycle_data = [d for d in self.ee_trajectory_data if d['cycle'] == cycle]
            
            # Plot pick trajectory
            pick_data = [d for d in cycle_data if d['goal'] == "Pick"]
            if pick_data:
                pick_x = [d['x'] for d in pick_data]
                pick_y = [d['y'] for d in pick_data]
                pick_z = [d['z'] for d in pick_data]
                ax.plot(pick_x, pick_y, pick_z, 'r-', linewidth=2, alpha=0.7, 
                        label=f"Cycle {cycle+1} Pick" if cycle == 0 else "")
            
            # Plot place trajectory
            place_data = [d for d in cycle_data if d['goal'] == "Place"]
            if place_data:
                place_x = [d['x'] for d in place_data]
                place_y = [d['y'] for d in place_data]
                place_z = [d['z'] for d in place_data]
                ax.plot(place_x, place_y, place_z, 'b-', linewidth=2, alpha=0.7, 
                        label=f"Cycle {cycle+1} Place" if cycle == 0 else "")
        
        # Plot the goal positions
        pick_pos = self.pick_pose.position
        place_pos = self.place_pose.position
        ax.scatter([pick_pos.x], [pick_pos.y], [pick_pos.z], 
                   color='red', s=100, marker='o', label='Pick Position')
        ax.scatter([place_pos.x], [place_pos.y], [place_pos.z], 
                   color='blue', s=100, marker='o', label='Place Position')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('End-Effector Trajectory During Pick and Place Operations')
        ax.legend()
        
        plt.savefig(os.path.join(self.data_dir, 'ee_trajectory_3d.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.get_logger().info('3D trajectory plot saved')
        
    def plot_cycle_times(self):
        """Plot cycle times for all completed cycles"""
        if not self.cycle_times:
            self.get_logger().warn('No cycle time data to plot')
            return
            
        plt.figure(figsize=(10, 6))
        cycles = range(1, len(self.cycle_times) + 1)
        plt.bar(cycles, self.cycle_times, color='skyblue')
        
        # Add average line
        avg_cycle_time = np.mean(self.cycle_times)
        plt.axhline(y=avg_cycle_time, color='r', linestyle='-', label=f'Average: {avg_cycle_time:.2f}s')
        
        plt.xlabel('Cycle Number')
        plt.ylabel('Time (seconds)')
        plt.title('Pick and Place Cycle Times')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.data_dir, 'cycle_times.png'), dpi=300)
        plt.close()
        
        self.get_logger().info('Cycle times plot saved')
        
    def plot_planning_times(self):
        """Plot planning times for pick and place operations"""
        if not self.planning_times_pick or not self.planning_times_place:
            self.get_logger().warn('No planning time data to plot')
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pick planning times
        pick_cycles = range(1, len(self.planning_times_pick) + 1)
        ax1.bar(pick_cycles, self.planning_times_pick, color='coral')
        
        # Add average line for pick planning times
        avg_pick_plan = np.mean(self.planning_times_pick)
        ax1.axhline(y=avg_pick_plan, color='r', linestyle='-', label=f'Avg: {avg_pick_plan:.2f}s')
        
        ax1.set_xlabel('Cycle Number')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Pick Planning Times')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Place planning times
        place_cycles = range(1, len(self.planning_times_place) + 1)
        ax2.bar(place_cycles, self.planning_times_place, color='skyblue')
        
        # Add average line for place planning times
        avg_place_plan = np.mean(self.planning_times_place)
        ax2.axhline(y=avg_place_plan, color='r', linestyle='-', label=f'Avg: {avg_place_plan:.2f}s')
        
        ax2.set_xlabel('Cycle Number')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Place Planning Times')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'planning_times.png'), dpi=300)
        plt.close()
        
        self.get_logger().info('Planning times plot saved')
        
    def plot_2d_projections(self):
        """Create 2D projections of the end-effector trajectory"""
        if not self.ee_trajectory_data:
            self.get_logger().warn('No trajectory data to plot 2D projections')
            return
            
        fig, axs = plt.subplots(2, 2, figsize=(16, 14))
        
        # XY Projection
        axs[0, 0].set_title('XY Projection')
        axs[0, 0].set_xlabel('X (m)')
        axs[0, 0].set_ylabel('Y (m)')
        axs[0, 0].grid(True)
        
        # XZ Projection
        axs[0, 1].set_title('XZ Projection')
        axs[0, 1].set_xlabel('X (m)')
        axs[0, 1].set_ylabel('Z (m)')
        axs[0, 1].grid(True)
        
        # YZ Projection
        axs[1, 0].set_title('YZ Projection')
        axs[1, 0].set_xlabel('Y (m)')
        axs[1, 0].set_ylabel('Z (m)')
        axs[1, 0].grid(True)
        
        # 3D view in the last subplot
        ax3d = fig.add_subplot(2, 2, 4, projection='3d')
        ax3d.set_title('3D Trajectory')
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')
        
        # Limit to 5 cycles for clarity
        max_cycles_to_plot = min(self.current_cycle, 5)
        
        for cycle in range(max_cycles_to_plot):
            cycle_data = [d for d in self.ee_trajectory_data if d['cycle'] == cycle]
            
            # Process pick and place separately for coloring
            for goal_type, color, marker in [("Pick", 'red', '-'), ("Place", 'blue', '-')]:
                goal_data = [d for d in cycle_data if d['goal'] == goal_type]
                if goal_data:
                    x = [d['x'] for d in goal_data]
                    y = [d['y'] for d in goal_data]
                    z = [d['z'] for d in goal_data]
                    
                    # Plot projections
                    axs[0, 0].plot(x, y, color=color, linestyle=marker, alpha=0.7)
                    axs[0, 1].plot(x, z, color=color, linestyle=marker, alpha=0.7)
                    axs[1, 0].plot(y, z, color=color, linestyle=marker, alpha=0.7)
                    
                    # Plot 3D
                    ax3d.plot(x, y, z, color=color, linestyle=marker, alpha=0.7)
        
        # Mark goal positions on all plots
        pick_pos = self.pick_pose.position
        place_pos = self.place_pose.position
        
        for ax, coord1, coord2, p1, p2, q1, q2 in [
            (axs[0, 0], 'x', 'y', pick_pos.x, pick_pos.y, place_pos.x, place_pos.y),
            (axs[0, 1], 'x', 'z', pick_pos.x, pick_pos.z, place_pos.x, place_pos.z),
            (axs[1, 0], 'y', 'z', pick_pos.y, pick_pos.z, place_pos.y, place_pos.z)
        ]:
            ax.scatter(p1, p2, color='red', s=100, marker='o', label='Pick Position')
            ax.scatter(q1, q2, color='blue', s=100, marker='o', label='Place Position')
            ax.legend()
        
        # Add 3D markers
        ax3d.scatter(pick_pos.x, pick_pos.y, pick_pos.z, color='red', s=100, marker='o', label='Pick Position')
        ax3d.scatter(place_pos.x, place_pos.y, place_pos.z, color='blue', s=100, marker='o', label='Place Position')
        ax3d.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'trajectory_projections.png'), dpi=300)
        plt.close()
        
        self.get_logger().info('2D trajectory projections plot saved')
    
    def print_performance_statistics(self):
        """Calculate and print performance statistics"""
        self.get_logger().info("\nPerformance Statistics:")
        
        if self.cycle_times:
            avg_cycle_time = np.mean(self.cycle_times)
            std_cycle_time = np.std(self.cycle_times)
            self.get_logger().info(f"Average Cycle Time: {avg_cycle_time:.4f} ± {std_cycle_time:.4f} seconds")
        
        if self.planning_times_pick:
            avg_pick_plan = np.mean(self.planning_times_pick)
            std_pick_plan = np.std(self.planning_times_pick)
            self.get_logger().info(f"Average Pick Planning Time: {avg_pick_plan:.4f} ± {std_pick_plan:.4f} seconds")
        
        if self.planning_times_place:
            avg_place_plan = np.mean(self.planning_times_place)
            std_place_plan = np.std(self.planning_times_place)
            self.get_logger().info(f"Average Place Planning Time: {avg_place_plan:.4f} ± {std_place_plan:.4f} seconds")
        
        # Calculate average total planning time
        all_planning_times = self.planning_times_pick + self.planning_times_place
        if all_planning_times:
            avg_planning_time = np.mean(all_planning_times)
            std_planning_time = np.std(all_planning_times)
            self.get_logger().info(f"Average Planning Time: {avg_planning_time:.4f} ± {std_planning_time:.4f} seconds")
        
        # Calculate execution statistics
        if self.execution_times_pick and self.execution_times_place:
            avg_pick_exec = np.mean(self.execution_times_pick)
            avg_place_exec = np.mean(self.execution_times_place)
            avg_exec_time = np.mean(self.execution_times_pick + self.execution_times_place)
            self.get_logger().info(f"Average Pick Execution Time: {avg_pick_exec:.4f} seconds")
            self.get_logger().info(f"Average Place Execution Time: {avg_place_exec:.4f} seconds")
            self.get_logger().info(f"Average Execution Time: {avg_exec_time:.4f} seconds")
        
        # Save statistics to file
        stats_file = os.path.join(self.data_dir, 'performance_statistics.txt')
        with open(stats_file, 'w') as f:
            f.write("UR5e Pick and Place Performance Statistics\n")
            f.write("=========================================\n\n")
            
            if self.cycle_times:
                f.write(f"Number of completed cycles: {len(self.cycle_times)}\n")
                f.write(f"Average Cycle Time: {avg_cycle_time:.4f} ± {std_cycle_time:.4f} seconds\n")
            
            if all_planning_times:
                f.write(f"Average Planning Time: {avg_planning_time:.4f} ± {std_planning_time:.4f} seconds\n")
                f.write(f"Average Pick Planning Time: {avg_pick_plan:.4f} ± {std_pick_plan:.4f} seconds\n")
                f.write(f"Average Place Planning Time: {avg_place_plan:.4f} ± {std_place_plan:.4f} seconds\n")
            
            if self.execution_times_pick and self.execution_times_place:
                f.write(f"Average Execution Time: {avg_exec_time:.4f} seconds\n")
                f.write(f"Average Pick Execution Time: {avg_pick_exec:.4f} seconds\n")
                f.write(f"Average Place Execution Time: {avg_place_exec:.4f} seconds\n")
        
        self.get_logger().info(f"Performance statistics saved to {stats_file}")

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