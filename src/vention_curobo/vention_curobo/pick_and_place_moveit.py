#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point
from moveit_msgs.msg import DisplayTrajectory, RobotState
from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import GetPositionFK

import math
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from std_msgs.msg import Header
from moveit_msgs.msg import MoveItErrorCodes
from visualization_msgs.msg import Marker, MarkerArray
import csv
import os

from geometry_msgs.msg import Pose
from vention_curobo.generate_plots import Plots

class UR5ePickPlaceMoveitNode(Node):
    def __init__(self):
        super().__init__('ur5e_pick_place_moveit')
        
        # Set up publisher to publish the joint_states for RViz2
        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 10)
        
        # Set up MoveIt2 action client
        self.moveit_action_client = ActionClient(self, MoveGroup, 'move_action')
        
        # Set up publisher for MoveIt trajectory visualization
        self.display_trajectory_publisher = self.create_publisher(
            DisplayTrajectory, '/move_group/display_planned_path', 10)
            
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

        # Execute pick and place operation
        self.plotter = Plots(self)
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
                    self.add_point_marker(x, y, z)
                    
                    # Store position data with metadata
                    self.store_ee_position(x, y, z, self.current_cycle, self.current_goal)
                else:
                    self.get_logger().error(f'FK calculation failed for point {idx} with error code: {response.error_code.val}')
            except Exception as e:
                self.get_logger().error(f'FK service call failed for point {idx}: {str(e)}')

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
        ret = False
        # Send the goal
        send_goal_future = self.moveit_action_client.send_goal_async(goal)
        
        # Wait for goal acceptance
        planning_start_time = self.get_clock().now()
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        planning_end_time = self.get_clock().now()
        planning_time = (planning_end_time - planning_start_time).nanoseconds / 1e9
        if not goal_handle.accepted:
            self.get_logger().error(f'{goal_type} goal rejected by MoveIt')
            assert False
        
        self.get_logger().info(f'{goal_type} goal accepted by MoveIt, waiting for result...')
        
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        if result.error_code.val != 1:  # MoveItErrorCodes.SUCCESS = 1
            self.get_logger().error(f'{goal_type} planning failed with error code: {result.error_code.val}')
            planning_time = 0
            execution_time = 0
        else:
            # planning_time = result.planning_time
            self.get_logger().info(f'{goal_type} motion plan succeeded')
            self.get_logger().info(f'{goal_type} planning time: {planning_time:.4f} seconds')
            self.compute_fk_for_trajectory(result.planned_trajectory)
            # Publish trajectory for visualization
            # self.publish_trajectory_for_rviz(result.planned_trajectory)
            self.execute_trajectory(result.planned_trajectory)
            last_point = result.planned_trajectory.joint_trajectory.points[-1]
            execution_time= last_point.time_from_start.sec + last_point.time_from_start.nanosec / 1e9
            self.get_logger().info(f'{goal_type} execution time: {execution_time:.4f} seconds')
            ret = True
        if is_pick:
            self.planning_times_pick.append(planning_time)
            self.execution_times_pick.append(execution_time)
        else:
            self.planning_times_place.append(planning_time)
            self.execution_times_place.append(execution_time)
    
        return ret
    
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
        
        
        self.get_logger().info(f'Executing trajectory with {len(points)} points')
        self.get_logger().debug(f'Joint names in trajectory: {joint_names}')

        for i, point in enumerate(points):
                
            # Create joint state message
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "ur_base_link"
            msg.name = joint_names 
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
            
        self.get_logger().info('Finished executing trajectory')
        
        # Publish the final complete visualization
        self.publish_trajectory_markers()
    
    def move_to_pose(self, pose, is_pick=True):
        """Plan and execute motion to the specified pose"""
        goal = self.create_moveit_goal(target_pose=pose)
        return self.plan_and_execute(goal, is_pick)
    
    def execute_pick_and_place_cycles(self):
        """Execute complete pick and place operation for multiple cycles"""
        self.get_logger().info(f'Starting {self.num_cycles} pick and place cycles...')
        
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
            
            # Move to place position
            self.current_goal = "Place"
            self.get_logger().info(f'Cycle {self.current_cycle + 1}: Moving to place position...')
            success = self.move_to_pose(self.place_pose, is_pick=False)
            if not success:
                self.get_logger().error(f'Cycle {self.current_cycle + 1}: Failed to move to place position')
            
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
            
            self.current_cycle += 1
            
        self.get_logger().info('All pick and place cycles completed')

        if self.current_cycle > 0:
            self.plotter.generate_plots()

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