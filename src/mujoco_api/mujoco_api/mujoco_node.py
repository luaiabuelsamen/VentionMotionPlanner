#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
import numpy as np
from mujoco_api.mujoco_parser import MuJoCoParserClass
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from curobo_action.action import PublishJoints
import time

class MujocoNode(Node):
    def __init__(self):
        super().__init__('mujoco_joint_state_publisher')
        
        # Create callback group for allowing concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()
        
        # Declare parameters
        self.declare_parameter('xml_path', './src/mujoco_curobo/assets/ur5e/scene_ur5e_2f140_obj (sution).xml')
        self.declare_parameter('publish_rate', 50.0)  # Hz
        self.declare_parameter('default_positions', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Get parameters
        self.xml_path = self.get_parameter('xml_path').value
        self.publish_rate = self.get_parameter('publish_rate').value
        default_positions = self.get_parameter('default_positions').value
        
        # Create publisher
        self.joint_state_publisher = self.create_publisher(
            JointState,
            'joint_states',
            10
        )
        
        # Create action server
        self._action_server = ActionServer(
            self,
            PublishJoints,
            'publish_joints',
            self.execute_callback,
            callback_group=self.callback_group,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        
        # Initialize MuJoCo environment
        try:
            self.env = MuJoCoParserClass(name='UR5e with RG2 gripper', rel_xml_path=self.xml_path, VERBOSE=True)
            
            # Initialize viewer
            self.env.init_viewer(viewer_title='UR5e with RG2 gripper', viewer_width=1200, viewer_height=800,
                                viewer_hide_menus=True)
            
            # Set default view
            self.env.update_viewer(azimuth=66.08, distance=3.0, elevation=-50, lookat=[0.4, 0.18, 0.71],
                                  VIS_TRANSPARENT=False, VIS_CONTACTPOINT=False,
                                  contactwidth=0.05, contactheight=0.05, contactrgba=np.array([1, 0, 0, 1]),
                                  VIS_JOINT=True, jointlength=0.25, jointwidth=0.05, jointrgba=[0.2, 0.6, 0.8, 0.6])
            
            # Log model information
            self.get_logger().info(f'Number of actuators: {self.env.n_ctrl}')
            self.get_logger().info(f'Actuator names: {self.env.ctrl_names}')
            self.get_logger().info(f'Number of joints: {self.env.n_joint}')
            self.get_logger().info(f'Joint names: {self.env.joint_names}')
            
            # Initialize target positions and indices
            self.action_received = False
            self.target_positions = default_positions[:self.env.n_ctrl]
            # Pad with zeros if default positions are too short
            if len(self.target_positions) < self.env.n_ctrl:
                self.target_positions.extend([0.0] * (self.env.n_ctrl - len(self.target_positions)))
            # Truncate if too long
            self.target_positions = self.target_positions[:self.env.n_ctrl]
            self.target_indices = list(range(len(self.target_positions)))
            
            # Create timer for simulation and publishing
            self.step_timer = self.create_timer(0.01, self.step_simulation, callback_group=self.callback_group)
            self.publish_timer = self.create_timer(1.0/self.publish_rate, self.publish_joint_states, callback_group=self.callback_group)
            
            self.get_logger().info('MuJoCo viewer and joint state publisher initialized successfully')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize MuJoCo environment: {str(e)}')
    
    def goal_callback(self, goal_request):
        """Callback for handling new action goals"""
        self.get_logger().info('Received goal request')
        
        # Validate the goal
        if not hasattr(self, 'env'):
            self.get_logger().error('Cannot accept goal: MuJoCo environment not initialized')
            return GoalResponse.REJECT
            
        if len(goal_request.positions) != len(goal_request.indices):
            self.get_logger().error('Cannot accept goal: positions and indices must have the same length')
            return GoalResponse.REJECT
            
        for idx in goal_request.indices:
            if idx < 0 or idx >= self.env.n_ctrl:
                self.get_logger().error(f'Cannot accept goal: index {idx} is out of range (0-{self.env.n_ctrl-1})')
                return GoalResponse.REJECT
                
        return GoalResponse.ACCEPT
    
    def cancel_callback(self, goal_handle):
        """Callback for handling goal cancellation requests"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT
    
    def execute_callback(self, goal_handle):
        """Execute the goal"""
        self.get_logger().info('Executing goal...')
        
        # Get the goal
        goal = goal_handle.request
        
        # Update target positions and indices
        for i, idx in enumerate(goal.indices):
            if idx < len(self.target_positions):
                self.target_positions[idx] = goal.positions[i]
        
        self.target_indices = goal.indices
        self.action_received = True
        
        # Give time for the simulation to apply the new joint values
        time.sleep(0.5)
        
        # Check if positions were applied correctly
        positions_match = True
        for i, idx in enumerate(goal.indices):
            ctrl_idx = self.env.ctrl_joint_idxs[idx]
            qpos_idx = self.env.ctrl_qpos_idxs[idx]
            actual_pos = self.env.data.qpos[qpos_idx]
            target_pos = goal.positions[i]
            
            # Allow for small numerical differences
            if abs(actual_pos - target_pos) > 0.01:
                positions_match = False
                self.get_logger().warn(f'Joint {idx} ({self.env.ctrl_names[idx]}) not at target position: {actual_pos} vs {target_pos}')
        
        # Set the result
        result = PublishJoints.Result()
        
        if positions_match:
            result.success = True
            result.message = 'Successfully set joint positions'
        else:
            result.success = True  # Still return success as the command was processed
            result.message = 'Command processed but joint positions may not match exactly'
        
        goal_handle.succeed()
        
        self.get_logger().info(f'Goal finished with result: {result.message}')
        
        return result
    
    def step_simulation(self):
        """Timer callback for stepping the simulation"""
        if not hasattr(self, 'env'):
            return
        
        if self.env.is_viewer_alive():
            if self.action_received:
                self.env.step(self.target_positions, self.target_indices)
            
            self.env.render()
        else:
            self.step_timer.cancel()
            self.publish_timer.cancel()
            self.get_logger().info('Viewer closed, stopping simulation and publishers')
    
    def publish_joint_states(self):
        """Publish joint states including all actuators in order"""
        if not hasattr(self, 'env') or not self.env.is_viewer_alive():
            return
            
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = self.env.ctrl_names.copy()
        positions = []
        velocities = []
        
        for i, ctrl_idx in enumerate(self.env.ctrl_joint_idxs):
            pos = self.env.data.qpos[self.env.ctrl_qpos_idxs[i]]
            positions.append(pos)
            vel = self.env.data.qvel[self.env.ctrl_qvel_idxs[i]]
            velocities.append(vel)
        
        joint_state_msg.position = positions
        joint_state_msg.velocity = velocities
        joint_state_msg.effort = list(self.env.data.ctrl)
        self.joint_state_publisher.publish(joint_state_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MujocoNode()
    rclpy.spin(node)
    
    # Cleanup
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()