#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
import numpy as np
from mujoco_parser import MuJoCoParserClass

class SimpleMuJoCoViewerNode(Node):
    def __init__(self):
        super().__init__('mujoco_viewer_node')
        
        # Declare only essential parameter
        self.declare_parameter('xml_path', './assets/ur5e/scene_ur5e_2f140_obj (sution).xml')
        
        # Get parameter
        self.xml_path = self.get_parameter('xml_path').value
        self.get_logger().info(f'Loading MuJoCo model from: {self.xml_path}')
        
        # Check if the XML file exists
        if not os.path.exists(self.xml_path):
            self.get_logger().error(f'XML file not found at {self.xml_path}')
            return
        
        # Initialize MuJoCo environment
        try:
            self.env = MuJoCoParserClass(name='UR5e with RG2 gripper', rel_xml_path=self.xml_path, VERBOSE=True)
            
            # Initialize viewer with default settings
            self.env.init_viewer(viewer_title='UR5e with RG2 gripper', viewer_width=1200, viewer_height=800,
                                viewer_hide_menus=True)
            
            # Set a nice default view
            self.env.update_viewer(azimuth=66.08, distance=3.0, elevation=-50, lookat=[0.4, 0.18, 0.71],
                                  VIS_TRANSPARENT=False, VIS_CONTACTPOINT=False,
                                  contactwidth=0.05, contactheight=0.05, contactrgba=np.array([1, 0, 0, 1]),
                                  VIS_JOINT=True, jointlength=0.25, jointwidth=0.05, jointrgba=[0.2, 0.6, 0.8, 0.6])
            
            # Create a timer for simulation steps
            self.step_timer = self.create_timer(0.01, self.step_simulation)
            self.get_logger().info('MuJoCo viewer initialized successfully')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize MuJoCo environment: {str(e)}')
    
    def step_simulation(self):
        """Timer callback for stepping the simulation"""
        if not hasattr(self, 'env'):
            return
        if self.env.is_viewer_alive():
            self.env.render()
        else:
            self.step_timer.cancel()
            self.get_logger().info('Simulation completed or viewer closed')

def main(args=None):
    rclpy.init(args=args)
    node = SimpleMuJoCoViewerNode()
    rclpy.spin(node)
    
    # Cleanup
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()