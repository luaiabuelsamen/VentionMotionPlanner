#!/usr/bin/env python3
import threading

from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from rclpy.node import Node

class RvizPublisher(Node):
    def __init__(self, joint_names, obstacles):
        super().__init__('rviz_publisher')

        # Publishers
        self.joint_publisher = self.create_publisher(JointState, "/joint_states", 10)
        self.marker_publisher = self.create_publisher(MarkerArray, "/obstacle_markers", 10)

        # Robot joint names
        self.joint_names = joint_names
        self.obstacles = obstacles

        self.get_logger().info("RVIZ Publisher Initialized")

    def publish_joint_states(self, joint_angles):
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = self.joint_names

        joint_state.position = joint_angles
        self.current_joint_state = joint_state
        self.joint_publisher.publish(joint_state)

    def _persist_joint_state(self):
        """Continuously publish the last known state."""
        rate = self.create_rate(10)
        while True:
            if self.current_joint_state:
                self.joint_publisher.publish(self.current_joint_state)
            rate.sleep()
    
    def publish_obstacles(self):
        marker_array = MarkerArray()
        for i, (obstacle_name, obstacle) in enumerate(self.obstacles["cuboid"].items()):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.scale.x = obstacle["dims"][0]
            marker.scale.y = obstacle["dims"][1]
            marker.scale.z = obstacle["dims"][2]
            marker.color.a = 1.0  # Fully opaque
            marker.color.r = 0.0  # Red for visibility
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.position.x = obstacle["pose"][0]
            marker.pose.position.y = obstacle["pose"][1]
            marker.pose.position.z = obstacle["pose"][2]
            marker.pose.orientation.x = float(obstacle["pose"][3])
            marker.pose.orientation.y = float(obstacle["pose"][4])
            marker.pose.orientation.z = float(obstacle["pose"][5])
            marker.pose.orientation.w = float(obstacle["pose"][6])

            marker_array.markers.append(marker)

        self.marker_publisher.publish(marker_array)