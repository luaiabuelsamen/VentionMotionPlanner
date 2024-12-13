#!/usr/bin/env python3

import os
import yaml

from ament_index_python.packages import get_package_share_directory


from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

class RvizPublisher():
    def __init__(self):

        # Publishers
        self.joint_publisher = self.create_publisher(JointState, "/joint_states", 10)
        self.marker_publisher = self.create_publisher(MarkerArray, "/obstacle_markers", 10)

        # Robot joint names
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        self.create_timer(0.1, self.publish_joint_states)  # 10 Hz
        self.create_timer(0.5, self.publish_obstacles)  # 2 Hz

        self.get_logger().info("Collision Avoidance Joint Publisher Initialized")

    def publish_joint_states(self, joint_angles):
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = self.joint_names

        joint_state.position = joint_angles
        self.joint_publisher.publish(joint_state)

    def publish_obstacles(self):
        pass
        # marker_array = MarkerArray()
        # for i, (obstacle_name, obstacle) in enumerate(self.obstacles["cuboid"].items()):
        #     marker = Marker()
        #     marker.header.frame_id = "world"
        #     marker.header.stamp = self.get_clock().now().to_msg()
        #     marker.ns = "obstacles"
        #     marker.id = i
        #     marker.type = Marker.CUBE
        #     marker.action = Marker.ADD
        #     marker.scale.x = obstacle["dims"][0]
        #     marker.scale.y = obstacle["dims"][1]
        #     marker.scale.z = obstacle["dims"][2]
        #     marker.color.a = 1.0  # Fully opaque
        #     marker.color.r = 1.0  # Red for visibility
        #     marker.color.g = 0.0
        #     marker.color.b = 0.0
        #     marker.pose.position.x = obstacle["pose"][0]
        #     marker.pose.position.y = obstacle["pose"][1]
        #     marker.pose.position.z = obstacle["pose"][2]
        #     marker.pose.orientation.x = float(obstacle["pose"][3])
        #     marker.pose.orientation.y = float(obstacle["pose"][4])
        #     marker.pose.orientation.z = float(obstacle["pose"][5])
        #     marker.pose.orientation.w = float(obstacle["pose"][6])

        #     marker_array.markers.append(marker)

        # self.marker_publisher.publish(marker_array)
        # self.get_logger().info(f"Published {len(marker_array.markers)} obstacle markers.")

