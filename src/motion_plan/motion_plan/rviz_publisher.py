from sensor_msgs.msg import JointState
from visualization_msgs.msg import MarkerArray, Marker
import tf2_ros
from geometry_msgs.msg import TransformStamped
from curobo.geom.types import Mesh, Cuboid

class RvizPublisher:
    def __init__(self, joint_names, obstacles, ParentNode):
        self.ros_node = ParentNode

        # Publishers
        self.joint_publisher = self.ros_node.create_publisher(JointState, "/joint_states", 10)
        self.marker_publisher = self.ros_node.create_publisher(MarkerArray, "/obstacle_markers", 10)

        # TF2 Broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self.ros_node)
        # self.obstacle_tf_broadcaster = tf2_ros.TransformBroadcaster(self.ros_node)

        self.joint_names = joint_names
        self.obstacles = obstacles
        self.current_joint_state = [0, 0, 0, 0, 0, 0]
        self.ros_node.get_logger().info("RVIZ Publisher Initialized")

    def publish_joint_states(self, joint_states=None):
        """Publish joint states and update base link transform"""
        ts = self.ros_node.get_clock().now().to_msg()
        joint_state = JointState()
        joint_state.header.stamp = ts
        joint_state.header.frame_id = "base_link"
        joint_state.name = self.joint_names
        joint_state.position = joint_states if joint_states is not None else self.current_joint_state
        self.joint_publisher.publish(joint_state)
        transform_stamped = TransformStamped()
        transform_stamped.header.stamp = ts

        transform_stamped.header.frame_id = "world"
        transform_stamped.child_frame_id = "base_link"
        transform_stamped.transform.translation.x = 0.0  # Modify as needed
        transform_stamped.transform.translation.y = 0.0
        transform_stamped.transform.translation.z = 0.0
        transform_stamped.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(transform_stamped)

    def publish_obstacles(self):
        marker_array = MarkerArray()
        self.ros_node.get_logger().info("Publishing obstacles to RViz")
        # Loop through all obstacles in self.obstacles
        for i, obstacle in enumerate(self.obstacles):
            marker = Marker()
            marker.header.frame_id = "world"  # Coordinate frame for the mesh
            marker.ns = "obstacles"           # Namespace for this obstacle
            marker.id = i                     # Unique ID for each obstacle

            # Check if the obstacle is a Mesh or a Cuboid
            if isinstance(obstacle, Mesh):
                marker.type = Marker.MESH_RESOURCE  # Use Mesh type for markers
                marker.action = Marker.ADD         # Action to add the marker

                # Set the mesh resource path (Make sure this path is correct)
                mesh_file_path = obstacle.file_path
                marker.mesh_resource = "package://motion_plan/CAD/base_link.STL"

                # Set scale (you may need to adjust depending on the size of your mesh)
                marker.scale.x = 1.0
                marker.scale.y = 1.0
                marker.scale.z = 1.0

                # Set color (optional)
                marker.color.a = 1.0  # Fully opaque
                marker.color.r = 0.0  # Red for visibility
                marker.color.g = 1.0
                marker.color.b = 0.0

                # Set pose (position and orientation)
                marker.pose.position.x = float(obstacle.pose[0])
                marker.pose.position.y = float(obstacle.pose[1])
                marker.pose.position.z = float(obstacle.pose[2])
                marker.pose.orientation.x = float(obstacle.pose[3])
                marker.pose.orientation.y = float(obstacle.pose[4])
                marker.pose.orientation.z = float(obstacle.pose[5])
                marker.pose.orientation.w = float(obstacle.pose[6])

            elif isinstance(obstacle, Cuboid):
                marker.type = Marker.CUBE       # Use Cube type for cuboids
                marker.action = Marker.ADD      # Action to add the marker

                # Set the cuboid's scale based on its dimensions
                marker.scale.x = float(obstacle.dims[0])
                marker.scale.y = float(obstacle.dims[1])
                marker.scale.z = float(obstacle.dims[2])

                # Set color (optional)
                marker.color.a = 1.0  # Fully opaque
                marker.color.r = 0.0  # Red for visibility
                marker.color.g = 1.0
                marker.color.b = 0.0

                # Set pose (position and orientation)
                marker.pose.position.x = float(obstacle.pose[0])
                marker.pose.position.y = float(obstacle.pose[1])
                marker.pose.position.z = float(obstacle.pose[2])
                marker.pose.orientation.x = float(obstacle.pose[3])
                marker.pose.orientation.y = float(obstacle.pose[4])
                marker.pose.orientation.z = float(obstacle.pose[5])
                marker.pose.orientation.w = float(obstacle.pose[6])

            # Add the marker to the MarkerArray
            marker_array.markers.append(marker)

        # Publish the MarkerArray
        self.ros_node.get_logger().info(str(marker_array))
        self.marker_publisher.publish(marker_array)
