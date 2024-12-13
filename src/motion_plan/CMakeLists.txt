cmake_minimum_required(VERSION 3.5)
project(motion_plan)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

# Install YAML file and any additional configurations
install(
    DIRECTORY configs/
    DESTINATION share/${PROJECT_NAME}/configs
)

# Export dependencies
ament_export_dependencies(rclpy sensor_msgs visualization_msgs)

ament_package()
