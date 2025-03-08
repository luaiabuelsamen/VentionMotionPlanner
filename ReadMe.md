# Mujoco Curobo ROS2 Package

This ROS2 package integrates the Curobo robot with the Mujoco simulation. The system is configured through a **`demos.yml`** file, which acts as the master configuration for various robot and world setups. 

## Folder Structure

```
ros2_ws/
├── assets
├── config
├── src
└── ReadMe.md
```

- **assets/**: Robot meshes and Mujoco models.
- **config/**: Configuration files, including `demos.yml`.
- **src/**: ROS2 source code.

## Configuration

The **`demos.yml`** file defines demo setups, including:
- **Robot & World Configurations**
- **Mujoco Model Path & Meshes**

Example:

```yaml
ur5e_with_gantry:
  curobo:
    world_config_file: config/curobo/world_config/gantry_table.yml
    robot_config_file: ur5e_robotiq_2f_140_x.yml
  mujoco:
    xml_path: assets/ur5e/scene_ur5e_2f140_obj_gantry.xml
    meshes:
      - bs_link: assets/ur5e/mesh/gantry/bs_link.STL
```

## Running the Demo

To launch a demo, use the following command:

```bash
ros2 launch curobo_api curobo.launch.py demo:=ur5e_with_gantry
```

This will load the robot and world configuration, and start the Mujoco simulation.

## Demo GIF

![Demo GIF](https://github.com/luaiabuelsamen/VentionMotionPlanner/blob/main/src/mujoco_curobo/demo.gif)

## Requirements

- ROS2 (e.g., Foxy, Galactic)
- Mujoco
- Curobo configurations

## Building the Workspace

Build the workspace before launching:

```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```