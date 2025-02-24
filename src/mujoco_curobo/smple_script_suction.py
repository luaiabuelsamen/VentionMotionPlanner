import numpy as np
import mujoco
import time
from mujoco_parser import MuJoCoParserClass

# Load MuJoCo environment

import numpy as np
import mujoco
import time
from mujoco_parser import MuJoCoParserClass

# Load MuJoCo environment
xml_path = './assets/ur5e/scene_ur5e_2f140_obj.xml'
env = MuJoCoParserClass(name='UR5e Vacuum Gripper', rel_xml_path=xml_path, VERBOSE=True)

# Initialize viewer
env.init_viewer(viewer_title='UR5e Vacuum Gripper', viewer_width=1200, viewer_height=800,
                viewer_hide_menus=True)
env.update_viewer(azimuth=66.08, distance=3.0, elevation=-50, lookat=[0.4, 0.18, 0.71])

# Get body IDs
vacuum_gripper_id = env.model.body('vacuum_gripper').id
cube_id = env.model.body('cube_1').id

# Define joint positions for pick and place
pick_joint_angles = np.array([0.0, -1.57, 1.57, 0.0, 0.0, 0.0])  # Pick position
lift_joint_angles = np.array([0.0, -1.2, 1.2, 0.0, 0.0, 0.0])  # Lift after pick
place_joint_angles = np.array([0.5, -1.57, 1.57, 0.0, 0.0, 0.0])  # Place position

# Track welding state
holding_block = None

def add_weld_constraint(env):
    """Weld the block to the vacuum gripper."""
    with env.model:
        env.model.equality.add(
            "weld",
            body1="vacuum_gripper",
            body2="cube_1",
            relpose=[0, 0, 0, 1, 0, 0, 0]  # No relative offset
        )
    print("Block welded to vacuum gripper.")

def remove_weld_constraint(env):
    """Release the block."""
    with env.model:
        env.model.equality.clear()
    print("Block released.")

# Simulation loop
try:
    while env.get_sim_time() < 100.0 and env.is_viewer_alive():
        ee_pos = env.data.xpos[vacuum_gripper_id]  # Get end-effector position
        cube_pos = env.data.xpos[cube_id]  # Get block position

        # Step 1: Move to Pick Position
        if holding_block is None:
            env.step(ctrl=pick_joint_angles, ctrl_idxs=[0, 1, 2, 3, 4, 5])
            dist = np.linalg.norm(ee_pos - cube_pos)

            # Attach block if close
            if dist < 0.02:
                holding_block = "cube_1"
                add_weld_constraint(env)

        # Step 2: Lift and Move to Place
        elif holding_block:
            env.step(ctrl=lift_joint_angles, ctrl_idxs=[0, 1, 2, 3, 4, 5])
            env.step(ctrl=place_joint_angles, ctrl_idxs=[0, 1, 2, 3, 4, 5])

            # Step 3: Release the Block when touching the floor
            if cube_pos[2] <= 0.02:
                remove_weld_constraint(env)
                holding_block = None

        env.render()

except KeyboardInterrupt:
    print("Simulation interrupted.")

finally:
    print("Exporter process finished.")
