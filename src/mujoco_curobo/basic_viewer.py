import yaml
import os
import numpy as np
import mujoco
import time
import torch
from mujoco_parser import MuJoCoParserClass


xml_path = './assets/ur5e/scene_ur5e_2f140_obj (sution).xml'
env = MuJoCoParserClass(name='UR5e with RG2 gripper', rel_xml_path=xml_path, VERBOSE=True)




env.init_viewer(viewer_title='UR5e with RG2 gripper', viewer_width=1200, viewer_height=800,
                viewer_hide_menus=True)
env.update_viewer(azimuth=66.08, distance=3.0, elevation=-50, lookat=[0.4, 0.18, 0.71],
                  VIS_TRANSPARENT=False, VIS_CONTACTPOINT=False,
                  contactwidth=0.05, contactheight=0.05, contactrgba=np.array([1, 0, 0, 1]),
                  VIS_JOINT=True, jointlength=0.25, jointwidth=0.05, jointrgba=[0.2, 0.6, 0.8, 0.6])
# q_init_upright = np.array([0, -np.pi / 2, 0, 0, np.pi / 2, 0])
# env.reset()
# env.forward(q=q_init_upright, joint_idxs=env.idxs_forward)

while (env.get_sim_time() < 100.0) and env.is_viewer_alive():
    # if env.get_sim_time() < 2:
    #     env.step([1],[7])
    # else:
    #     env.step([0],[7])
    # env.step([-1.57  ,0.   ,0.    ,1.57  ,0.  ,  0.  , 1.0],[0, 1, 2, 3, 4, 5, 6])
    env.render()
# raceback (most recent call last):
#   File "/home/jetson3/ros2_ws/src/mujoco_curobo/basic_viewer.py", line 11, in <module>
#     env = MuJoCoParserClass(name='UR5e with RG2 gripper', rel_xml_path=xml_path, VERBOSE=True)
#   File "/home/jetson3/ros2_ws/src/mujoco_curobo/mujoco_parser.py", line 30, in __init__
#     self._parse_xml()
#   File "/home/jetson3/ros2_ws/src/mujoco_curobo/mujoco_parser.py", line 48, in _parse_xml
#     self.model            = mujoco.MjModel.from_xml_path(self.full_xml_path)
# ValueError: XML Error: Schema violation: unrecognized attribute: 'gap'

# Element 'adhesion', line 0
