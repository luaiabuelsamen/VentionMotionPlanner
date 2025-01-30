import numpy as np
import matplotlib.pyplot as plt
from mujoco_parser import MuJoCoParserClass
from util import sample_xyzs, rpy2r, get_interp_const_vel_traj

xml_path = 'assets/ur5e/scene_ur5e_rg2_d435i_obj.xml'
env = MuJoCoParserClass(name='UR5e with RG2 gripper', rel_xml_path=xml_path, VERBOSE=True)

obj_names = [body_name for body_name in env.body_names if body_name is not None and (body_name.startswith("obj_"))]
n_obj = len(obj_names)
xyzs = sample_xyzs(n_sample=n_obj, x_range=[0.75, 1.25], y_range=[-0.38, 0.38], z_range=[0.025, 0.025], min_dist=0.2)
colors = np.array([plt.cm.gist_rainbow(x) for x in np.linspace(0, 1, n_obj)])

for obj_idx, obj_name in enumerate(obj_names):
    jntadr = env.model.body(obj_name).jntadr[0]
    env.model.joint(jntadr).qpos0[:3] = xyzs[obj_idx, :]
    geomadr = env.model.body(obj_name).geomadr[0]

platform_xyz = np.random.uniform([0.6, -0.3, 0.01], [1.0, 0.3, 0.01])
env.model.body('red_platform').pos = platform_xyz
env.model.body('base').pos = np.array([0.18, 0, 0])

q_init_upright = np.array([0, -np.pi/2, 0, 0, np.pi/2, 0])
env.reset()
env.forward(q=q_init_upright, joint_idxs=env.idxs_forward)
R_trgt = rpy2r(np.radians([-180, 0, 90]))

pick_position = env.get_p_body(obj_names[0])
pick_position[2] += 0.005
pre_grasp_position = pick_position + np.array([0.0, 0.0, 0.1])
q_ik_pregrasp = env.solve_ik_repel(body_name='tcp_link', p_trgt=pre_grasp_position, R_trgt=R_trgt, IK_P=True, IK_R=True, q_init=np.array(q_init_upright), idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian, RESET=False, DO_RENDER=False, render_every=1, th=1 * np.pi / 180.0, err_th=1e-3, w_weight=0.5)
q_ik = env.solve_ik(body_name='tcp_link', p_trgt=pick_position, R_trgt=R_trgt, IK_P=True, IK_R=True, q_init=np.array(q_ik_pregrasp), idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian, RESET=False, DO_RENDER=False, render_every=1, th=1 * np.pi / 180.0, err_th=1e-3, w_weight=0.5)
post_pick_position = pick_position + np.array([0.0, 0.0, 0.1])
q_ik_postpick = env.solve_ik(body_name='tcp_link', p_trgt=post_pick_position, R_trgt=R_trgt, IK_P=True, IK_R=True, q_init=np.array(q_ik), idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian, RESET=False, DO_RENDER=False, render_every=1, th=1 * np.pi / 180.0, err_th=1e-3, w_weight=0.5)

q_traj_combined = np.vstack([q_init_upright, q_ik_pregrasp, q_ik])
times, q_traj = get_interp_const_vel_traj(q_traj_combined, vel=np.radians(30), HZ=env.HZ)

place_position = platform_xyz + np.array([0.0, 0.0, 0.03])
q_ik_place = env.solve_ik(body_name='tcp_link', p_trgt=place_position, R_trgt=R_trgt, IK_P=True, IK_R=True, q_init=np.array(q_traj[-1, :]), idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian, RESET=False, DO_RENDER=False, render_every=1, th=1 * np.pi / 180.0, err_th=1e-3, w_weight=0.5)
pre_place_position = place_position + np.array([0.0, 0.0, 0.1])
q_ik_preplace = env.solve_ik(body_name='tcp_link', p_trgt=pre_place_position, R_trgt=R_trgt, IK_P=True, IK_R=True, q_init=np.array(q_ik_place), idxs_forward=env.idxs_forward, idxs_jacobian=env.idxs_jacobian, RESET=False, DO_RENDER=False, render_every=1, th=1 * np.pi / 180.0, err_th=1e-3, w_weight=0.5)

q_traj_place_combined = np.vstack([q_ik_postpick, q_ik_preplace, q_ik_place])
times_place, q_traj_place = get_interp_const_vel_traj(q_traj_place_combined, vel=np.radians(15), HZ=env.HZ)

env.init_viewer(viewer_title='UR5e with RG2 gripper', viewer_width=1200, viewer_height=800, viewer_hide_menus=True)
env.update_viewer(azimuth=66.08, distance=3.0, elevation=-50, lookat=[0.4, 0.18, 0.71], VIS_TRANSPARENT=False, VIS_CONTACTPOINT=False, contactwidth=0.05, contactheight=0.05, contactrgba=np.array([1, 0, 0, 1]), VIS_JOINT=True, jointlength=0.25, jointwidth=0.05, jointrgba=[0.2, 0.6, 0.8, 0.6])

tick = 0
while tick < q_traj.shape[0]:
    q = q_traj[tick, :]
    tcp_position = env.get_p_body('tcp_link')
    q = np.append(q_traj[tick, :], 1.0)
    env.step(ctrl=q, ctrl_idxs=[0, 1, 2, 3, 4, 5, 6])
    env.render()
    tick += 1

for _ in range(500):
    q = np.append(q_traj[-1, :], 0)
    env.step(ctrl=q, ctrl_idxs=[0, 1, 2, 3, 4, 5, 6])
    env.render()

tick = 0
gripper_opened = False
while tick < q_traj_place.shape[0]:
    q = q_traj_place[tick, :]
    tcp_position = env.get_p_body('tcp_link')
    if not gripper_opened and np.linalg.norm(tcp_position - pre_place_position) < 0.01:
        gripper_opened = True
        q = np.append(q_traj_place[tick, :], 1.0)
    else:
        q = np.append(q_traj_place[tick, :], 0.0 if not gripper_opened else 1.0)
    env.step(ctrl=q, ctrl_idxs=[0, 1, 2, 3, 4, 5, 6])
    env.render()
    tick += 1