source install/setup.bash
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [0, 0.9, 0.1, 0.0, 0.0, -1.0, 0.0], mpc: false}"
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [-0.9, 0, 0.22, 0.0, 0.0, -1.0, 0.0], mpc: false}"
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [0, -0.5, 0.2, 0.0, 0.0, -1.0, 0.0], mpc: false}"