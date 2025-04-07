source install/setup.bash
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [0.5, 0.6, 0.2, 0 , 0.0, -1.0, 0.0]}"
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [0.5, 0.6, 0.03, 0, 0.0, -1.0, 0.0]}"
ros2 action send_goal /publish_joints curobo_action/action/PublishJoints "{positions: [1.0], indices: [7]}"
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [0.5, 0.6, 0.3,0 , 0.0, -1.0, 0.0]}"
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [-0.6, 0.6, 0.3,0 , 0.0, -1.0, 0.0]}"
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [-0.6, 0.6, 0.03, 0 , 0.0, -1.0, 0.0]}"
ros2 action send_goal /publish_joints curobo_action/action/PublishJoints "{positions: [0.0], indices: [7]}"
ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [-0.6, 0.6, 0.3,0 , 0.0, -1.0, 0.0]}"


#ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [-0.5, -0.7, 0.1, 0.0, 0.0, -1.0, 0.0]}"
#ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [-0.5, -0.7, 0.02, 0.0, 0.0, -1.0, 0.0]}"
#ros2 action send_goal /publish_joints curobo_action/action/PublishJoints "{positions: [1.0], indices: [7]}"
#ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [-0.5, -0.7, 0.25, 0.0, 0.0, -1.0, 0.0]}"
#ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [0.5, 0.7, 0.25, 0.0, 0.0, -1.0, 0.0]}"
#ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [0.5, 0.7, 0.02, 0.0, 0.0, -1.0, 0.0]}"
#ros2 action send_goal /publish_joints curobo_action/action/PublishJoints "{positions: [0.0], indices: [7]}"
#ros2 action send_goal /movel curobo_action/action/MoveL "{goal_pose: [0.5, 0.7, 0.2, 0.0, 0.0, -1.0, 0.0]}"
