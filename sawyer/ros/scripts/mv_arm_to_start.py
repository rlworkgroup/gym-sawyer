import rospy
import intera_interface

INITIAL_ROBOT_JOINT_POS = {
    'right_j0': -0.79328125, 
    'right_j1': 0.4875126953125,
    'right_j2': -2.848359375, 
    'right_j3': 0.9184482421875,
    'right_j4': -2.932041015625,
    'right_j5': -0.0460625,
    'right_j6': 0.0387646484375,
}


print("\nMoving arm to detect origin\n")
rospy.init_node("Init_arm_pos")
limb = intera_interface.Limb('right')
limb.set_joint_position_speed(0.2)
limb.move_to_joint_positions(INITIAL_ROBOT_JOINT_POS, timeout=60.0)

