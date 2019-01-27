"""Ask robot to chase block."""

import copy
import sys

from geometry_msgs.msg import Pose, Point, Quaternion
import moveit_commander
import intera_interface
import rospy
from tf import TransformListener
from copy import deepcopy
from get_task_srv.srv import get_task

class Robot(object):
    def __init__(self, limb='right', tip_name="right_gripper_tip"):
        self._limb_name = limb
        self._limb = intera_interface.Limb(limb)        
        self.gripper = intera_interface.Gripper(limb)
        self._tip_name = tip_name        

    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(start_angles)

    def approach(self, pose):
        joint_angles = self._limb.ik_request(pose, self._tip_name)
        self._guarded_move_to_joint_position(joint_angles)

    def _guarded_move_to_joint_position(self, joint_angles, timeout=60.0):
        self._limb.set_joint_position_speed(0.1)
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles, timeout=timeout)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")


class Runner:

    def __init__(self, robot):
        self.task = 0
        self._robot_frame = "base_d"
        self._peg_frame = "peg"
        self._hole_frame = "hole"
        self._cur_gripper_pose = Pose()
        self._prev_gripper_pose = Pose()
        self._tf_listener = TransformListener()
        self.robot = robot

    def handle_get_task(self, req):
        return self.task

    def reach_peg(self, use_wp1=False, use_wp2=False, peg_hover=0.10): #hover_dis = 0.15, 0.2
        self.robot.gripper.open()
        self._tf_listener.waitForTransform(self._robot_frame, self._peg_frame, rospy.Time(0), rospy.Duration(2))
        trans, _ = self._tf_listener.lookupTransform(self._robot_frame, self._peg_frame, rospy.Time(0))
        peg_position = Point(x=trans[0], y=trans[1], z=trans[2])
        
        if use_wp1:
	    wp1 = Pose()
	    wp1.position = Point(0.657, 0.372, 0.15)
	    wp1.orientation = Quaternion(0, 1, 0, 0)
	    self.robot.approach(wp1)

        if use_wp2:
	    wp2 = Pose()
	    wp2.position = Point(0.672, 0.170, 0.15)
	    wp2.orientation = Quaternion(0, 1, 0, 0)
	    self.robot.approach(wp2)

        print("Reaching Peg")
        self._cur_gripper_pose = Pose()
        self._cur_gripper_pose.position = peg_position
        self._cur_gripper_pose.position.z += peg_hover
        self._cur_gripper_pose.position.y -= 0.01
        self._cur_gripper_pose.position.x += 0.02
        self._cur_gripper_pose.orientation.x = 0
        self._cur_gripper_pose.orientation.y = 1
        self._cur_gripper_pose.orientation.z = 0
        self._cur_gripper_pose.orientation.w = 0
        self.robot.approach(self._cur_gripper_pose)

    def pick_up_peg(self, peg_hover=0.10): #hover_dis = 0.2, 0.3
        self.task += 1
        down = peg_hover + 0.02
        self._cur_gripper_pose.position.z -= down
        print("Picking up block")
        self.robot.approach(self._cur_gripper_pose)
        self.robot.gripper.close()
    
    def reach_box_lid(self, use_wp1=False, use_wp2=False, use_wp3=False, use_wp4=False, peg_hover=0.1, lid_hover=0.22): #peg_hover = 0.2, 0.3 lid_hover=0.17, 0.27
        self.task += 1
        self._cur_gripper_pose.position.z += peg_hover
        print("Going up")
        self.robot.approach(self._cur_gripper_pose)

        if use_wp1:
	    wp1 = Pose()
	    wp1.position = Point(0.91, 0.207, 0.2)
	    wp1.orientation = Quaternion(0, 1, 0, 0)
	    self.robot.approach(wp1)

        if use_wp2:
	    wp2 = Pose()
	    wp2.position = Point(0.677, 0.244, 0.2)
	    wp2.orientation = Quaternion(0, 1, 0, 0)
	    self.robot.approach(wp2)

        if use_wp3:
	    wp3 = Pose()
	    wp3.position = Point(0.622, 0.291, 0.2)
	    wp3.orientation = Quaternion(0, 1, 0, 0)
	    self.robot.approach(wp3)
        
        if use_wp4:
	    wp4 = Pose()
	    wp4.position = Point(0.73, 0.0, 0.3)
	    wp4.orientation = Quaternion(0, 1, 0, 0)
	    self.robot.approach(wp4)
        
        self._tf_listener.waitForTransform(self._robot_frame, self._hole_frame, rospy.Time(0), rospy.Duration(2))
        trans, _ = self._tf_listener.lookupTransform(self._robot_frame, self._hole_frame, rospy.Time(0))
        
        hole_position = Point(x=trans[0], y=trans[1], z=trans[2])
        self._cur_gripper_pose = Pose()
        self._cur_gripper_pose.position = hole_position
        self._cur_gripper_pose.position.z += lid_hover
        self._cur_gripper_pose.position.x -= 0.01
        #self._cur_gripper_pose.position.y -= 0.01
        self._cur_gripper_pose.orientation.x = 0
        self._cur_gripper_pose.orientation.y = 1
        self._cur_gripper_pose.orientation.z = 0
        self._cur_gripper_pose.orientation.w = 0

        print("Going above hole")
        self.robot.approach(self._cur_gripper_pose)

    def insert_peg_in_lid(self, lid_hover=0.22): #lid_hover = 0.17, 0.27
        self.task += 1
        print("Going down")
        down = lid_hover - 0.06
        self._cur_gripper_pose.position.z -= down
        self.robot.approach(self._cur_gripper_pose)
        self._prev_gripper_pose = deepcopy(self._cur_gripper_pose)

    def open_lid(self):
        self.task += 1
        self._tf_listener.waitForTransform(self._robot_frame, "box", rospy.Time(0), rospy.Duration(2))
        trans, _ = self._tf_listener.lookupTransform(self._robot_frame, "box", rospy.Time(0))
        self._cur_gripper_pose.position.x = trans[0] - 0.01
        self._cur_gripper_pose.position.y = trans[1] + 0.06
        self.robot.approach(self._cur_gripper_pose)

    def close_lid(self):
        self.task += 1
        self._prev_gripper_pose.position.x -= 0.01
        self.robot.approach(self._prev_gripper_pose)
        self._cur_gripper_pose = deepcopy(self._prev_gripper_pose)
    
    def remove_peg(self, lid_hover=0.22): #lid_hover = 0.17, 0.27
        self.task += 1
        up = lid_hover - 0.06
        self._cur_gripper_pose.position.z += up
        self.robot.approach(self._cur_gripper_pose)

    def get_obs(self, peg_use_wp1, peg_use_wp2, box_use_wp1, box_use_wp2, box_use_wp3, box_use_wp4, peg_hover, lid_hover):
	self.reach_peg(use_wp1=peg_use_wp1, use_wp2=peg_use_wp2, peg_hover=peg_hover) #0
	self.pick_up_peg(peg_hover=peg_hover) #1
	self.reach_box_lid(use_wp1=box_use_wp1, use_wp2=box_use_wp2, use_wp3=box_use_wp3, use_wp4=box_use_wp4, peg_hover=peg_hover, lid_hover=lid_hover) #2
	self.insert_peg_in_lid(lid_hover=lid_hover) #3
	self.open_lid() #4
	self.close_lid() #5
	self.remove_peg() #6      

if __name__ == '__main__':
    INITIAL_ROBOT_JOINT_POS = {
        'right_j0': -0.140923828125,
        'right_j1': -1.2789248046875,
        'right_j2': -3.043166015625,
        'right_j3': -2.139623046875,
        'right_j4': -0.047607421875,
        'right_j5': -0.7052822265625,
        'right_j6': -1.4102060546875,
    }
    #CONFIG
    peg_use_wp1 = False
    peg_use_wp2 = False
    peg_hover = 0.2 #0.10, 0.15, 0.2
    box_use_wp1 = False
    box_use_wp2 = False
    box_use_wp3 = False
    box_use_wp4 = True
    lid_hover = 0.17 #0.17, 0.19, 0.22

    rospy.init_node('demo_toyworld', anonymous=True)
    
    robot = Robot()
    robot.move_to_start(start_angles=INITIAL_ROBOT_JOINT_POS)
    usr_input = raw_input("Type 'done' when you are ready.").lower()
    if usr_input == 'done':
	runner = Runner(robot)
        print("Starting get_task service.")
	rospy.Service('get_task', get_task, runner.handle_get_task)
	runner.get_obs(peg_use_wp1, peg_use_wp2, box_use_wp1, box_use_wp2, box_use_wp3, box_use_wp4, peg_hover, lid_hover)
    elif usr_input == 'exit':
        sys.exit()
