"""Ask robot to chase block."""

import copy
import sys

from geometry_msgs.msg import Pose, Point
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

    def reach_peg(self):
        self.robot.gripper.open()
        self._tf_listener.waitForTransform(self._robot_frame, self._peg_frame, rospy.Time(0), rospy.Duration(2))
        trans, _ = self._tf_listener.lookupTransform(self._robot_frame, self._peg_frame, rospy.Time(0))
        peg_position = Point(x=trans[0], y=trans[1], z=trans[2])
        
        print("Going above peg")
        self._cur_gripper_pose = Pose()
        self._cur_gripper_pose.position = peg_position
        self._cur_gripper_pose.position.z += 0.10
        self._cur_gripper_pose.position.y -= 0.01
        self._cur_gripper_pose.position.x += 0.02
        self._cur_gripper_pose.orientation.x = 0
        self._cur_gripper_pose.orientation.y = 1
        self._cur_gripper_pose.orientation.z = 0
        self._cur_gripper_pose.orientation.w = 0
        self.robot.approach(self._cur_gripper_pose)

    def pick_up_peg(self):
        self.task += 1
        self._cur_gripper_pose.position.z -= 0.12
        print("Picking up block")
        self.robot.approach(self._cur_gripper_pose)
        self.robot.gripper.close()
    
    def reach_box_lid(self):
        self.task += 1
        self._cur_gripper_pose.position.z += 0.13
        print("Going up")
        self.robot.approach(self._cur_gripper_pose)

        self._tf_listener.waitForTransform(self._robot_frame, self._hole_frame, rospy.Time(0), rospy.Duration(2))
        trans, _ = self._tf_listener.lookupTransform(self._robot_frame, self._hole_frame, rospy.Time(0))
        
        hole_position = Point(x=trans[0], y=trans[1], z=trans[2])
        self._cur_gripper_pose = Pose()
        self._cur_gripper_pose.position = hole_position
        self._cur_gripper_pose.position.z += 0.22
        self._cur_gripper_pose.position.x -= 0.015
        #self._cur_gripper_pose.position.y -= 0.01
        self._cur_gripper_pose.orientation.x = 0
        self._cur_gripper_pose.orientation.y = 1
        self._cur_gripper_pose.orientation.z = 0
        self._cur_gripper_pose.orientation.w = 0

        print("Going above hole")
        self.robot.approach(self._cur_gripper_pose)

    def insert_peg_in_lid(self):
        self.task += 1
        print("Going down")
        self._cur_gripper_pose.position.z -= 0.16
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
    
    def remove_peg(self):
        self.task += 1
        self._cur_gripper_pose.position.z += 0.16
        self.robot.approach(self._cur_gripper_pose)

    def get_obs_0(self):
        self.reach_peg() #0
        self.pick_up_peg() #1
        self.reach_box_lid() #2
        self.insert_peg_in_lid() #3
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

    rospy.init_node('demo_toyworld', anonymous=True)
    
    robot = Robot()
    robot.move_to_start(start_angles=INITIAL_ROBOT_JOINT_POS)
    
    runner = Runner(robot)
    rospy.Service('get_task', get_task, runner.handle_get_task)
    runner.get_obs_0()
