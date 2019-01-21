"""Ask robot to chase block."""

import copy
import sys

from geometry_msgs.msg import Pose, Point
import moveit_commander
import intera_interface
import rospy
from tf import TransformListener
from copy import deepcopy
from sawyer.ros.worlds import BlockWorld
from get_task_srv.srv import get_task

TASK = 0

class ChaseBlock(object):
    def __init__(self, limb='right', tip_name="right_gripper_tip"):
        self._limb_name = limb
        self._limb = intera_interface.Limb(limb)
        self._limb.set_joint_position_speed(0.1)
        self._tip_name = tip_name
        self._gripper = intera_interface.Gripper(limb)

    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(start_angles)

    def approach(self, pose):
        self._limb.set_joint_position_speed(0.1)
        joint_angles = self._limb.ik_request(pose, self._tip_name)
        self._guarded_move_to_joint_position(joint_angles)

    def _guarded_move_to_joint_position(self, joint_angles, timeout=60.0):
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles, timeout=timeout)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")

class Runner:

    def __init__(self):
        self.task = 0

    def handle_get_task(self, req):
        return self.task

    def run(self):
        moveit_commander.roscpp_initialize(sys.argv)

        rospy.init_node('demo_chaseblock', anonymous=True)

        moveit_robot = moveit_commander.RobotCommander()
        moveit_scene = moveit_commander.PlanningSceneInterface()
        moveit_group_name = 'right_arm'
        moveit_group = moveit_commander.MoveGroupCommander(moveit_group_name)
        moveit_frame = moveit_robot.get_planning_frame()

        robot_frame = "base_d"
        block_frame = "peg"
        rospy.Service('get_task', get_task, self.handle_get_task)

        cb = ChaseBlock()
        
        # Open Gripper
        cb._gripper.open()
        tf_listener = TransformListener()
        tf_listener.waitForTransform(robot_frame, block_frame, rospy.Time(0), rospy.Duration(2))
        trans, _ = tf_listener.lookupTransform(robot_frame, block_frame, rospy.Time(0))
        block_position = Point(x=trans[0], y=trans[1], z=trans[2])

        # Reach block position
        print("Going above block")
        target_pose = Pose()
        target_pose.position = block_position
        target_pose.position.z += 0.10
        target_pose.position.y -= 0.01
        target_pose.position.x -= 0.01
        target_pose.orientation.x = 0
        target_pose.orientation.y = 1
        target_pose.orientation.z = 0
        target_pose.orientation.w = 0
        cb.approach(target_pose)
        
        # Pick up block
        self.task += 1
        target_pose.position.z -= 0.15
        print("Picking up block")
        cb.approach(target_pose)
        cb._gripper.close()
        target_pose.position.z += 0.13
        print("Going up")
        cb.approach(target_pose)
        
        # Insert in lid
        self.task += 1
        hole_frame = "hole"
        tf_listener.waitForTransform(robot_frame, hole_frame, rospy.Time(0), rospy.Duration(2))
        trans, _ = tf_listener.lookupTransform(robot_frame, hole_frame, rospy.Time(0))
        
        hole_position = Point(x=trans[0], y=trans[1], z=trans[2])
        target_pose = Pose()
        target_pose.position = hole_position
        target_pose.position.z += 0.22
        target_pose.position.x -= 0.03
        target_pose.position.y -= 0.01
        target_pose.orientation.x = 0
        target_pose.orientation.y = 1
        target_pose.orientation.z = 0
        target_pose.orientation.w = 0

        print("Going above hole")
        cb.approach(target_pose)
            
        print("Going down")
        target_pose.position.z -= 0.16
        cb.approach(target_pose)
        target_pos_copy = deepcopy(target_pose)
        
        # Open lid
        self.task += 1
        tf_listener.waitForTransform(robot_frame, "box", rospy.Time(0), rospy.Duration(2))
        trans, _ = tf_listener.lookupTransform(robot_frame, "box", rospy.Time(0))
        target_pose.position.x = trans[0] - 0.05
        target_pose.position.y = trans[1] + 0.04
        #cur_joints = cb._limb.joint_angles()
        #cur_pose = cb._limb.fk_request(cur_joints).pose_stamp[0].pose
        #new_pose = copy.deepcopy(cur_pose)
        #new_pose.position.x -= 0.02
        cb.approach(target_pose)

        # Close lid
        self.task += 1
        target_pos_copy.position.x -= 0.01
        cb.approach(target_pos_copy)

        # Remove from lid
        self.task += 1
        target_pos_copy.position.z += 0.16
        cb.approach(target_pos_copy)

if __name__ == '__main__':
    runner = Runner()
    runner.run()
