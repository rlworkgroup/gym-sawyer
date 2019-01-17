"""Ask robot to open box lid."""

import copy
import sys

from geometry_msgs.msg import Pose, Point
import moveit_commander
import intera_interface
import rospy
from sawyer.ros.worlds import ToyWorld

class ToyBoxExp:
    def __init__(self, limb='right', tip_name="right_gripper_tip"):
        self._limb_name = limb
        self._limb = intera_interface.Limb(limb)
        self._limb.set_joint_position_speed(0.1)
        self._tip_name = tip_name
        self._gripper = intera_interface.Gripper()


def run():
    moveit_commander.roscpp_initialize(sys.argv)

    rospy.init_node('demo_toyworld', anonymous=True)

    moveit_robot = moveit_commander.RobotCommander()
    moveit_scene = moveit_commander.PlanningSceneInterface()
    moveit_group_name = 'right_arm'
    moveit_group = moveit_commander.MoveGroupCommander(moveit_group_name)
    moveit_frame = moveit_robot.get_planning_frame()

    toyworld = ToyWorld(moveit_scene, moveit_frame, simulated=False)
    toyworld.initialize()
    obs = toyworld.get_observation()
    print(obs)

if __name__ == '__main__':
    run()
