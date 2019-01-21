"""Ask robot to open box lid."""

import copy
import sys

from geometry_msgs.msg import Pose, Point
import moveit_commander
import intera_interface
import rospy
from sawyer.ros.worlds.toy_world import ToyWorld
from sawyer.ros.robots import Sawyer
from datetime import datetime
import pickle
from get_task_srv.srv import get_task

def run():
    moveit_commander.roscpp_initialize(sys.argv)

    rospy.init_node('demo_toyworld', anonymous=True)

    moveit_robot = moveit_commander.RobotCommander()
    moveit_scene = moveit_commander.PlanningSceneInterface()
    moveit_group_name = 'right_arm'
    moveit_group = moveit_commander.MoveGroupCommander(moveit_group_name)
    moveit_frame = moveit_robot.get_planning_frame()
    limb = intera_interface.Limb('right')

    # Init Robot
    robot = Sawyer(moveit_group)
    robot.reset()

    # Init World
    toyworld = ToyWorld(moveit_scene, moveit_frame, simulated=False)
    toyworld.initialize()
    
    get_task_srv = rospy.ServiceProxy('get_task', get_task)
    r = rospy.Rate(5)
    observations = []
    obs_file = open('obs.pickle', 'wb')
    
    while not rospy.is_shutdown():
        try:
            task = get_task_srv().task
            print("Task: {0}".format(task))
            obs = { 'timestamp': datetime.now(), 'task': task, **toyworld.get_observation(), **robot.get_observation() }
            observations.append(obs)
        except rospy.ServiceException as exc:
            print("Waiting for service")
        r.sleep()

    pickle.dump(observations, obs_file)
    

if __name__ == '__main__':
    run()
