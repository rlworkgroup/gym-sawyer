"""Ask robot to open box lid."""

import copy
import sys

from geometry_msgs.msg import Pose, Point
import intera_interface
import rospy
from sawyer.ros.worlds.toy_world import ToyWorld
from datetime import datetime
import pickle
from get_task_srv.srv import get_task
import numpy as np

def run():
    rospy.init_node('collect_toy_data', anonymous=True)

    limb = intera_interface.Limb('right')
    # Init World
    toyworld = ToyWorld(None, None, simulated=False)
    toyworld.initialize()
    
    get_task_srv = rospy.ServiceProxy('get_task', get_task)
    r = rospy.Rate(5)
    observations = []
    obs_file = open('obs.pickle', 'wb')
    
    while not rospy.is_shutdown():
        try:
            task = get_task_srv().task
            print("Task: {0}".format(task))

            robot_obs = {
                'gripper_position': np.array(limb.endpoint_pose()['position']),
                'gripper_orientation': np.array(limb.endpoint_pose()['orientation']),
                'gripper_lvel': np.array(limb.endpoint_velocity()['linear']),
                'gripper_avel': np.array(limb.endpoint_velocity()['angular']),
                'gripper_force': np.array(limb.endpoint_effort()['force']),
                'gripper_torque': np.array(limb.endpoint_effort()['torque']),
                'robot_joint_angles': np.array(list(limb.joint_angles().values())),
                'robot_joint_velocities': np.array(list(limb.joint_velocities().values())),
                'robot_joint_efforts': np.array(list(limb.joint_efforts().values()))
            }
            obs = { 'timestamp': datetime.now(), 'task': task, **toyworld.get_observation(), **robot_obs }
            observations.append(obs)
        except rospy.ServiceException as exc:
            print("Waiting for service")
        r.sleep()

    pickle.dump(observations, obs_file)
    

if __name__ == '__main__':
    run()
