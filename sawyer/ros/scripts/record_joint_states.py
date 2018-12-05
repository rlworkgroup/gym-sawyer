import rospy
from sensor_msgs.msg import JointState

import numpy as np
import pdb


def trajectory_collector(num_trajectories):
    def joint_cb(data):
        if not do_collect:
            return

        rospy.loginfo(rospy.get_caller_id() + ": Sawyer Joint States\n" + 
            "[" + str(data.position) + "], " +
            "[" + str(data.velocity) + "], " +
            "[" + str(data.effort) + "]" )

        if len(data.name) == 9 and data.name[0] == 'head_pan':
            traj_data['seq'].append(data.header.seq)
            traj_data['secs'].append(data.header.stamp.secs)
            traj_data['nsecs'].append(data.header.stamp.nsecs)
            traj_data['pos'].append(list(data.position))
    

    do_collect = False
    task_data = []
    i = 0

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/robot/joint_states", JointState, joint_cb)

    while not rospy.is_shutdown() or i >= 5:
        # Prompt to start collecting task
        print("Press any key to start collecting trajectory %d.\n" % (i+1) +
              "Then any key to terminate trajectory collection.")
        raw_input()

        # Reinitialize data structure and set cb to start collecting
        traj_data = dict(
            seq=[],
            secs=[],
            nsecs=[],
            pos=[])
        do_collect = True
        raw_input()

        # Set cb to stop collecting. Prompt for termination.
        do_collect = False
        # Append trajectory to list
        for k,v in traj_data.items():
            traj_data[k] = np.array(v)

        task_data.append(traj_data)
        print("Trajectory %d collected. To stop collecting, press 's'. Press any other key to continue." % (i+1))
        if raw_input() == 's':
            break
        
        i += 1
    
    print("Collected %d tasks" % (i+1))
    np.savez('traj_data.npz', task_data=task_data)


if __name__ == "__main__":
    trajectory_collector(5)