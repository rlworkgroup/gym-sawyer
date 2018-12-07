import rospy
from sensor_msgs.msg import JointState

import numpy as np
import pdb

def trajectory_collector(num_trajectories, save_file='traj_data.npz'):
    def joint_cb(data):
        try:
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
        except:
            pdb.set_trace()
    

    do_collect = False
    task_data = []
    i = 0

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/robot/joint_states", JointState, joint_cb)

    while not rospy.is_shutdown() and i < num_trajectories:
        # Prompt to start collecting task
        print("Press any key to start collecting trajectory %d.\n" % (i) +
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

        print("Trajectory %d collected. To end script, press 's'. To redo this trajectory, press 'r'. Press any other key to continue." % (i))
        inchar = raw_input()
        if inchar == 'r':
            continue

        task_data.append(traj_data)
        i += 1
        if inchar == 's':
            break
        
    
    print("Collected %d tasks, saved to %s" % (i, save_file))
    np.savez(save_file, task_data=task_data)


if __name__ == "__main__":
    num_trajectories = 10
    print("Will collect %d trajectories" % (num_trajectories))
    trajectory_collector(num_trajectories)