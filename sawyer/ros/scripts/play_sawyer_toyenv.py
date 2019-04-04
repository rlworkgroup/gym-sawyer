import joblib
import tensorflow as tf
import time
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from sawyer.ros.envs.sawyer import ToyEnv
from garage.envs import normalize
import rospy
import argparse
import warnings

warnings.filterwarnings("ignore")

init_joints = [None]
#Task2
init_joints.append({"right_j0": -0.0997240372, "right_j1": -0.610547676, "right_j2": -2.59113404, "right_j3": -0.905767104, "right_j4": -0.452365661, "right_j5": -1.37397618, "right_j6": -1.06285533})
#Task3
init_joints.append({"right_j0": -0.0901561516, "right_j1": -0.604187949, "right_j2": -2.60107727, "right_j3": -0.887399366, "right_j4": -0.446288342, "right_j5": -1.38250364, "right_j6": -1.06853579})
#Task4
init_joints.append({'right_j0': -0.372912109375, 'right_j1': -1.19783984375, 'right_j2': -3.025119140625, 'right_j3': -1.4749130859375, 'right_j4': -0.0459287109375, 'right_j5': -1.288650390625, 'right_j6': -1.6450283203125})
#Task5
init_joints.append({"right_j0": -0.314972785, "right_j1": -1.03170101, "right_j2": -3.12122423, "right_j3": -1.62847136, "right_j4": -0.0107168662, "right_j5": -0.976596656, "right_j6": -1.6864605})
#Task6
init_joints.append({"right_j0": -0.356721665, "right_j1": -0.878626453, "right_j2": -2.98844738, "right_j3": -1.40505256, "right_j4": -0.113361042, "right_j5": -1.05316158, "right_j6": -1.57639082})
#Task7
init_joints.append({"right_j0": -0.314972785, "right_j1": -1.03170101, "right_j2": -3.12122423, "right_j3": -1.62847136, "right_j4": -0.0107168662, "right_j5": -0.976596656, "right_j6": -1.6864605})

def print_obs(obs, label):
    print("######### {0} ##########".format(label))
    gripper_pos = obs[:3]
    gripper_state = obs[3]
    box_pos = obs[4:7]
    box_ori = obs[7:11]
    lid_pos = obs[11:14]
    lid_ori = obs[14:18]
    peg_pos = obs[18:21]
    peg_ori = obs[21:]
    str_obs = "gripper pos: {0}\ngripper_state: {1}\nbox_pos: {2}\nbox_ori: {3}\nlid_pos: {4}\nlid_ori: {5}\npeg_pos: {6}\npeg_ori: {7}".format(
                gripper_pos, gripper_state, box_pos, box_ori, lid_pos, lid_ori, peg_pos, peg_ori)
    print(str_obs)


def play(v):
    with tf.Session():
        snapshot = joblib.load(v.pkl_file)
        policy = snapshot['policy']
        if v.robot:
            rospy.init_node('test')
            task = int(v.pkl_file[-5])
            env = TfEnv(normalize(ToyEnv(initial_joints=init_joints[task-1])))
            menv = snapshot['env']
        else:
            env = snapshot['env']

        for rollout in range(v.rollouts):
            policy.reset()
            obs = env.reset()
            mobs = menv.reset()
            i = 0
            for step in range(v.max_rollout_length):
                obs[14] = -obs[14]
                obs[17] = -obs[17]
                #obs[21:] = [1., 0., 0., 0.] #Peg ori
                #obs[4:7] = [0.756, -0.042, 0.002] #Box pos
                #obs[7:11] = [0.707, 0., 0., 0.707] #Box ori
                #obs[11:14] = [0.75599857, -0.035, 0.1278] #Lid Pos
                #obs[14:18] = [0.707, 0., 0., 0.707] #Lid ori
                #print_obs(mobs, "Sim")
                #print_obs(obs, "Robot")
                #break
                act, dist_info = policy.get_action(obs)
                obs, rew, done, info = env.step(act)
                time.sleep(v.dt)
                if v.render:
                    env.render()
                i += 1
                print("STEP: " + str(i))

                if done:
                    continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='pkl_file', action='store', required=True)
    parser.add_argument('-dt', dest='dt', action='store', default=0.5)
    parser.add_argument('-r', dest='rollouts', type=int, action='store', default=1)
    parser.add_argument('-l', dest='max_rollout_length', type=int, action='store', default=30)
    parser.add_argument('--robot', dest='robot', action='store_true', default=False)
    parser.add_argument('--render', dest='render', action='store_true', default=False)
    
    args = parser.parse_args()
    play(args)
