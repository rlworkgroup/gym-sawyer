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

def play(v):
    with tf.Session():
        snapshot = joblib.load(v.pkl_file)
        policy = snapshot['policy']
        if v.robot:
            rospy.init_node('test')
            env = TfEnv(normalize(ToyEnv()))
        else:
            env = snapshot['env']

        for rollout in range(v.rollouts):
            policy.reset()
            obs = env.reset()
            i = 0
            for step in range(v.max_rollout_length):
                obs[14] = -obs[14]
                obs[17] = -obs[17]
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
    parser.add_argument('-r', dest='rollouts', action='store', default=1)
    parser.add_argument('-l', dest='max_rollout_length', action='store', default=30)
    parser.add_argument('--robot', dest='robot', action='store_true', default=False)
    parser.add_argument('--render', dest='render', action='store_true', default=False)
    
    args = parser.parse_args()
    play(args)
