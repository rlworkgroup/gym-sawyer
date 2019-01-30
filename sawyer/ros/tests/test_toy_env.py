import rospy
import numpy as np
from sawyer.ros.envs.sawyer import ToyEnv

rospy.init_node('test_toy_env')
toy_env = ToyEnv(simulated=False, control_mode='task_space')

action = np.array([0.1, 0, 0, 0])

i = 0
while i < 2:
    obs, r, done, info = toy_env.step(action)
    print(obs) 
    print(r)
    print(done)
    print(info)
    i += 1

toy_env.reset()
