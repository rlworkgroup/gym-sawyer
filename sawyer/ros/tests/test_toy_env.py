import rospy
import numpy as np
from sawyer.ros.envs.sawyer import ToyEnv

rospy.init_node('test_toy_env')
toy_env = ToyEnv(simulated=False)

action = np.array([-0.79328125, 0.4875126953125, -2.848359375, 0.9184482421875, -2.932041015625, -0.0460625, 0.0387646484375, 0])

i = 0
while i < 2:
    obs, r, done, info = toy_env.step(action)
    i += 1

toy_env.reset()
