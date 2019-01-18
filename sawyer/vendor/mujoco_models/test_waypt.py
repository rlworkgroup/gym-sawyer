from sawyer.mujoco.waypoint_sequencer import WaypointSequencer, toyenv_distance, toyenv_action
from sawyer.mujoco.envs import ToyEnv
import numpy as np
import time

WAYPOINTS = [
    [0.85, 0.065, 0.25, 1.],
    [0.85, 0.065, 0.03, 1.],
    [0.85, 0.065, 0.03, 0.],
    [0.85, 0.065, 0.25, 0.],
]

def forwardn(n):
    for _ in range(n):
        env.sim.forward()
        env.render()
        time.sleep(0.002)

def forward():
    while True:
        env.sim.forward()
        env.render()
        time.sleep(0.002)

env = ToyEnv(xml_path='toyenv_task.xml', xml_config='task')
waypoints = np.asarray(WAYPOINTS).reshape((-1, 4))
ws = WaypointSequencer(env, waypoints, toyenv_distance, toyenv_action)

forwardn(1000)
ws.run(True)
forwardn(1000)