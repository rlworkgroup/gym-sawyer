from sawyer.mujoco.waypoint_sequencer import WaypointSequencer, toyenv_distance, toyenv_action, toyenv_eval
from sawyer.mujoco.envs import ToyEnv
import numpy as np
import time

WAYPOINTS = [
    [0.82, 0.265, 0.20, 1.],
    [0.82, 0.265, 0.14, 1.],
    [0.82, 0.265, 0.14, -1.],
    [0.82, 0.265, 0.20, -1.],
    [0.66, 0.0, 0.25, -1.],
    [0.66, 0.0, 0.15, -1.],
    [0.75, 0.0, 0.15, -1.],
    [0.66, 0.0, 0.15, -1.],
    [0.66, 0.0, 0.25, -1.],
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

env = ToyEnv(xml_path='toyenv_task.xml')
waypoints = np.asarray(WAYPOINTS).reshape((-1, 4))
ws = WaypointSequencer(env, waypoints, toyenv_distance, toyenv_action, toyenv_eval)

forwardn(500)
for _ in range(5):
    ws.reset()
    ws.run(render=True)
forwardn(1000)
