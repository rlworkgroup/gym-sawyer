import numpy as np

from sawyer.mujoco.tasks.base import ComposableTask


class ReachTask(ComposableTask):
    """
    Task to move robot gripper to a desired location.

    Reward function is based on the following heuristics:
    - Positive reward for a small distance between goal and gripper position
    """
    def __init__(self,
                 location,
                 success_thresh=0.01,
                 completion_bonus=0,
                 c_dist=1):
        self._location = location
        self._success_thresh = success_thresh
        self._completion_bonus = completion_bonus
        self._c_dist = c_dist

    def compute_reward(self, obs, info):
        gripper_pos = info['gripper_position']
        d = np.linalg.norm(gripper_pos - self._location, axis=-1)
        return -self._c_dist * d

    def is_success(self, obs, info):
        gripper_pos = info['gripper_position']
        d = np.linalg.norm(gripper_pos - self._location, axis=-1)
        return d < self._success_thresh

    @property
    def completion_bonus(self):
        return self._completion_bonus
