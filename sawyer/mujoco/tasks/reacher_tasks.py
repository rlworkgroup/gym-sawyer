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
                 never_done=False,
                 success_thresh=0.01,
                 completion_bonus=0,
                 c_dist=0.25,
                 c_action=0.5):
        self._location = location
        self._never_done = never_done
        self._success_thresh = success_thresh
        self._completion_bonus = completion_bonus
        self._c_dist = c_dist
        self._c_action = c_action

    def compute_reward(self, obs, info):
        gripper_pos = info['gripper_position']
        action = info['action']

        r_dist = -np.linalg.norm(gripper_pos - self._location, axis=-1) * self._c_dist
        r_act = -np.linalg.norm(action) * self._c_action

        return r_dist + r_act

    def is_success(self, obs, info):
        if self._never_done:
            return False
        gripper_pos = info['gripper_position']
        d = np.linalg.norm(gripper_pos - self._location, axis=-1)
        return d < self._success_thresh

    @property
    def completion_bonus(self):
        return self._completion_bonus


class ReachWithGraspTask(ComposableTask):
    """
    Task to move robot gripper to a desired location while grasping an object.
    """
    def __init__(self,
                 location,
                 grasp_object,
                 never_done=False,
                 success_thresh=0.01,
                 completion_bonus=0.,
                 c_dist=0.25,
                 c_grasp=0.75,
                 c_action=0.5):
        self._location = location
        self._never_done = never_done
        self._grasp_object = grasp_object
        self._success_thresh = success_thresh
        self._completion_bonus = completion_bonus
        self._c_dist = c_dist
        self._c_grasp = c_grasp
        self._c_action = c_action

    def compute_reward(self, obs, info):
        gripper_pos = info['gripper_position']
        action = info['action']
        grasped = info['grasped_{}'.format(self._grasp_object)]

        r_dist = -np.linalg.norm(gripper_pos - self._location, axis=-1) * self._c_dist
        r_grasp = grasped * self._c_grasp
        r_act = -np.linalg.norm(action) * self._c_action

        return r_dist + r_grasp + r_act

    def is_success(self, obs, info):
        if self._never_done:
            return False
        gripper_pos = info['gripper_position']
        if self._grasp_object:
            grasped = info['grasped_{}'.format(self._grasp_object)]
        else:
            grasped = True
        d = np.linalg.norm(gripper_pos - self._location, axis=-1)
        return grasped and d < self._success_thresh

    @property
    def completion_bonus(self):
        return self._completion_bonus
