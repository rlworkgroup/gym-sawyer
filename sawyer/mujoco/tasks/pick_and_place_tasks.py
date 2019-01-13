import numpy as np

from sawyer.mujoco.tasks.base import ComposableTask


class PickTask(ComposableTask):
    """
    Task to pick up an object with the robot gripper.

    Reward function is based on the following heuristics:
    - Positive reward for moving gripper closer to object
    - Positive reward for grasping object
    """
    def __init__(self,
                 pick_object,
                 success_thresh=0.01,
                 completion_bonus=0,
                 c_dist=1,
                 c_grasp=100):
        self._pick_object = pick_object
        self._success_thresh = success_thresh
        self._completion_bonus = completion_bonus
        self._c_dist = c_dist
        self._c_grasp = c_grasp

    def compute_reward(self, obs, info):
        gripper_pos = info['gripper_position']
        obj_pos = info['world_obs']['{}_position'.format(self._pick_object)]
        grasped = info['grasped_{}'.format(self._pick_object)]
        d = np.linalg.norm(gripper_pos - obj_pos, axis=-1)

        return -self._c_dist * d + self._c_grasp * grasped

    def is_success(self, obs, info):
        gripper_pos = info['gripper_position']
        obj_pos = info['world_obs']['{}_position'.format(self._pick_object)]
        grasped = info['grasped_{}'.format(self._pick_object)]
        d = np.linalg.norm(gripper_pos - obj_pos, axis=-1)

        return grasped and d < self._success_thresh

    @property
    def completion_bonus(self):
        return self._completion_bonus


class PlaceTask(ComposableTask):
    """
    Task to place object at a desired location.

    Reward function is based on the following heuristics:
    - Positive reward for moving gripper closer to goal
    - Negative reward for releasing object before reaching the goal position
    - Positive reward for releasing object near goal position
    """
    def __init__(self,
                 place_object,
                 location,
                 success_thresh=0.01,
                 completion_bonus=0,
                 c_dist=1,
                 c_early_release=100,
                 c_release=10):
        self._place_object = place_object
        self._location = location
        self._success_thresh = success_thresh
        self._completion_bonus = completion_bonus
        self._c_dist = c_dist
        self._c_early_release = c_early_release
        self._c_release = c_release

    def compute_reward(self, obs, info):
        obj_pos = info['world_obs']['{}_position'.format(self._place_object)]
        released = not info['grasped_{}'.format(self._place_object)]
        d = np.linalg.norm(obj_pos - self._location, axis=-1)

        if d < self._success_thresh:
            return -self._c_dist * d + self._c_release * released
        else:
            return -self._c_dist * d - self._c_early_release * released

    def is_success(self, obs, info):
        obj_pos = info['world_obs']['{}_position'.format(self._place_object)]
        released = not info['grasped_{}'.format(self._place_object)]
        d = np.linalg.norm(obj_pos - self._location, axis=-1)

        return released and d < self._success_thresh

    @property
    def completion_bonus(self):
        return self._completion_bonus
