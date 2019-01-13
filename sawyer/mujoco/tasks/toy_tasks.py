import numpy as np

from sawyer.mujoco.tasks.base import ComposableTask


class InsertTask(ComposableTask):
    """
    Task to insert a key object into an upward facing lock.

    The task assumes the key is already grasped and the gripper is close to the
    lock hole.

    Reward function is based on the following heuristics:
    - Positive reward for a smaller z coordinate of the key
    - Negative reward for releasing object
    """
    def __init__(self,
                 key_object,
                 lock_object,
                 success_thresh=0.01,
                 completion_bonus=0,
                 c_dist=1,
                 c_release=100):
        self._key_object = key_object
        self._lock_object = lock_object
        self._completion_bonus = completion_bonus
        self._c_dist = c_dist
        self._c_release = c_release

    def compute_reward(self, obs, info):
        key_pos = info['world_obs']['{}_position'.format(self._key_object)]
        lock_pos = info['world_obs']['{}_position'.format(self._lock_object)]
        released = not info['grasped_{}'.format(self._key_object)]

        key_height = key_pos[2] - lock_pos[2]  # Should always be positive
        return -self._c_dist * key_height - self._c_release * released

    def is_success(self, obs, info):
        key_pos = info['world_obs']['{}_position'.format(self._key_object)]
        lock_pos = info['world_obs']['{}_position'.format(self._lock_object)]
        grasped = info['grasped_{}'.format(self._key_object)]

        key_height = key_pos[2] - lock_pos[2]  # Should always be positive
        return grasped and key_height < self._success_thresh

    @property
    def completion_bonus(self):
        return self._completion_bonus


class RemoveTask(ComposableTask):
    """
    Task to remove a key object from an upward facing lock.

    The task assumes the key is already grasped and the gripper is close to the
    lock hole.

    Reward function is based on the following heuristics:
    - Positive reward for a larger z coordinate of the key
    - Negative reward for releasing object
    """
    def __init__(self,
                 key_object,
                 lock_object,
                 success_thresh=0.01,
                 completion_bonus=0,
                 c_dist=1,
                 c_release=100):
        self._key_object = key_object
        self._lock_object = lock_object
        self._completion_bonus = completion_bonus
        self._c_dist = c_dist
        self._c_release = c_release

    def compute_reward(self, obs, info):
        key_pos = info['world_obs']['{}_position'.format(self._key_object)]
        lock_pos = info['world_obs']['{}_position'.format(self._lock_object)]
        released = not info['grasped_{}'.format(self._key_object)]

        key_height = key_pos[2] - lock_pos[2]  # Should always be positive
        return (-self._c_dist * (self._success_thresh - key_height) -
                self._c_release * released)

    def is_success(self, obs, info):
        key_pos = info['world_obs']['{}_position'.format(self._key_object)]
        lock_pos = info['world_obs']['{}_position'.format(self._lock_object)]
        grasped = info['grasped_{}'.format(self._key_object)]

        key_height = key_pos[2] - lock_pos[2]  # Should always be positive
        return grasped and key_height > self._success_thresh

    @property
    def completion_bonus(self):
        return self._completion_bonus

class OpenTask(ComposableTask):
    """
    Task to open a toy box lid on a lateral sliding joint with an inserted peg.

    The task assumes there is already a key object inserted into the lid hole.

    Reward function is based on the following heuristics:
    - Positive reward for increased lateral distance between box and lid
    - Negative reward for releasing key object
    """
    def __init__(self,
                 box_object,
                 lid_object,
                 success_thresh=0.05,
                 completion_bonus=0,
                 c_dist=1,
                 c_release=100):
        self._box_object = box_object
        self._lid_object = lid_object
        self._success_thresh = success_thresh
        self._completion_bonus = completion_bonus
        self._c_dist = c_dist
        self._c_release = c_release

    def compute_reward(self, obs, info):
        lid_pos = info['world_obs']['{}_position'.format(self._lid_object)]
        box_pos = info['world_obs']['{}_position'.format(self._box_object)]
        released = not info['grasped_{}'.format(self._key_object)]

        lat_dist = np.linalg.norm(lid_pos[:2] - box_pos[:2])
        return (-self._c_dist * (self._success_thresh - lat_dist) -
                self._c_release * released)

    def is_success(self, obs, info):
        lid_pos = info['world_obs']['{}_position'.format(self._lid_object)]
        box_pos = info['world_obs']['{}_position'.format(self._box_object)]
        grasped = info['grasped_{}'.format(self._key_object)]

        lat_dist = np.linalg.norm(lid_pos[:2] - box_pos[:2])
        return grasped and lat_dist > self._success_thresh

    @property
    def completion_bonus(self):
        return self._completion_bonus


class CloseTask(ComposableTask):
    """
    Task to close a toy box lid with an inserted peg.

    The task assumes there is already a key object inserted into the lid hole.

    Reward function is based on the following heuristics:
    - Positive reward for decreased lateral distance between box and lid
    - Negative reward for releasing key object
    """
    def __init__(self,
                 box_object,
                 lid_object,
                 key_object,
                 success_thresh=0.05,
                 completion_bonus=0,
                 c_dist=1,
                 c_insert=0.1,
                 c_release=100):
        self._box_object = box_object
        self._lid_object = lid_object
        self._key_object = key_object
        self._success_thresh = success_thresh
        self._completion_bonus = completion_bonus
        self._c_dist = c_dist
        self._c_insert = c_insert
        self._c_release = c_release

    def compute_reward(self, obs, info):
        lid_pos = info['world_obs']['{}_position'.format(self._lid_object)]
        box_pos = info['world_obs']['{}_position'.format(self._box_object)]
        released = not info['grasped_{}'.format(self._key_object)]

        lat_dist = np.linalg.norm(lid_pos[:2] - box_pos[:2])
        return -self._c_dist * lat_dist - self._c_release * released

    def is_success(self, obs, info):
        lid_pos = info['world_obs']['{}_position'.format(self._lid_object)]
        box_pos = info['world_obs']['{}_position'.format(self._box_object)]
        grasped = info['grasped_{}'.format(self._key_object)]

        lat_dist = np.linalg.norm(lid_pos[:2] - box_pos[:2])
        return grasped and lat_dist < self._success_thresh

    @property
    def completion_bonus(self):
        return self._completion_bonus
