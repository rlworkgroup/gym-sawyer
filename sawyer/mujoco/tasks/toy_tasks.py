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
                 never_done=False,
                 success_thresh=0.01,
                 target_z_pos=0.20,
                 completion_bonus=0,
                 c_dist=0.1,
                 c_grasp=0.9):
        self._key_object = key_object
        self._lock_object = lock_object
        self._never_done = never_done
        self._success_thresh = success_thresh
        self._target_z_pos = target_z_pos
        self._completion_bonus = completion_bonus
        self._c_dist = c_dist
        self._c_grasp = c_grasp

    def compute_reward(self, obs, info):
        key_pos = info['world_obs']['{}_position'.format(self._key_object)]
        lock_site = info['hole_site']
        grasped = info['grasped_{}'.format(self._key_object)]

        target_pos = np.array([lock_site[0], lock_site[1], self._target_z_pos])

        r_dist = -np.linalg.norm(target_pos - key_pos)
        r_grasp = grasped * self._c_grasp
        return r_dist + r_grasp

    def is_success(self, obs, info):
        if self._never_done:
            return False
        key_pos = info['world_obs']['{}_position'.format(self._key_object)]
        lock_site = info['hole_site']
        grasped = info['grasped_{}'.format(self._key_object)]

        target_pos = np.array([lock_site[0], lock_site[1], self._target_z_pos])

        r_dist = np.linalg.norm(target_pos - key_pos)
        return (grasped and
            np.linalg.norm(target_pos - key_pos) < self._success_thresh)

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
                 never_done=False,
                 success_thresh=0.01,
                 target_z_pos=0.35,
                 completion_bonus=0,
                 c_dist=0.1,
                 c_grasp=0.9):
        self._key_object = key_object
        self._lock_object = lock_object
        self._never_done = never_done
        self._success_thresh = success_thresh
        self._target_z_pos = target_z_pos
        self._completion_bonus = completion_bonus
        self._c_dist = c_dist
        self._c_grasp = c_grasp

    def compute_reward(self, obs, info):
        key_pos = info['world_obs']['{}_position'.format(self._key_object)]
        lock_site = info['hole_site']
        grasped = info['grasped_{}'.format(self._key_object)]

        target_pos = np.array([lock_site[0], lock_site[1], self._target_z_pos])

        r_dist = -np.linalg.norm(target_pos - key_pos)
        r_grasp = grasped * self._c_grasp
        return r_dist + r_grasp

    def is_success(self, obs, info):
        if self._never_done:
            return False
        key_pos = info['world_obs']['{}_position'.format(self._key_object)]
        lock_site = info['hole_site']
        grasped = info['grasped_{}'.format(self._key_object)]

        target_pos = np.array([lock_site[0], lock_site[1], self._target_z_pos])

        r_dist = np.linalg.norm(target_pos - key_pos)
        return (grasped and
            np.linalg.norm(target_pos - key_pos) < self._success_thresh)

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
                 lid_object,
                 key_object,
                 never_done=False,
                 success_thresh=0.01,
                 target_lid_jpos=-0.05,
                 completion_bonus=0,
                 c_jdist=0.2,
                 c_xydist=0.8):
        self._lid_object = lid_object
        self._key_object = key_object
        self._never_done = never_done
        self._success_thresh = success_thresh
        self._target_lid_jpos = target_lid_jpos
        self._completion_bonus = completion_bonus
        self._c_jdist = c_jdist
        self._c_xydist = c_xydist

    def compute_reward(self, obs, info):
        key_pos = info['world_obs']['{}_position'.format(self._key_object)]
        lid_joint_state = info['lid_joint_state']
        lock_site = info['hole_site']
        grasped = info['grasped_{}'.format(self._key_object)]

        dxy_peg2hole = key_pos[:2] - lock_site[:2]

        r_jdist = (1 - np.tanh(10. * np.abs(lid_joint_state - self._target_lid_jpos))) * 0.2
        r_peg2hole = (1 - np.tanh(np.linalg.norm(dxy_peg2hole))) * self._c_xydist
        return int(grasped) * (r_jdist + r_peg2hole)

    def is_success(self, obs, info):
        if self._never_done:
            return False
        key_pos = info['world_obs']['{}_position'.format(self._key_object)]
        lid_joint_state = info['lid_joint_state']
        lock_site = info['hole_site']
        grasped = info['grasped_{}'.format(self._key_object)]

        dxy_peg2hole = key_pos[:2] - lock_site[:2]

        return (grasped and
            np.linalg.norm(dxy_peg2hole) < self._success_thresh and
            lid_joint_state <= self._target_lid_jpos)

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
                 lid_object,
                 key_object,
                 never_done=False,
                 success_thresh=0.01,
                 target_lid_jpos=-0.05,
                 completion_bonus=0,
                 c_jdist=0.2,
                 c_xydist=0.8):
        self._lid_object = lid_object
        self._key_object = key_object
        self._never_done = never_done
        self._success_thresh = success_thresh
        self._target_lid_jpos = target_lid_jpos
        self._completion_bonus = completion_bonus
        self._c_jdist = c_jdist
        self._c_xydist = c_xydist

    def compute_reward(self, obs, info):
        key_pos = info['world_obs']['{}_position'.format(self._key_object)]
        lid_joint_state = info['lid_joint_state']
        lock_site = info['hole_site']
        grasped = info['grasped_{}'.format(self._key_object)]

        dxy_peg2hole = key_pos[:2] - lock_site[:2]

        r_jdist = (1 - np.tanh(10. * np.abs(lid_joint_state - self._target_lid_jpos))) * self._c_jdist
        r_peg2hole = (1 - np.tanh(np.linalg.norm(dxy_peg2hole))) * self._c_xydist
        return int(grasped) * (r_jdist + r_peg2hole)

    def is_success(self, obs, info):
        if self._never_done:
            return False
        key_pos = info['world_obs']['{}_position'.format(self._key_object)]
        lid_joint_state = info['lid_joint_state']
        lock_site = info['hole_site']
        grasped = info['grasped_{}'.format(self._key_object)]

        dxy_peg2hole = key_pos[:2] - lock_site[:2]

        return (grasped and
            np.linalg.norm(dxy_peg2hole) < self._success_thresh and
            lid_joint_state >= self._target_lid_jpos)

    @property
    def completion_bonus(self):
        return self._completion_bonus
