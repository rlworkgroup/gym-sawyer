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
                 target_z_pos=0.182,
                 completion_bonus=0,
                 c_insert=400,
                 c_grasp=100):
        self._key_object = key_object
        self._lock_object = lock_object
        self._never_done = never_done
        self._success_thresh = success_thresh
        self._target_z_pos = target_z_pos
        self._completion_bonus = completion_bonus
        self._c_insert = c_insert
        self._c_grasp = c_grasp
        self._init_dist = None
        self._target_pos = None

    def compute_reward(self, obs, info):
        key_pos = info['world_obs']['{}_position'.format(self._key_object)]
        lock_site = info['hole_site']
        grasped = info['grasped_{}'.format(self._key_object)]

        if self._target_pos is None:            
            self._target_pos = np.array([lock_site[0] + 0.008, lock_site[1], self._target_z_pos])

        if self._init_dist is None:
            self._init_dist = np.linalg.norm(self._target_pos - key_pos, axis=-1)

        r_insert = (self._init_dist - np.linalg.norm(self._target_pos - key_pos, axis=-1)) / self._init_dist
        
        return int(grasped) * (self._c_grasp + (self._c_insert * r_insert))

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
                 target_z_pos=0.22,
                 completion_bonus=0,
                 c_remove=400,
                 c_grasp=100):
        self._key_object = key_object
        self._lock_object = lock_object
        self._never_done = never_done
        self._success_thresh = success_thresh
        self._target_z_pos = target_z_pos
        self._completion_bonus = completion_bonus
        self._c_remove = c_remove
        self._c_grasp = c_grasp
        self._target_pos = None
        self._init_dist = None

    def compute_reward(self, obs, info):
        key_pos = info['world_obs']['{}_position'.format(self._key_object)]
        lock_site = info['hole_site']
        grasped = info['grasped_{}'.format(self._key_object)]

        if self._target_pos is None:            
            self._target_pos = np.array([lock_site[0] + 0.008, lock_site[1], self._target_z_pos])

        if self._init_dist is None:
            self._init_dist = np.linalg.norm(self._target_pos - key_pos, axis=-1)

        r_remove = (self._init_dist - np.linalg.norm(self._target_pos - key_pos, axis=-1)) / self._init_dist
        
        return int(grasped) * (self._c_grasp + (self._c_remove * r_remove))

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
                 open_lid_state=-0.055,
                 close_lid_state=-0.01,                 
                 completion_bonus=0,                 
                 c_open=50,
                 c_keeppeginhole=200,
                 c_grasp=50):
        self._lid_object = lid_object
        self._key_object = key_object
        self._never_done = never_done
        self._success_thresh = success_thresh
        self._open_lid_state = open_lid_state
        self._norm = max(abs(open_lid_state), abs(close_lid_state))
        self._completion_bonus = completion_bonus
        self._c_open = c_open
        self._c_keeppeginhole = c_keeppeginhole
        self._c_grasp = c_grasp

    def compute_reward(self, obs, info):
        key_pos = info['world_obs']['{}_position'.format(self._key_object)]
        lid_joint_state = info['lid_joint_state']
        lock_site = info['hole_site']
        grasped = info['grasped_{}'.format(self._key_object)]

        r_keep_peg_in_hole = -1 * np.linalg.norm(key_pos - lock_site) 
        r_open_lid = 1 - np.abs((self._open_lid_state - lid_joint_state) / self._norm)        

        return int(grasped) * (self._c_grasp + (r_open_lid * self._c_open)) + (r_keep_peg_in_hole * self._c_keeppeginhole)      

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
            lid_joint_state <= self._open_lid_state)

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
                 open_lid_state=-0.055,
                 close_lid_state=-0.01,                 
                 completion_bonus=0,
                 c_close=50,
                 c_keeppeginhole=200,
                 c_grasp=50):
        self._lid_object = lid_object
        self._key_object = key_object
        self._never_done = never_done
        self._success_thresh = success_thresh
        self._close_lid_state = close_lid_state
        self._norm = max(abs(open_lid_state), abs(close_lid_state))        
        self._completion_bonus = completion_bonus
        self._c_keeppeginhole = c_keeppeginhole
        self._c_close = c_close
        self._c_grasp = c_grasp

    def compute_reward(self, obs, info):
        key_pos = info['world_obs']['{}_position'.format(self._key_object)]
        lid_joint_state = info['lid_joint_state']
        lock_site = info['hole_site']
        grasped = info['grasped_{}'.format(self._key_object)]

        r_keep_peg_in_hole = -1 * np.linalg.norm(key_pos - lock_site) 
        r_close_lid = 1 - np.abs((self._close_lid_state - lid_joint_state) / self._norm)        

        return int(grasped) * (self._c_grasp + (r_close_lid * self._c_close)) + (r_keep_peg_in_hole * self._c_keeppeginhole)    

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
            lid_joint_state >= self._close_lid_state)

    @property
    def completion_bonus(self):
        return self._completion_bonus
