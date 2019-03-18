import numpy as np
import os.path as osp
import warnings

from gym.spaces.box import Box

from sawyer.garage.core import Serializable
from sawyer.garage.envs.mujoco import MujocoEnv
from sawyer.garage.envs.mujoco.mujoco_env import MODEL_DIR
from sawyer.garage.misc.overrides import overrides
from sawyer.mujoco.robots import PositionSpaceSawyer, TaskSpaceSawyer
from sawyer.mujoco.robots.sawyer import COLLISION_WHITELIST
from sawyer.mujoco.tasks import (ReachTask, PickTask, PlaceTask, InsertTask,
                                 RemoveTask, OpenTask, CloseTask)
from sawyer.mujoco.worlds import ToyWorld


TOYENV_COLLISION_WHITELIST = COLLISION_WHITELIST + [
    # If you are okay with the grippers touching the lid
    # ("r_gripper_l_finger_tip", "box_lid"),
    # ("r_gripper_r_finger_tip", "box_lid"),

    # Need to whitelist objects with table
    ("pedestal_table", "box_base"),
    ("pedestal_table", "peg"),

    # Need to whitelist peg with box
    ("box_base", "peg"),
    ("box_lid", "peg"),
]


class ToyEnv(MujocoEnv, Serializable):
    # Env for working with a toy box and sawyer
    # ToyEnv is composed of multiple subtasks.
    def __init__(self,
                 robot=None,
                 world=None,
                 xml_path='toyenv_default.xml',
                 xml_config='default',
                 task_list=None,
                 completion_bonus=0.,
                 action_scale=1.0,
                 collision_penalty=0.,
                 terminate_on_collision=False,
                 randomize_start_jpos=False,
                 collision_whitelist=TOYENV_COLLISION_WHITELIST,
                 **kwargs):

        warnings.warn(
            "ToyEnv is under development and has not been tested thoroughly. "
            "We make no guarantees that the env is solvable.")

        self._xml_config = xml_config
        self._completion_bonus = completion_bonus
        self._action_scale = action_scale
        self._collision_penalty = collision_penalty
        self._terminate_on_collision = terminate_on_collision
        self._randomize_start_jpos = randomize_start_jpos

        #TODO: Dynamically specify the Robot and World with dm_control.mjcf
        # default config: Position control + 1 box w/ lid and 1 peg
        if self._xml_config == 'default':
            self._robot = robot or PositionSpaceSawyer(
                    self, randomize_start_jpos=randomize_start_jpos)
            self._world = world or ToyWorld(self, xml_config=xml_config)
        elif self._xml_config == 'task':
            self._robot = robot or TaskSpaceSawyer(
                self, randomize_start_jpos=randomize_start_jpos)
            self._world = world or ToyWorld(self, xml_config=xml_config)
        else:
            warnings.warn("Unknown ToyEnv xml_config: {}".format(xml_config))

        # Populate task list
        if task_list:
            self._task_list = task_list
        else:
            tasks = [
                PickTask,    # Pick up peg
                ReachTask,   # Move peg above box
                InsertTask,  # Insert peg into hole
                OpenTask,    # Open box lid
                RemoveTask,  # Remove peg from hole
                PlaceTask,   # Place peg back
                # PickTask,    # Pick up block
                # PlaceTask,   # Place block into box
                # PickTask,    # Pick up peg
                # ReachTask,   # Move peg above box
                # InsertTask,  # Insert peg into hole
                # CloseTask,   # Close box lid
                # PlaceTask,   # Place peg back
                ]
            task_args = [
                {'pick_object': 'peg'},
                {'location': [0.65, 0., 0.]},
                {'key_object': 'peg', 'lock_object': 'box_lid'},
                {'box_object': 'box_base', 'lid_object': 'box_lid'},
                {'key_object': 'peg', 'lock_object': 'box_lid'},
                {'place_object': 'peg', 'location': [0.65, 0., 0.]},
            ]
            self._task_list = [task(**kwa) for task, kwa in
                               zip(tasks, task_args)]
        self._active_task = self._task_list[0]

        file_path = osp.join(MODEL_DIR, xml_path)
        super(ToyEnv, self).__init__(file_path=file_path, **kwargs)

        # Populate and id-based whitelist of acceptable body ID contacts
        self._collision_whitelist = []
        for c in collision_whitelist:
            # Hedge our bets by allowing both orderings
            self._collision_whitelist.append((
                self.sim.model.body_name2id(c[0]),
                self.sim.model.body_name2id(c[1])
            ))
            self._collision_whitelist.append((
                self.sim.model.body_name2id(c[1]),
                self.sim.model.body_name2id(c[0])
            ))

        self._robot.initialize()
        self._world.initialize()
        Serializable.quick_init(self, locals())


    @property
    @overrides
    def action_space(self):
        return self._robot.action_space

    @property
    @overrides
    def observation_space(self):
        spaces = []
        spaces.append(self._world.observation_space)
        spaces.append(self._robot.observation_space)

        high = np.concatenate([sp.high for sp in spaces]).ravel()
        low = np.concatenate([sp.low for sp in spaces]).ravel()
        return Box(high, low, dtype=np.float32)

    @overrides
    def reset(self):
        self._step = 0
        self._active_task = self._task_list[0]

        super(ToyEnv, self).reset()
        self._robot.reset()
        self._world.reset()

        return self.get_obs()

    def render(self, mode='human'):
        #TODO: Add markers and stuff, should be handled by Robot and World
        super(ToyEnv, self).render(mode=mode)

    @overrides
    def step(self, action):
        assert action.shape == self.action_space.shape

        # NOTE: you MUST copy the action if you modify it
        a = action.copy()

        # Clip to action space
        a *= self._action_scale
        a = np.clip(a, self.action_space.low, self.action_space.high)

        self._robot.step(a)
        self._step += 1
        obs = self.get_obs()

        # Robot obs
        robot_obs = self._robot.get_observation()

        # World obs
        world_obs = self._world.get_observation()

        # Grasp state obs
        grasped_peg_obs = self.has_object('peg:head')
        # has_block_obs = self.has_object('block:body')

        # Computing collision detection is expensive so cache the result
        in_collision = self.in_collision
        info = {
            'l': self._step,
            'in_collision': in_collision,
            'robot_obs': robot_obs,
            'world_obs': world_obs,
            'gripper_position': self._robot.gripper_position,
            'gripper_state': self._robot.gripper_state,
            'grasped_peg': grasped_peg_obs,
            # 'grasped_block': grasped_block_obs,
        }

        r = self.compute_reward(obs, info)
        done = False
        successful = False

        if self._active_task.is_success(obs, info):
            r += self._active_task.completion_bonus
            done = self.next_task()

        if self.is_success(obs, info):
            r += self._completion_bonus
            done = True
            successful = True

        if in_collision:
            r -= self._collision_penalty
            if self._terminate_on_collision:
                done = True
                successful = False

        info['r'] = r
        info['d'] = done
        info['is_success'] = successful

        return obs, r, done, info

    def get_obs(self):
        # Robot obs
        robot_obs = self._robot.get_observation()

        # World obs
        world_obs = self._world.get_observation()

        # Construct obs specified by observation_space
        obs = []
        if self._xml_config == 'default':
            obs.append(robot_obs['sawyer_joint_position'])
            obs.append(world_obs['box_lid_position'])
            obs.append(world_obs['box_lid_rotation'])
            obs.append(world_obs['peg_position'])
            obs.append(world_obs['peg_rotation'])
            obs = np.concatenate(obs).ravel()
        elif self._xml_config == 'task':
            obs.append(robot_obs['sawyer_gripper_position'])
            obs.append(world_obs['box_lid_position'])
            obs.append(world_obs['box_lid_rotation'])
            obs.append(world_obs['peg_position'])
            obs.append(world_obs['peg_rotation'])
            obs = np.concatenate(obs).ravel()
        else:
            warnings.warn("Don't know how to pull observation data for config:"
                " {}".format(self._xml_config))
        return obs

    def compute_reward(self, obs, info):
        #TODO: The env could partially calculate the reward as well
        return self._active_task.compute_reward(obs, info)

    def is_success(self, obs, info):
        return (self._active_task == self._task_list[-1] and
            self._active_task.is_success(obs, info))

    def done(self, obs, info):
        return (self.is_success(obs, info) or
            (self.in_collision and self._terminate_on_collision))

    def has_object(self, obj_name):
        contacts = tuple()
        for coni in range(self.sim.data.ncon):
            con = self.sim.data.contact[coni]
            contacts += ((con.geom1, con.geom2), )
        finger_id_1 = self.sim.model.geom_name2id('finger_tip_1')
        finger_id_2 = self.sim.model.geom_name2id('finger_tip_2')
        object_id = self.sim.model.geom_name2id(obj_name)
        if ((finger_id_1, object_id) in contacts or
            (object_id, finger_id_1) in contacts) and (
                (finger_id_2, object_id) in contacts or
                (object_id, finger_id_2) in contacts):
            return True
        else:
            return False

    def next_task(self):
        # Set up env to for next task in sequence
        active_task_idx = self._task_list.index(self._active_task)
        if active_task_idx + 1 == len(self._task_list):
            return True  # Done with all tasks
        else:
            self._active_task = self._task_list[active_task_idx + 1]
            return False

    @property
    def in_collision(self):
        for c in self._get_collisions():
           if c not in self._collision_whitelist:
               return True
        return False

    def _get_collision_names(self, whitelist=True):
        contacts = []
        for c in self._get_collisions():
            if c not in self._collision_whitelist or not whitelist:
                contacts.append((
                    self.sim.model.body_id2name(c[0]),
                    self.sim.model.body_id2name(c[1])
                ))
        return contacts

    def _get_collisions(self):
        for c in self.sim.data.contact[:self.sim.data.ncon]:
            if c.geom1 != 0 and c.geom2 !=0:
                yield (
                    self.sim.model.geom_bodyid[c.geom1],
                    self.sim.model.geom_bodyid[c.geom2]
                )
