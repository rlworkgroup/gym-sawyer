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

REACHERENV_COLLISION_WHITELIST = COLLISION_WHITELIST + [

]

class ReacherEnv(MujocoEnv, Serializable):
    def __init__(self,
                 goal_position,
                 robot=None,
                 xml_path='reacherenv_default.xml',
                 xml_config='default',                 
                 completion_bonus=0.,
                 action_scale=1.0,
                 collision_penalty=0.,
                 success_threshold=0.01,
                 reward_type=None,
                 terminate_on_collision=False,
                 randomize_start_jpos=False,                 
                 collision_whitelist=REACHERENV_COLLISION_WHITELIST,
                 **kwargs):

        self._desired_goal = goal_position
        self._xml_config = xml_config
        self._completion_bonus = completion_bonus
        self._action_scale = action_scale
        self._collision_penalty = collision_penalty
        self._success_threshold = success_threshold
        self._terminate_on_collision = terminate_on_collision
        self._reward_type = reward_type # Supported options: 'sparse' 

        if self._xml_config == 'default':
            self._robot = robot or PositionSpaceSawyer(
                    self, randomize_start_jpos=randomize_start_jpos)
        elif self._xml_config == 'task':
            self._robot = robot or TaskSpaceSawyer(
                self, randomize_start_jpos=randomize_start_jpos)
        else:
            warnings.warn("Unknown ReacherEnv xml_config: {}".format(xml_config))

        file_path = osp.join(MODEL_DIR, xml_path)
        super(ReacherEnv, self).__init__(file_path=file_path, **kwargs)

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
        Serializable.quick_init(self, locals())

    @property
    @overrides
    def action_space(self):
        return self._robot.action_space

    @property
    @overrides
    def observation_space(self):
        return self._robot.observation_space

    @overrides
    def reset(self):
        self._step = 0

        super(ReacherEnv, self).reset()
        self._robot.reset()

        return self.get_obs()

    def render(self, mode='human'):
        super(ReacherEnv, self).render(mode=mode)
        viewer = self.get_viewer()
        viewer.add_marker(pos=np.array(self._desired_goal), label="goal", size=0.01 * np.ones(3),)

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

        # Computing collision detection is expensive so cache the result
        in_collision = self.in_collision
        info = {
            'l': self._step,
            'in_collision': in_collision,
            'robot_obs': robot_obs,
            'gripper_position': self._robot.gripper_position,
            'gripper_state': self._robot.gripper_state
        }

        r = self.compute_reward(info)
        done = False
        successful = False

        if self.is_success(info):
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

        # Construct obs specified by observation_space
        obs = []
        if self._xml_config == 'default':
            obs.append(robot_obs['sawyer_joint_position'])
            obs.append(robot_obs['sawyer_gripper_state'])
            obs = np.concatenate(obs).ravel()
        elif self._xml_config == 'task':
            obs.append(robot_obs['sawyer_gripper_position'])
            obs.append(robot_obs['sawyer_gripper_state'])
            obs = np.concatenate(obs).ravel()
        else:
            warnings.warn("Don't know how to pull observation data for config:"
                " {}".format(self._xml_config))
        return obs

    def compute_reward(self, info):
        cur_position = info['gripper_position']        
        dist_to_goal = np.linalg.norm(cur_position - self._desired_goal, axis=-1)

        if self._reward_type == 'sparse':
            return (dist_to_goal < self._success_threshold).astype(np.float32)

        return -1 * dist_to_goal

    def is_success(self, info):
        cur_position = info['gripper_position']        
        dist_to_goal = np.linalg.norm(cur_position - self._desired_goal, axis=-1)
        return dist_to_goal <= self._success_threshold

    def done(self, info):
        return (self.is_success(info) or
            (self.in_collision and self._terminate_on_collision))

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
