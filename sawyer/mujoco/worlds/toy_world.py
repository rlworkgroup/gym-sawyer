import numpy as np
import os.path as osp

from gym.envs.robotics.utils import reset_mocap_welds, reset_mocap2body_xpos
from gym.spaces.box import Box
import numpy as np

from sawyer.garage.envs.mujoco.mujoco_env import MODEL_DIR
from sawyer.garage.misc.overrides import overrides
from sawyer.mujoco.worlds.base import WorldObject, World


class BoxWithLid(WorldObject):
    def __init__(self,
                 env,
                 initial_lid_pos,
                 name,
                 resource='objects/box_with_lid.xml'):
        """
        Box with Lid. See xml for more info.
        """
        self._env = env
        self._initial_lid_pos = initial_lid_pos
        self._name = name
        self._resource = resource

    @property
    def box_name(self):
        return '{}_base'.format(self.name)

    @property
    def lid_name(self):
        return '{}_lid'.format(self.name)

    @property
    def initial_pos(self):
        return self._initial_pos

    @property
    def random_delta_range(self):
        return self._random_delta_range

    @property
    def observation_space(self):
        return Box(np.inf, -np.inf, shape=(7,), dtype=np.float32)

    def get_observation(self):
        lid_xpos = self._env.sim.data.get_body_xpos(self.lid_name)
        lid_xquat = self._env.sim.data.get_body_xquat(self.lid_name)
        return {
            '{}_position'.format(self.lid_name): lid_xpos,
            '{}_rotation'.format(self.lid_name): lid_xquat,
        }

    def reset(self):
        self._env.sim.data.set_joint_qpos(
            '{}:joint'.format(self.lid_name), self._initial_lid_pos)


class BlockPeg(WorldObject):
    def __init__(self,
                 env,
                 initial_pos,
                 random_delta_range,
                 name,
                 resource='objects/block_peg.xml'):
        """
        Block attached to peg. See xml for more info.
        """
        self._env = env
        self._initial_pos = np.asarray(initial_pos)
        self._random_delta_range = np.asarray(random_delta_range)
        self._name = name
        self._resource = resource

        assert self._initial_pos.shape == (3,)
        assert self._random_delta_range.shape == (2,)

    @property
    def initial_pos(self):
        return self._initial_pos

    @property
    def random_delta_range(self):
        return self._random_delta_range

    @property
    def observation_space(self):
        return Box(np.inf, -np.inf, shape=(7,), dtype=np.float32)

    def get_observation(self):
        body_xpos = self._env.sim.data.get_body_xpos(self.name)
        body_xquat = self._env.sim.data.get_body_xquat(self.name)
        return {
            '{}_position'.format(self.name): body_xpos,
            '{}_rotation'.format(self.name): body_xquat,
        }

    def reset(self):
        body_xpos = (self.initial_pos +
            np.random.uniform(*self.random_delta_range, size=3))
        body_qpos = np.concatenate((body_xpos, [1, 0, 0, 0]), axis=0)
        self._env.sim.data.set_joint_qpos(
            '{}:joint'.format(self.name), body_qpos)

class ToyWorld(World):
    def __init__(self,
                 env,
                 xml_config,
                 box_lid_pos=0.,
                 peg_pos=[0.75, 0.1, 0.1],
                 peg_delta_range=[0., 0.]):
        """
        World containing a box with a lid, and a peg to open it.
        """
        self._env = env
        self._xml_config = xml_config
        self._box_lid_pos = box_lid_pos
        self._peg_pos = peg_pos
        self._peg_delta_range = peg_delta_range

        # Manually specify what objects exist for each xml config
        #TODO: add support for dm_control.mjcf
        self._boxes = []
        self._pegs = []

        if self._xml_config == 'default' or self._xml_config == 'task':
            box = BoxWithLid(
                env=self._env,
                initial_lid_pos=self._box_lid_pos,
                name='box')
            self._boxes.append(box)
            peg = BlockPeg(
                env=self._env,
                initial_pos=self._peg_pos,
                random_delta_range=self._peg_delta_range,
                name='peg')
            self._pegs.append(peg)

    def initialize(self):
        return

    def reset(self):
        for obj in self._boxes + self._pegs:
            obj.reset()
        return self.get_observation()

    def get_observation(self):
        obs = {}
        for obj in self._boxes + self._pegs:
            obs = {**obs, **obj.get_observation()}
        return obs

    @property
    def observation_space(self):
        spaces = []
        for obj in self._boxes + self._pegs:
            spaces.append(obj.observation_space)

        high = np.concatenate([sp.high for sp in spaces]).ravel()
        low = np.concatenate([sp.low for sp in spaces]).ravel()
        return Box(high, low, dtype=np.float32)
