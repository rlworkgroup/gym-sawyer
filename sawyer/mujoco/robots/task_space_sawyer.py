import numpy as np

from gym.envs.robotics.utils import reset_mocap_welds, reset_mocap2body_xpos
from gym.spaces.box import Box

from sawyer.garage.misc.overrides import overrides
from sawyer.mujoco.robots import Sawyer

class TaskSpaceSawyer(Sawyer):
    def __init__(self,
                 env,
                 action_low=np.array([-0.01, -0.01, -0.01, -1.]),
                 action_high=np.array([0.01, 0.01, 0.01, 1.]),
                 mujoco_steps=5,
                 **kwargs):
        self._action_low = action_low
        self._action_high = action_high
        self._mujoco_steps = mujoco_steps

        super(TaskSpaceSawyer, self).__init__(env, **kwargs)

    @overrides
    def initialize(self):
        reset_mocap_welds(self._env.sim)
        self._env.sim.forward()

    @overrides
    def get_observation(self):
        return {
            'control_scheme': 'task',
            'sawyer_gripper_position': self.gripper_position,
            'sawyer_gripper_state': np.array([self.gripper_state]),
        }

    @overrides
    def reset(self):
        self.joint_positions = self.INITIAL_JOINT_STATE
        self._env.sim.data.mocap_quat[0, :] = np.array([0, 0, 1, 0])
        self._env.sim.data.set_mocap_pos('mocap', self.INITIAL_MOCAP_POS)
        self._env.sim.forward()
        for _ in range(100):
            self._env.sim.step()

        return self.get_observation()

    @property
    @overrides
    def action_space(self):
        return Box(
            low=self._action_low, high=self._action_high, dtype=np.float32)

    @property
    @overrides
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32)

    def step(self, action):
        # Does not update the MuJoCo env entirely, just the Robot state
        reset_mocap2body_xpos(self._env.sim)
        self._env.sim.data.mocap_pos[0, :3] = self._env.sim.data.mocap_pos[0, :3] + action[:3]
        self._env.sim.data.mocap_quat[0, :] = np.array([0, 0, 1, 0])
        self.gripper_state = 1 if action[3] > 0 else -1
        for _ in range(self._mujoco_steps):
            try:
                self._env.sim.step()
            except Exception:
                pass
        self._env.sim.forward()
