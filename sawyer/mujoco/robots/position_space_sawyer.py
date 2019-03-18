import numpy as np

from gym.spaces.box import Box

from sawyer.garage.misc.overrides import overrides
from sawyer.mujoco.robots import Sawyer

class PositionSpaceSawyer(Sawyer):
    def __init__(self,
                 env,
                 action_low=np.full(9, -0.02),
                 action_high=np.full(9, 0.02),
                 **kwargs):
        self._action_low = action_low
        self._action_high = action_high

        super(PositionSpaceSawyer, self).__init__(env, **kwargs)

    @overrides
    def initialize(self):
        # Position space sawyer should not have any mocaps to initialize
        return

    @overrides
    def get_observation(self):
        return {
            'control_scheme': 'position',
            'sawyer_joint_position': self.joint_positions[2:],
            'sawyer_gripper_state': np.array([self.gripper_state])
        }

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
            shape=(8,),
            dtype=np.float32)

    def step(self, action):
        # Does not update the MuJoCo env entirely, just the Robot state
        curr_pos = self.joint_positions

        next_pos = np.clip(
            action + curr_pos[:],
            self.joint_position_space.low[:],
            self.joint_position_space.high[:]
        )
        for _ in range(4):
            self._env.sim.data.ctrl[:] = next_pos
            self._env.sim.forward()
            self._env.sim.step()
