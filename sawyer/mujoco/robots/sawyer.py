import numpy as np

from gym.envs.robotics.utils import reset_mocap_welds, reset_mocap2body_xpos
from gym.spaces.box import Box

from sawyer.garage.misc.overrides import overrides
from sawyer.mujoco.robots.base import Robot

COLLISION_WHITELIST = [
    # Liberal whitelist here
    # Remove this section for a more conservative policy

    # The head seems to have a large collision body
    ("head", "right_l0"),
    ("head", "right_l1"),
    ("head", "right_l1_2"),
    ("head", "right_l2"),
    ("head", "right_l2_2"),
    ("head", "right_l3"),
    ("head", "right_l4"),
    ("head", "right_l4_2"),
    ("head", "right_l5"),
    ("head", "right_l6"),
    ("head", "right_gripper_base"),
    ("head", "r_gripper_l_finger_tip"),
    ("head", "r_gripper_r_finger"),
    ("head", "r_gripper_r_finger_tip"),
    ("head", "r_gripper_l_finger"),

    # Close but fine
    ("right_l0", "right_l4_2"),
    ("right_l4_2", "right_l1_2"),
    ("right_l2_2", "pedestal_table"),

    # Trivially okay below this line
    ("r_gripper_l_finger_tip", "r_gripper_r_finger_tip"),
    ("r_gripper_l_finger_tip", "r_gripper_r_finger"),
    ("r_gripper_r_finger_tip", "r_gripper_l_finger"),
]

class Sawyer(Robot):
    GRIPPER_STATE_SCALE = 0.020833
    INITIAL_JOINT_STATE = np.array([0., 0.,             # grippers
                                    -0.140923828125,    # j0
                                    -1.2789248046875,   # j1
                                    -3.043166015625,    # j2
                                    -2.139623046875,    # j3
                                    -0.047607421875,    # j4
                                    -0.7052822265625,   # j5
                                    -1.4102060546875])  # j6
    INITIAL_MOCAP_POS = np.array([0.48889082, 0.1913024 , 0.23794687])

    def __init__(self,
                 env,
                 randomize_start_jpos=False):
        self._env = env
        self._randomize_start_jpos = randomize_start_jpos
        #TODO: Implement start config sampler

    @overrides
    def reset(self):
        if self._randomize_start_jpos:
            jpos = self.joint_position_space.sample()
            self._env.sim.forward()
            attempts = 1
            while (hasattr(self._env, '_collision_whitelist') and
                   self._env.in_collision(self._env.model)):
                if attempts > 1000:
                    print("Gave up after 1000 attempts")
                    raise RuntimeError

                self._env.sim.data.ctrl[:] = self.joint_position_space.sample()
                self._env.sim.forward()
                for _ in range(100):
                    self._env.sim.step()
                attempts += 1
        else:
            self._env.sim.data.ctrl[:] = self.INITIAL_JOINT_STATE
            self._env.sim.forward()
            for _ in range(100):
                self._env.sim.step()

        return self.get_observation()

    @property
    def joint_position_space(self):
        low = np.array(
            [-0.020833, -0.020833, -3.0503, -3.8095, -3.0426, -3.0439, -2.9761,
             -2.9761, -4.7124])
        high = np.array(
            [0.020833, 0.020833, 3.0503, 2.2736, 3.0426, 3.0439, 2.9761,
             2.9761, 4.7124])
        return Box(low, high, dtype=np.float32)

    @property
    def joint_positions(self):
        curr_pos = []
        curr_pos.append(self._env.sim.data.
            get_joint_qpos('r_gripper_l_finger_joint'))
        curr_pos.append(self._env.sim.data.
            get_joint_qpos('r_gripper_r_finger_joint'))
        for i in range(7):
            curr_pos.append(
                self._env.sim.data.get_joint_qpos('right_j{}'.format(i)))
        return np.array(curr_pos)

    @joint_positions.setter
    def joint_positions(self, jpos):
        self._env.sim.data.set_joint_qpos('r_gripper_l_finger_joint', jpos[0])
        self._env.sim.data.set_joint_qpos('r_gripper_r_finger_joint', jpos[1])
        for i, p in enumerate(jpos[2:]):
            self._env.sim.data.set_joint_qpos('right_j{}'.format(i), p)

    @property
    def gripper_position(self):
        return self._env.sim.data.get_body_xpos('r_gripper_r_finger_tip')

    @property
    def gripper_state(self):
        state = self._env.sim.data.ctrl[0] / self.GRIPPER_STATE_SCALE
        state = state * 2 - 1
        return state

    @gripper_state.setter
    def gripper_state(self, state):
        state = (state + 1.) / 2.
        self._env.sim.data.ctrl[:2] = np.array([
            state * self.GRIPPER_STATE_SCALE,
            -state * self.GRIPPER_STATE_SCALE])
