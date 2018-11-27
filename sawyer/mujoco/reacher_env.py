import numpy as np

from sawyer.garage.core.serializable import Serializable
from sawyer.mujoco.sawyer_env import SawyerEnv
from sawyer.mujoco.sawyer_env import Configuration
from sawyer.mujoco.sawyer_env import SawyerEnvWrapper
from sawyer.garage.misc.overrides import overrides
from gym.spaces import Box


class ReacherEnv(SawyerEnv):
    def __init__(self,
                 goal_position,
                 start_position=None,
                 randomize_start_position=False,
                 **kwargs):
        def generate_start_goal():
            nonlocal start_position
            if start_position is None or randomize_start_position:
                center = np.array([0.65, 0, 0]) 
                start_position = np.concatenate([center[:2], [0.15]])

            start = Configuration(
                gripper_pos=start_position,
                gripper_state=1,
                object_grasped=False,
                object_pos=goal_position,
                joint_pos=None)
            goal = Configuration(
                gripper_pos=goal_position,
                gripper_state=1,
                object_grasped=False,
                object_pos=goal_position,
                joint_pos=None)

            return start, goal

        SawyerEnv.__init__(self,
                           start_goal_config=generate_start_goal,
                           file_path='reacher.xml',
                           **kwargs)

    def get_obs(self):
        gripper_pos = self.gripper_position
        if self._control_method == 'task_space_control':
            obs = np.concatenate([gripper_pos])
        elif self._control_method == 'position_control':
            obs = np.concatenate([self.joint_positions[2:], gripper_pos])
        else:
            raise NotImplementedError

        self._achieved_goal = gripper_pos
        self._desired_goal = self._goal_configuration.gripper_pos

        return {
            'observation': obs,
            'achieved_goal': self._achieved_goal,
            'desired_goal': self._desired_goal,
            'has_object': False,
            'gripper_state': self.gripper_state,
            'gripper_pos': gripper_pos,
            'object_pos': self._desired_goal,
        }

    def compute_reward(env, achieved_goal, desired_goal, info):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if env._reward_type == 'sparse':
            return (d < env._distance_threshold).astype(np.float32)

        return -d

    @overrides
    def reset(self):
        super(ReacherEnv, self).reset()

        attempts = 1
        if self._randomize_start_jpos:
            self.joint_positions = self.joint_position_space.sample()
            self.sim.forward()
            while hasattr(self, "_collision_whitelist") and self.in_collision:
                if attempts > 1000:
                    print("Gave up after 1000 attempts")

                self.sim.data.ctrl[:] = self.joint_position_space.sample()
                self.sim.forward()
                for _ in range(100):
                    self.sim.step()
                attempts += 1
        else:
            self.sim.data.ctrl[:] = np.array([0, 0, -0.140923828125, -1.2789248046875, -3.043166015625,
                    -2.139623046875, -0.047607421875, -0.7052822265625, -1.4102060546875,])
            self.sim.forward()
            for _ in range(100):
                self.sim.step()

        return self.get_obs()


class SimpleReacherEnv(SawyerEnvWrapper, Serializable):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.reward_range = None
        self.metadata = None
        super().__init__(ReacherEnv(*args, **kwargs))
