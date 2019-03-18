"""Testing for sawyer envrionments. """

from sawyer.mujoco.envs import ReacherEnv
from tests.helpers import step_env
import numpy as np

class TestSawyerEnvs():
    def test_reacher_position_control(self):
        """Testing reacherenv for position control."""
        env = ReacherEnv(goal_position=(0.65, 0.0, 0.0), 
                            xml_path='reacherenv_default.xml',
                            xml_config='default')
        step_env(env, n=500, render=True)

    def test_reacher_task_control(self):
        """Testing reacherenv for position control."""
        env = ReacherEnv(goal_position=(0.65, 0, 0),
                            xml_path='reacherenv_task.xml',
                            xml_config='task')
        for _ in range(200):
            action = np.array([0.01, 0, 0, -1])
            env.step(action)
            env.render()



