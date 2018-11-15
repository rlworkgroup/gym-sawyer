"""Testing for sawyer envrionments. """

from sawyer.mujoco import ReacherEnv
from tests.helpers import step_env

class TestSawyerEnvs():
    def test_reacher(self):
        """Testing for reacher."""
        tasks = [(0.3, -0.3, 0.30), (0.3, 0.3, 0.30)]

        env = ReacherEnv(goal_position=tasks[0])
        step_env(env, n=5, render=True)

