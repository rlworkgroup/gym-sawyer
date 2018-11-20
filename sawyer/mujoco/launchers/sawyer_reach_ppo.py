from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from garage.tf.algos import PPO
from garage.envs import normalize
from sawyer.mujoco.reacher_env import SimpleReacherEnv
from garage.envs.env_spec import EnvSpec
from garage.misc.instrument import run_experiment
from garage.tf.envs import TfEnv
from sandbox.embed2learn.policies.gaussian_mlp_policy import GaussianMLPPolicy
from garage.tf.baselines import GaussianMLPBaseline

GOALS = [
    # (  ?,    ?,   ?)
    (0.6, 0.3, 0.3),
    # (0.3, 0.6, 0.15),
    # (-0.3, 0.6, 0.15),
]


def run_task(v):
    v = SimpleNamespace(**v)

    # Environment
    env = SimpleReacherEnv(goal_position=GOALS[0], control_method="position_control", completion_bonus=5)

    env = TfEnv(env)

    # Policy
    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=(64, 32),
        init_std=v.policy_init_std,
    )

    baseline = GaussianMLPBaseline(env_spec=env.spec)

    algo = PPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v.batch_size,  # 4096
        max_path_length=v.max_path_length,
        n_itr=1000,
        discount=0.99,
        step_size=0.2,
        optimizer_args=dict(batch_size=32, max_epochs=10),
        plot=True,
    )
    algo.train()


config = dict(
    batch_size=4096,
    max_path_length=100,  # 50
    policy_init_std=0.1,  # 1.0
)

run_experiment(
    run_task,
    exp_prefix='sawyer_reach_ppo_position',
    n_parallel=4,
    seed=1,
    variant=config,
    plot=True,
)
