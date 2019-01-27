"""ToyEnv task for the Sawyer robot."""

import sys
import gym
import moveit_commander
import numpy as np

from sawyer.mujoco.tasks import (ReachTask, PickTask, PlaceTask, InsertTask,
                                 RemoveTask, OpenTask, CloseTask)
from sawyer.ros.envs.sawyer.sawyer_env import SawyerEnv
from sawyer.ros.robots.sawyer import Sawyer
from sawyer.ros.worlds.toy_world import ToyWorld
from sawyer.garage.core import Serializable
from sawyer.ros.util.common import rate_limited
try:
    from sawyer.config import STEP_FREQ
except ImportError:
    raise NotImplementedError(
        "Please set STEP_FREQ in sawyer.config_personal.py!"
        "example 1: "
        "   STEP_FREQ = 5")

class ToyEnv(SawyerEnv, Serializable):
    def __init__(self, simulated=False, task_list=None):
        Serializable.quick_init(self, locals())

        self.simulated = simulated

        # Initialize moveit to get safety check
        moveit_commander.roscpp_initialize(sys.argv)
        self._moveit_robot = moveit_commander.RobotCommander()
        self._moveit_scene = moveit_commander.PlanningSceneInterface()
        self._moveit_group_name = 'right_arm'
        self._moveit_group = moveit_commander.MoveGroupCommander(
            self._moveit_group_name)

        self._robot = Sawyer(moveit_group=self._moveit_group_name)
        self._world = ToyWorld(self._moveit_scene,
                               self._moveit_robot.get_planning_frame(),
                               simulated)

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
                {'key_object': 'peg', 'lid_object': 'box_lid'},
                {'key_object': 'peg', 'lock_object': 'box_lid'},
                {'place_object': 'peg', 'location': [0.65, 0., 0.]},
            ]
            self._task_list = [task(**kwa) for task, kwa in
                               zip(tasks, task_args)]
        self._active_task = self._task_list[0]

        SawyerEnv.__init__(self, simulated=simulated)

    @property
    def observation_space(self):
        spaces = []
        spaces.append(self._world.observation_space)
        spaces.append(self._robot.observation_space)

        high = np.concatenate([sp.high for sp in spaces]).ravel()
        low = np.concatenate([sp.low for sp in spaces]).ravel()
        return Box(high, low, dtype=np.float32)

    @rate_limited(STEP_FREQ)
    def step(self, action):
        assert action.shape == self.action_space.shape

        # Do the action
        self._robot.send_command(action)

        self._step += 1
        obs = self.get_observation()

        # Robot obs
        robot_obs = self._robot.get_observation()

        # World obs
        world_obs = self._world.get_observation()
        hole_pose, _ = self._world.get_lid_hole_location()

        # Grasp state obs
        grasped_peg_obs = self.has_peg()
        
        in_collision = self._robot.in_collision_state

        info = {
            'l': self._step,
            'action': action,
            'in_collision': in_collision,
            'robot_obs': robot_obs,
            'world_obs': world_obs,
            'gripper_position': self._robot.gripper_pose['position'],
            'gripper_state': self._robot.gripper_state,
            'grasped_peg': grasped_peg_obs,
            'hole_site': hole_pose,
            'lid_joint_state': lid_joint_state,
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

    def reset(self):
        self._step = 0
        self._robot.reset()
        self._world.reset()

        return self.get_observation()

    def sample_goal(self):
        """
        Samples a new goal and returns it.
        """
        #TODO: We don't need this
        raise NotImplementedError

    def get_observation(self):
        # Robot obs
        robot_obs = self._robot.get_observation()

        # World obs
        world_obs = self._world.get_observation()

        # Construct obs specified by observation_space
        obs = []
        obs.append(robot_obs['sawyer_joint_position'])
        obs.append(world_obs['box_base_position'])
        obs.append(world_obs['box_lid_position'])
        obs.append(world_obs['peg_position'])
        obs = np.concatenate(obs).ravel()

        return obs

    def done(self, achieved_goal, goal):
        """
        :return if done: bool
        """
        #TODO: Should we use this?
        raise NotImplementedError

    def is_success(self, obs, info):
        return (self._active_task == self._task_list[-1] and
            self._active_task.is_success(obs, info))

    def next_task(self):
        # Set up env to for next task in sequence
        active_task_idx = self._task_list.index(self._active_task)
        if active_task_idx + 1 == len(self._task_list):
            return True  # Done with all tasks
        else:
            self._active_task = self._task_list[active_task_idx + 1]
            return False

    def reward(self, achieved_goal, goal):
        """
        Compute the reward for current step.
        """
        #TODO: Let's not use the GoalEnv interface
        raise NotImplementedError

    def compute_reward(self, obs, info):
        return self._active_task.compute_reward()

    def has_peg(self):
        gripper_state = self._robot.gripper_state
        if gripper_state != 0.0:
            return False

        peg_pose, _ = self._world.get_peg_location()
        gripper_pose = self._robot.gripper_pose['position']
        max_xy_diff = 0.02
        max_z_diff = 0.2
        return ( abs(peg_pose[0] - gripper_pose.x) < max_xy_diff and
            abs(peg_pose[1] - gripper_pose.y) < max_xy_diff and
            abs(peg_pose[2] - gripper_pose.z) < max_z_diff )

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, value):
        self._goal = value
