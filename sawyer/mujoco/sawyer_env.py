from collections import namedtuple
import os.path as osp

import gym
from gym.envs.robotics import rotations
from gym.envs.robotics.utils import reset_mocap_welds, reset_mocap2body_xpos
from gym.spaces import Box
import numpy as np

from sawyer.garage.envs.mujoco import MujocoEnv
from sawyer.garage.envs.mujoco.mujoco_env import MODEL_DIR
from sawyer.garage.misc.overrides import overrides

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

Configuration = namedtuple(
    "Configuration",
    ["gripper_pos", "gripper_state", "object_grasped", "object_pos"])


def default_reward_fn(env, achieved_goal, desired_goal, _info: dict):
    d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
    if env._reward_type == 'sparse':
        return (d < env._distance_threshold).astype(np.float32)
    return -d


def default_success_fn(env, achieved_goal, desired_goal, _info: dict):
    return (np.linalg.norm(achieved_goal - desired_goal, axis=-1) <
            env._distance_threshold).astype(np.float32)


def default_achieved_goal_fn(env):
    return env.gripper_position


def default_desired_goal_fn(env):
    if env._goal_configuration.object_grasped and not env.has_object:
        return env.object_position
    return env._goal_configuration.gripper_pos


class SawyerEnv(MujocoEnv, gym.GoalEnv):
    """Sawyer Robot Environments."""

    def __init__(self,
                 start_goal_config,
                 reward_fn=default_reward_fn,
                 success_fn=default_success_fn,
                 achieved_goal_fn=default_achieved_goal_fn,
                 desired_goal_fn=default_desired_goal_fn,
                 max_episode_steps=50,
                 completion_bonus=10,
                 distance_threshold=0.05,
                 for_her=False,
                 control_cost_coeff=0.,
                 action_scale=1.0,
                 randomize_start_jpos=False,
                 collision_whitelist=COLLISION_WHITELIST,
                 never_done=True,
                 terminate_on_collision=False,
                 collision_penalty=0.,
                 reward_type='dense',
                 control_method='task_space_control',
                 file_path='pick_and_place.xml',
                 free_object=True,
                 obj_in_env=False,
                 *args,
                 **kwargs):
        """
        Sawyer Environment.
        :param args:
        :param kwargs:
        """
        self._start_goal_config = start_goal_config
        self._reward_fn = reward_fn
        self._success_fn = success_fn
        self._achieved_goal_fn = achieved_goal_fn
        self._desired_goal_fn = desired_goal_fn

        self._start_configuration = None  # type: Configuration
        self._goal_configuration = None  # type: Configuration
        self._achieved_goal = None  # type: np.array
        self._desired_goal = None  # type: np.array
        self.gripper_state = 0.
        self._is_success = False
        self._never_done = never_done

        self._reward_type = reward_type
        self._control_method = control_method
        self._max_episode_steps = max_episode_steps
        self._completion_bonus = completion_bonus
        self._distance_threshold = distance_threshold
        self._step = 0
        self._for_her = for_her
        self._control_cost_coeff = control_cost_coeff
        self._action_scale = action_scale
        self._randomize_start_jpos = randomize_start_jpos
        self._terminate_on_collision = terminate_on_collision
        self._collision_penalty = collision_penalty
        self._free_object = free_object
        self._obj_in_env = obj_in_env
        file_path = osp.join(MODEL_DIR, file_path)
        MujocoEnv.__init__(self, file_path=file_path)

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
        # Only use the mocap when using task space control,
        # otherwise it will mess up the position control
        if self._control_method == "task_space_control":
            self.env_setup()

    def _sample_start_goal(self):
        if isinstance(self._start_goal_config, tuple):
            self._start_configuration, self._goal_configuration = self._start_goal_config
        else:
            self._start_configuration, self._goal_configuration = self._start_goal_config(
            )

    def env_setup(self):
        reset_mocap_welds(self.sim)
        self.sim.forward()

    def render(self, mode="human"):
        viewer = self.get_viewer()
        viewer.add_marker(pos=np.array(self._desired_goal), label="goal", size=0.01 * np.ones(3),)
        super(SawyerEnv, self).render(mode=mode)

    @property
    def joint_position_space(self):
        low = np.array(
            [-0.020833, -0.020833, -3.0503, -3.8095, -3.0426, -3.0439, -2.9761, -2.9761, -4.7124])
        high = np.array(
            [0.020833, 0.020833, 3.0503, 2.2736, 3.0426, 3.0439, 2.9761, 2.9761, 4.7124])
        return Box(low, high, dtype=np.float32)

    @property
    def joint_positions(self):
        curr_pos = []
        curr_pos.append(self.sim.data.get_joint_qpos("r_gripper_l_finger_joint"))
        curr_pos.append(self.sim.data.get_joint_qpos("r_gripper_r_finger_joint"))
        for i in range(7):
            curr_pos.append(
                self.sim.data.get_joint_qpos('right_j{}'.format(i)))
        return np.array(curr_pos)

    @joint_positions.setter
    def joint_positions(self, jpos):
        self.sim.data.set_joint_qpos("r_gripper_l_finger_joint", jpos[0])
        self.sim.data.set_joint_qpos("r_gripper_r_finger_joint", jpos[1])
        for i, p in enumerate(jpos[2:]):
            self.sim.data.set_joint_qpos('right_j{}'.format(i), p)

    def set_gripper_position(self, position):
        reset_mocap2body_xpos(self.sim)
        self.sim.data.mocap_quat[:] = np.array([0, 1, 0, 0])
        self.sim.data.set_mocap_pos('mocap', position)
        for _ in range(100):
            self.sim.step()
            reset_mocap2body_xpos(self.sim)
            self.sim.data.mocap_quat[:] = np.array([0, 1, 0, 0])
            self.sim.data.set_mocap_pos('mocap', position)

    @property
    def gripper_position(self):
        return self.sim.data.get_body_xpos("r_gripper_r_finger_tip")

    def force_object_down(self):
        curr_position = self.object_position
        modified_position = np.concatenate([curr_position[:2], [0.04]])
        self.set_object_position(modified_position)
        self.sim.forward()

    def set_object_position(self, position):
        if self._free_object:
            object_qpos = np.concatenate((position, [1, 0, 0, 0]))
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        else:
            self.sim.data.set_joint_qpos("object0:joint1", position[0])
            self.sim.data.set_joint_qpos("object0:joint2", position[1])

    @property
    def object_position(self):
        return self.sim.data.get_geom_xpos('object0')

    @property
    def has_object(self):
        """Determine if the object is grasped"""
        contacts = tuple()
        for coni in range(self.sim.data.ncon):
            con = self.sim.data.contact[coni]
            contacts += ((con.geom1, con.geom2), )
        finger_id_1 = self.sim.model.geom_name2id('finger_tip_1')
        finger_id_2 = self.sim.model.geom_name2id('finger_tip_2')
        object_id = self.sim.model.geom_name2id('object0')
        if ((finger_id_1, object_id) in contacts or
            (object_id, finger_id_1) in contacts) and (
                (finger_id_2, object_id) in contacts or
                (object_id, finger_id_2) in contacts):
            return True
        else:
            return False

    @overrides
    @property
    def action_space(self):
        if self._control_method == 'torque_control':
            return super(SawyerEnv, self).action_space
        elif self._control_method == 'task_space_control':
            return Box(
                np.array([-0.15, -0.15, -0.15, -1.]),
                np.array([0.15, 0.15, 0.15, 1.]),
                dtype=np.float32)
        elif self._control_method == 'position_control':
            return Box(
                low=np.full(9, -0.02), high=np.full(9, 0.02), dtype=np.float32)
        else:
            raise NotImplementedError

    @overrides
    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.get_obs()['observation'].shape,
            dtype=np.float32)

    def step(self, action):
        assert action.shape == self.action_space.shape

        # Note: you MUST copy the action if you modify it
        a = action.copy()

        # Clip to action space
        a *= self._action_scale
        a = np.clip(a, self.action_space.low, self.action_space.high)
        if self._control_method == "torque_control":
            self.forward_dynamics(a)
            self.sim.forward()
        elif self._control_method == "task_space_control":
            reset_mocap2body_xpos(self.sim)
            self.sim.data.mocap_pos[0, :3] = self.sim.data.mocap_pos[0, :3] + a[:3]
            self.sim.data.mocap_quat[:] = np.array([0, 1, 0, 0])
            self.set_gripper_state(a[3])
            for _ in range(5):
                self.sim.step()
            self.sim.forward()
        elif self._control_method == "position_control":
            curr_pos = self.joint_positions

            next_pos = np.clip(
                a + curr_pos[:],
                self.joint_position_space.low[:],
                self.joint_position_space.high[:]
            )
            self.sim.data.ctrl[:] = next_pos[:]
            self.sim.forward()
            for _ in range(5):
                self.sim.step()
        else:
            raise NotImplementedError
        self._step += 1

        obs = self.get_obs()

        # collision checking is expensive so cache the value
        in_collision = self.in_collision

        info = {
            "l": self._step,
            "grasped": obs["has_object"],
            "gripper_state": obs["gripper_state"],
            "gripper_position": obs["gripper_pos"],
            "object_position": obs["object_pos"],
            "is_success": self._is_success,
            "in_collision": in_collision,
        }

        r = self.compute_reward(
            achieved_goal=self._achieved_goal,
            desired_goal=self._desired_goal,
            info=info)

        self._is_success = self._success_fn(self, self._achieved_goal,
                                            self._desired_goal, info)
        done = self._is_success and not self._never_done

        # collision detection
        if in_collision:
            r -= self._collision_penalty
            if self._terminate_on_collision:
                done = True

        if self._is_success:
            r = self._completion_bonus

        info["r"] = r
        info["d"] = done

        return obs, r, done, info

    def set_gripper_state(self, state):
        self.gripper_state = state
        state = (state + 1.) / 2.
        self.sim.data.ctrl[:2] = np.array([state * 0.020833, -state * 0.020833])

    def get_obs(self):
        gripper_pos = self.gripper_position
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('grip') * dt

        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        object_pos = self.object_position
        object_rot = rotations.mat2euler(
            self.sim.data.get_site_xmat('object0'))
        object_velp = self.sim.data.get_site_xvelp('object0') * dt
        object_velr = self.sim.data.get_site_xvelr('object0') * dt
        object_rel_pos = object_pos - gripper_pos
        object_velp -= grip_velp
        grasped = self.has_object
        obs = np.concatenate([
            gripper_pos,
            object_pos.ravel(),  # TODO remove object_pos (reveals task id)
            object_rel_pos.ravel(),
            object_rot.ravel(),
            object_velp.ravel(),
            object_velr.ravel(),
            grip_velp,
            qpos,
            qvel,
            [float(grasped), self.gripper_state],
        ])

        achieved_goal = self._achieved_goal_fn(self)
        desired_goal = self._desired_goal_fn(self)

        return {
            'observation': obs,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal,
            'gripper_state': self.gripper_state,
            'gripper_pos': gripper_pos,
            'has_object': grasped,
            'object_pos': object_pos,
        }

    def is_success(self):
        return self._is_success

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self._reward_fn(self, achieved_goal, desired_goal, info)

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

    @overrides
    def reset(self):
        self._step = 0
        super(SawyerEnv, self).reset()

        self._sample_start_goal()
        if self._obj_in_env:
            self.set_object_position(self._start_configuration.object_pos)
        self.sim.forward()
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


def ppo_info(info):
    info["task"] = [1]
    ppo_infos = {"episode": info}
    return ppo_infos


class SawyerEnvWrapper:
    def __init__(self,
                 env: SawyerEnv,
                 info_callback=ppo_info,
                 use_max_path_len=True):
        self.env = env
        self._info_callback = info_callback
        self._use_max_path_len = use_max_path_len

    def step(self, action):
        goal_env_obs, r, done, info = self.env.step(action=action)
        return goal_env_obs.get('observation'), r, done, info

    def reset(self):
        goal_env_obs = self.env.reset()
        return goal_env_obs.get('observation')

    def render(self, mode='human'):
        self.env.render(mode)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def close(self):
        self.env.close()
