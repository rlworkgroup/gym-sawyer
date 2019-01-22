import numpy as np
import time

def toyenv_distance(obs, waypoint):
    gripper_pos = obs[:3]
    goal_pos = waypoint[:3]
    gripper_state_dist = np.array([obs[3] - waypoint[3]])
    return np.concatenate((goal_pos - gripper_pos, gripper_state_dist), axis=-1)

def toyenv_action(dist, waypoint, epsilon=1e-8, action_scale=0.01):
    dist = dist[:3] / (np.linalg.norm(dist[:3]) + epsilon)
    action = dist * action_scale
    action = np.concatenate((action, [waypoint[3]]), axis=-1)
    return action

def toyenv_eval(dist, waypoint, success_thresh):
    return (np.linalg.norm(dist[:3]) < success_thresh and
        dist[3] < success_thresh)

class WaypointSequencer:
    """
    For recording a demo of an env stepping through a series of waypoints.
    """
    def __init__(self,
                 env,
                 waypoints,
                 distance_fn,
                 action_fn,
                 eval_fn,
                 max_n_steps=10000,
                 success_thresh=0.01):
        """
        Initializer.

        env: gym.Env
        waypoints: np.array w/ shape (n_waypts,) + env.action_space.shape
        distance_fn(obs, waypoint) --> (distance_vector)
        action_fn(distance_vector) --> action
        """
        self._env = env
        self._waypoints = waypoints
        self._distance_fn = distance_fn
        self._action_fn = action_fn
        self._eval_fn = eval_fn
        self._max_n_steps = max_n_steps
        self._success_thresh = success_thresh

        assert waypoints.shape[1] == env.action_space.shape[0]

    def reset(self):
        """
        Resets for new recording.
        """
        return self._env.reset()

    def run(self, render=False):
        """
        Runs env, attempts to reach sequence of waypoints.
        Returns a dataset of actions and observations once complete.
        """
        obs = self.reset()
        step = 0
        waypoint_idx = 0
        recording = []

        for step in range(self._max_n_steps):
            waypoint = self._waypoints[waypoint_idx, :]
            # Determine distance to waypoint
            dist = self._distance_fn(obs, waypoint)
            if (self._eval_fn(dist, waypoint, self._success_thresh)):
                waypoint_idx += 1
                if waypoint_idx < self._waypoints.shape[0]:
                    print("Reached waypoint {0} with obs:\n{1}".
                          format(waypoint, obs))
                    for _ in range(300):
                        self._env.step(np.array([0., 0., 0.0003, waypoint[3]]))
                        self._env.render()
                        time.sleep(0.002)
                else:
                    break

            # Generate action
            act = self._action_fn(dist, waypoint)
            recording.append((obs, act))

            # Step env
            obs, rew, done, info = self._env.step(act)
            print("task:{0}, reward: {1}".format(
                self._env._task_list.index(self._env._active_task),
                rew))
            if render:
                self._env.render()
                time.sleep(0.002)

            if done:
                print("Env terminated at step {0} at obs:\n{1}".format(step, obs))
                break

        print("Finished at step {}".format(step))
        recording = np.array(recording)
        return recording
