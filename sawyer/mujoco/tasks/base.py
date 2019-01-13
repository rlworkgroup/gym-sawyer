import abc

class ComposableTask(abc.ABC):
    """
    Framework for building a composable series of tasks for a single env.

    A ComposableTask owns the reward function for a particular task, and
    determines when the task is complete. The ComposableTask should be
    largely stateless, getting environment information through obs and info
    args.
    """
    @abc.abstractmethod
    def compute_reward(self, obs, info):
        """ Computes the reward for the task."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_success(self, obs, info):
        """ Determines whether the task has been completed. """
        raise NotImplementedError

    @abc.abstractproperty
    def completion_bonus(self):
        """ Getter for completion bonus. """
        raise NotImplementedError
