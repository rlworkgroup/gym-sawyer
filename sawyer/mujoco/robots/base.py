import abc

class Robot(abc.ABC):
    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_observation(self):
        raise NotImplementedError

    @abc.abstractproperty
    def action_space(self):
        raise NotImplementedError

    @abc.abstractproperty
    def observation_space(self):
        raise NotImplementedError