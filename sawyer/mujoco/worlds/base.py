import abc

class WorldObject(abc.ABC):
    """
    Helper class for tracking a single object in a MuJoCo simulation.

    An WorldObject should stricly own 1 body, though the body may have child
    bodies of its own.
    """
    @property
    def name(self):
        # A WorldObject should have a globally unique name (used as prefix)
        return self._name
    
    @property
    def resource(self):
        # A WorldObject should have an xml file path 
        return self._resource

    @abc.abstractproperty
    def observation_space(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_observation(self):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError


class World(abc.ABC):
    """
    Helper class to track many objects in a MuJoCo simulation.

    Worlds do not track the robot.
    """
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
    def observation_space(self):
        raise NotImplementedError
