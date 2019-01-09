import collections
import os.path as osp

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import (Point, Pose, PoseStamped, Quaternion,
                               TransformStamped)
import gym
import numpy as np
import rospy

from sawyer.ros.worlds.gazebo import Gazebo
from sawyer.ros.worlds.world import World
import sawyer.garage.misc.logger as logger
try:
    from sawyer.garage.config import AR_TOPICS
except ImportError:
    raise NotImplementedError(
        "Please set AR_TOPICS in sawyer.garage.config_personal.py! "
        "example 1:"
        "   AR_TOPICS = ['<ar_topic_name>']"
        "example 2:"
        "   # if you are not using real robot and AR tags"
        "   AR_TOPICS = []")

class BoxWithLid:
    #TODO: Track lid separately from box
    def __init__(self, name, initial_pos, random_delta_range, resource=None):
        self._name = name
        self._resource = resource
        self._initial_pos = Point(
            x=initial_pos[0], y=initial_pos[1], z=initial_pos[2])
        self._random_delta_range = random_delta_range
        self._position = Point(
            x=initial_pos[0], y=initial_pos[1], z=initial_pos[2])
        self._orientation = Quaternion(x=0., y=0., z=0., w=1.)

    @property
    def random_delta_range(self):
        return self._random_delta_range

    @property
    def name(self):
        return self._name

    @property
    def resource(self):
        return self._resource

    @property
    def initial_pos(self):
        return self._initial_pos

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        self._orientation = value

    def reset(self):
        random_delta = np.zeros(2)
        new_pos = self.initial_pos
        while np.linalg.norm(random_delta) < 0.1:
            random_delta = np.random.uniform(
                -self.random_delta_range,
                self.random_delta_range,
                size=2)
        new_pos.x += random_delta[0]
        new_pos.y += random_delta[1]
        return new_pos

    def get_observation(self):
        pos = np.array(
            [self.position.x, self.position.y, self._position.z])
        ori = np.array([
            self.orientation.x, self.orientation.y, self.orientation.z,
            self.orientation.w
        ])
        return {
            '{}_position'.format(self.name): pos,
            '{}_orientation'.format(self.name): ori,
        }

class BlockPeg:
    def __init__(self, name, initial_pos, random_delta_range, resource=None):
        self._name = name
        self._resource = resource
        self._initial_pos = Point(
            x=initial_pos[0], y=initial_pos[1], z=initial_pos[2])
        self._random_delta_range = random_delta_range
        self._position = Point(
            x=initial_pos[0], y=initial_pos[1], z=initial_pos[2])
        self._orientation = Quaternion(x=0., y=0., z=0., w=1.)

    @property
    def random_delta_range(self):
        return self._random_delta_range

    @property
    def name(self):
        return self._name

    @property
    def resource(self):
        return self._resource

    @property
    def initial_pos(self):
        return self._initial_pos

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        self._orientation = value

    @property
    def observation_space(self):
        return Box(np.inf, -np.inf, shape=(6,), dtype=np.float32)

    def reset(self):
        random_delta = np.zeros(2)
        new_pos = self.initial_pos
        while np.linalg.norm(random_delta) < 0.1:
            random_delta = np.random.uniform(
                -self.random_delta_range,
                self.random_delta_range,
                size=2)
        new_pos.x += random_delta[0]
        new_pos.y += random_delta[1]
        return new_pos

    def get_observation(self):
        pos = np.array(
            [self.position.x, self.position.y, self._position.z])
        ori = np.array([
            self.orientation.x, self.orientation.y, self.orientation.z,
            self.orientation.w
        ])
        return {
            '{}_position'.format(self.name): pos,
            '{}_orientation'.format(self.name): ori,
        }

class ToyWorld(World):
    def __init__(self, moveit_scene, frame_id, simulated=False):
        self._boxes = []
        self._pegs = []
        self._box_state_subs = []
        self._peg_state_subs = []
        self._simulated = simulated
        self._moveit_scene = moveit_scene
        self._frame_id = frame_id

    def initialize(self):
        if self._simulated:
            # Table
            Gazebo.load_gazebo_model(
                'table',
                Pose(position=Point(x=0.75, y=0.0, z=0.0)),
                osp.join(World.MODEL_DIR, 'cafe_table/model.sdf'))

            # Box with Lid
            Gazebo.load_gazebo_model(
                'box',
                Pose(position=Point(x=0.5725, y=0.1265, z=0.90)),
                osp.join(World.MODEL_DIR, 'box_with_lid/model.urdf'))
            box = BoxWithLid(
                name='box',
                initial_pos=(0.5725, 0.1265, 0.90),
                random_delta_range=0,
                resource=osp.join(World.MODEL_DIR, 'box_with_lid/model.urdf'))
            self._boxes.append(box)

            # Block Peg
            Gazebo.load_gazebo_model(
                'peg',
                Pose(position=Point(x=0.5725, y=0.1265, z=0.90)),
                osp.join(World.MODEL_DIR, 'peg/model.urdf'))
            peg = BlockPeg(
                name='peg',
                initial_pos=(0.5725, 0.1265, 0.90),
                random_delta_range=0,
                resource=osp.join(World.MODEL_DIR, 'block_peg/model.urdf'))
            self._pegs.append(peg)

            # Waiting for models to be loaded before initializing subscriber
            rospy.wait_for_message('/gazebo/model_states')
            self._obj_state_subs.append(
                rospy.Subscriber('/gazebo/model_states', ModelStates,
                    self._gazebo_update_obj_states))
        else:
            #TODO: AR tags don't have separate topics for each object!
            for ar_topic in AR_TOPICS:
                if topic_is_box:
                    box = BoxWithLid(
                        name='box',
                        initial_pos=(),
                        random_delta_range=0,
                        resource=ar_topic)
                    self._boxes.append(box)
                    self._obj_state_subs.append(
                        rospy.Subscriber(
                            box.resource,
                            TransformStamped,
                            self._ar_update_obj_states))
                elif topic_is_peg:
                    peg = BlockPeg(
                        name='peg',
                        initial_pos=(),
                        random_delta_range=0,
                        resource=ar_topic)
                    self._pegs.append(peg)

        # Add table to moveit
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self._frame_id
        pose_stamped.pose.position.x = 0.655
        pose_stamped.pose.position.y = 0
        # Leave redundant space
        pose_stamped.pose.position.z = -0.02
        pose_stamped.pose.orientation.x = 0
        pose_stamped.pose.orientation.y = 0
        pose_stamped.pose.orientation.z = 0
        pose_stamped.pose.orientation.w = 1.0
        self._moveit_scene.add_box('table', pose_stamped, (1.0, 0.9, 0.1))

    def _gazebo_update_obj_states(self, data):
        model_states = data
        model_names = model_states.name
        for obj in self._boxes + self._pegs:
            obj_idx = model_names.index(obj.name)
            obj_pose = model_states.pose[obj_idx]
            obj.position = obj_pose.position
            obj.orientation = obj_pos.orientation

    def _ar_update_obj_states(self, data):
        translation = data.transform.translation
        rotation = data.transform.rotation
        child_frame_id = data.child_frame_id

        for block in self._blocks:
            if block.resource == child_frame_id:
                block.position = translation
                block.orientation = rotation

    def reset(self):
        if self._simulated:
            for obj in self._boxes + self._pegs:
                new_pos = obj.reset()
                Gazebo.set_model_pose(
                    obj.name,
                    new_pose=Pose(
                        position=Point(x=new_pos.x, y=new_pos.y, z=new_pos.z)))
        else:
            for obj in self._boxes + self._pegs:
                new_pos = obj.reset()
                logger.log('new position for {} is x = {}, y = {}, z = {}'.
                    format(obj.name, new_pos.x, new_pos.y, new_pos.z))
                ready = False
                while not ready:
                    ans = input(
                        'Have you finished setting up {}?[Yes/No]\n'.format(
                            obj.name))
                    if ans.lower() == 'yes' or ans.lower() == 'y':
                        ready = True

    def terminate(self):
        for sub in self._box_state_subs + self._peg_state_subs:
            sub.unregister()

        if self._simulated:
            for obj in self._boxes + self._pegs:
                Gazebo.delete_gazebo_model(obj.name)
            Gazebo.delete_gazebo_model('table')
        else:
            ready = False
            while not ready:
                ans = input('Are you ready to exit?[Yes/No]\n')
                if ans.lower() == 'yes' or ans.lower() == 'y':
                    ready = True

        self._moveit_scene.remove_world_object('table')

    def get_observation(self):
        obs = {}
        for obj in self._boxes + self._pegs:
            obs = {**obs, **obj.get_observation()}
        return obs

    @property
    def observation_space(self):
        spaces = []
        for box in self._boxes:
            spaces.append(box.observation_space)
        for peg in self._pegs:
            spaces.append(peg.observation_space)

        high = np.concatenate([sp.high for sp in spaces]).ravel()
        low = np.concatenate([sp.low for sp in spaces]).ravel()
        return Box(high, low, dtype=np.float32)
