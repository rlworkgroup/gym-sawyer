import collections
import os.path as osp

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import (Point, Pose, PoseStamped, Quaternion,
                               TransformStamped)
import gym
import numpy as np
import rospy
from tf import TransformListener
from sawyer.ros.worlds.gazebo import Gazebo
from sawyer.ros.worlds.world import World
import sawyer.garage.misc.logger as logger

class BoxWithLid:    

    class Lid:
        def __init__(self, name, initial_pos, hole_name):
            self._name = name
            self._hole_name = hole_name
            self._initial_pos = Point(
                x=initial_pos[0], y=initial_pos[1], z=initial_pos[2])
            self._position = Point(
                x=initial_pos[0], y=initial_pos[1], z=initial_pos[2])
            self._orientation = Quaternion(x=0., y=0., z=0., w=1.)

        @property
        def name(self):
            return self._name

        @property
        def hole_name(self):
            return self._hole_name

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

    def __init__(self, name, init_pos, lid_name, lid_init_pos, hole_name, random_delta_range, resource=None):
        self._name = name
        self._resource = resource                 
        self._initial_pos = Point(x=init_pos[0], y=init_pos[1], z=init_pos[2])
        self._random_delta_range = random_delta_range         
        self._position = Point(x=init_pos[0], y=init_pos[1], z=init_pos[2])
        self._orientation = Quaternion(x=0., y=0., z=0., w=1.)        
        self._lid = self.Lid(lid_name, lid_init_pos, hole_name)

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
    def lid(self):
        return self._lid

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
        box_pos = np.array(
            [self.position.x, self.position.y, self._position.z])
        box_ori = np.array([
            self.orientation.x, self.orientation.y, self.orientation.z,
            self.orientation.w
        ])
        lid_pos = np.array(
            [self.lid.position.x, self.lid.position.y, self.lid.position.z])
        lid_ori = np.array([
            self.lid.orientation.x, self.lid.orientation.y, self.lid.orientation.z,
            self.lid.orientation.w
        ])

        return {
            '{}_base_position'.format(self.name): box_pos,
            '{}_base_orientation'.format(self.name): box_ori,
            '{}_lid_position'.format(self.name): lid_pos,
            '{}_lid_orientation'.format(self.name): lid_ori,
        }


class BlockPeg:
    def __init__(self, name, init_pos, random_delta_range, resource=None):
        self._name = name
        self._resource = resource
        self._init_pos = Point(
            x=init_pos[0], y=init_pos[1], z=init_pos[2])
        self._random_delta_range = random_delta_range
        self._position = Point(
            x=init_pos[0], y=init_pos[1], z=init_pos[2])
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
        return self._init_pos

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
        self._box = None
        self._peg = None
        self._simulated = simulated
        self._moveit_scene = moveit_scene
        self._frame_id = frame_id
        self._tf_listener = None
        self._base_frame = "base_d"

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
                Pose(position=Point(x=0.756, y=-0.042, z=0.007)),
                osp.join(World.MODEL_DIR, 'box_with_lid/model.urdf'))
            box = BoxWithLid(
                name='box',
                init_pos=(0.756, -0.042, 0.007),
                lid_name='lid',
                lid_init_pos=(0.751, -0.037, 0.130),
                hole_name='hole',                
                random_delta_range=1,
                resource=osp.join(World.MODEL_DIR, 'box_with_lid/model.urdf'))
            self._box = box

            # Block Peg
            Gazebo.load_gazebo_model(
                'peg',
                Pose(position=Point(x=0.793, y=0.293, z=0.092)),
                osp.join(World.MODEL_DIR, 'peg/model.urdf'))
            peg = BlockPeg(
                name='peg',
                initial_pos=(0.793, 0.293, 0.092),
                random_delta_range=1,
                resource=osp.join(World.MODEL_DIR, 'block_peg/model.urdf'))
            self._peg = peg

            # Waiting for models to be loaded before initializing subscriber
            rospy.wait_for_message('/gazebo/model_states')
            self._obj_state_sub = rospy.Subscriber('/gazebo/model_states', ModelStates,
                                       self._gazebo_update_obj_states)
        else:
            self._tf_listener = TransformListener()
            box_name = 'box'
            lid_name = 'lid'
            peg_name = 'peg'
            hole_name = 'hole'
            box_init_pos, _ = self.get_box_location(box_name)
            lid_init_pos, _ = self.get_lid_location(lid_name)            
            box = BoxWithLid(box_name, box_init_pos, lid_name, lid_init_pos, hole_name, 
                    random_delta_range=1)
            self._box = box

            peg_init_pos, _ = self.get_peg_location(peg_name)               
            peg = BlockPeg(name=peg_name, init_pos=peg_init_pos, random_delta_range=1)
            self._peg = peg

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
        if self._moveit_scene:
            self._moveit_scene.add_box('table', pose_stamped, (1.0, 0.9, 0.1))

    def _gazebo_update_obj_states(self, data):
        model_states = data
        model_names = model_states.name
        objs = [self._box, self._peg]
        for obj in objs:
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
            objs = [self._box, self._peg]
            for obj in objs:
                new_pos = obj.reset()
                Gazebo.set_model_pose(
                    obj.name,
                    new_pose=Pose(
                        position=Point(x=new_pos.x, y=new_pos.y, z=new_pos.z)))
        else:
            objs = [self._box, self._peg]
            for obj in objs:
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
        self._obj_state_sub.unregister()
        if self._simulated:
            objs = [self._box, self._peg]
            for obj in objs:
                Gazebo.delete_gazebo_model(obj.name)
            Gazebo.delete_gazebo_model('table')
        else:
            ready = False
            while not ready:
                ans = input('Are you ready to exit?[Yes/No]\n')
                if ans.lower() == 'yes' or ans.lower() == 'y':
                    ready = True

        if self._moveit_scene:
            self._moveit_scene.remove_world_object('table')

    def get_observation(self):
        if not self._simulated:
            #Update box and lid positions using apriltags
                        box_pos, box_ori = self.get_box_location()
                        self._box.position = Point(x=box_pos[0], y=box_pos[1], z=box_pos[2])
                        self._box.orientation = Quaternion(x=box_ori[0], y=box_ori[1], z=box_ori[2], w=box_ori[3])
                        
                        lid_pos, lid_ori = self.get_lid_location()
                        self._box.lid.position = Point(x=lid_pos[0], y=lid_pos[1], z=lid_pos[2])
                        self._box.lid.orientation = Quaternion(x=lid_ori[0], y=lid_ori[1], z=lid_ori[2], w=lid_ori[3])

            #Update peg position using apriltags
                        peg_pos, peg_ori = self.get_peg_location()
                        self._peg.position = Point(x=peg_pos[0], y=peg_pos[1], z=peg_pos[2])
                        self._peg.orientation = Quaternion(x=peg_ori[0], y=peg_ori[1], z=peg_ori[2], w=peg_ori[3])

                
        obs = {}
        objs = [self._box, self._peg]
        for obj in objs:
            obs = {**obs, **obj.get_observation()}
        return obs

    def get_box_location(self, box_frame=None):
        box_pos = None
        box_ori = None
        if box_frame is None:
            box_frame = self._box.name

        #Wait for transform base -> box
        self._tf_listener.waitForTransform(
            self._base_frame, box_frame, rospy.Time(0), rospy.Duration(2))
        #Get transform base -> box
        box_pos, box_ori = self._tf_listener.lookupTransform(
            self._base_frame, box_frame, rospy.Time(0))
        
        return box_pos, box_ori

    def get_lid_location(self, lid_frame=None):        
        lid_pos = None
        lid_ori = None
        if lid_frame is None:
            lid_frame = self._box.lid.name

        #Wait for transform base -> lid
        self._tf_listener.waitForTransform(
            self._base_frame, lid_frame, rospy.Time(0), rospy.Duration(2))
        #Get transform base -> lid
        lid_pos, lid_ori = self._tf_listener.lookupTransform(
            self._base_frame, lid_frame, rospy.Time(0))
        return lid_pos, lid_ori

    def get_peg_location(self, peg_frame=None):        
        peg_pos = None
        peg_ori = None
        if peg_frame is None:
            peg_frame = self._peg.name
        
        #Wait for transform base -> peg
        self._tf_listener.waitForTransform(
            self._base_frame, peg_frame, rospy.Time(0), rospy.Duration(2))
        #Get transform base -> peg
        peg_pos, peg_ori = self._tf_listener.lookupTransform(
            self._base_frame, peg_frame, rospy.Time(0))

        return peg_pos, peg_ori

    def get_lid_hole_location(self, hole_frame=None):
        if hole_frame is None:
            hole_frame = self._box.lid.hole_name
        self._tf_listener.waitForTransform(
            self._base_frame, hole_frame, rospy.Time(0), rospy.Duration(2))        
        hole_pos, hole_ori = self._tf_listener.lookupTransform(
            self._base_frame, hole_frame, rospy.Time(0))
        
        return hole_pos, hole_ori

    @property
    def observation_space(self):
        spaces = [self._box.observation_space, self._peg.observation_space]

        high = np.concatenate([sp.high for sp in spaces]).ravel()
        low = np.concatenate([sp.low for sp in spaces]).ravel()
        return Box(high, low, dtype=np.float32)
