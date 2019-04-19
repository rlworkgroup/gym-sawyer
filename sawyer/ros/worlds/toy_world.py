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
        def __init__(self, name, init_pose, hole_name):
            self._name = name
            self._hole_name = hole_name
            self._initial_pose = init_pose
            self._pose = Pose(init_pose.position, init_pose.orientation)

        @property
        def name(self):
            return self._name

        @property
        def initial_pose(self):
            return self._initial_pose

        @property
        def hole_name(self):
            return self._hole_name

        @property
        def position(self):
            return self._pose.position

        @position.setter
        def position(self, value):
            self._pose.position = value

        @property
        def orientation(self):
            return self._pose.orientation

        @orientation.setter
        def orientation(self, value):
            self._pose.orientation = value 

    def __init__(self, name, init_pose, lid_name, lid_init_pose, hole_name, resource=None):
        self._name = name
        self._resource = resource                 
        self._initial_pose = init_pose
        self._pose = Pose(init_pose.position, init_pose.orientation)        
        self._lid = self.Lid(lid_name, lid_init_pose, hole_name)

    @property
    def name(self):
        return self._name

    @property
    def resource(self):
        return self._resource

    @property
    def initial_pose(self):
        return self._initial_pose

    @property
    def position(self):
        return self._pose.position

    @position.setter
    def position(self, value):
        self._pose.position = value

    @property
    def orientation(self):
        return self._pose.orientation

    @orientation.setter
    def orientation(self, value):
        self._pose.orientation = value     

    @property
    def observation_space(self):
        return gym.spaces.Box(np.inf, -np.inf, shape=(6,), dtype=np.float32)   

    @property
    def lid(self):
        return self._lid

    def reset(self):
        init_position = self._initial_pose.position
        self.position = Point(x=init_position.x, y=init_position.y, z=init_position.z)
        return self.initial_pose

    def get_observation(self):
        box_pos = np.array(
            [self.position.x, self.position.y, self.position.z])
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
    def __init__(self, name, init_pose, resource=None):
        self._name = name
        self._resource = resource
        self._initial_pose = init_pose 
        self._pose = Pose(init_pose.position, init_pose.orientation)

    @property
    def name(self):
        return self._name

    @property
    def resource(self):
        return self._resource

    @property
    def initial_pose(self):
        return self._initial_pose

    @property
    def position(self):
        return self._pose.position

    @position.setter
    def position(self, value):
        self._pose.position = value

    @property
    def orientation(self):
        return self._pose.orientation

    @orientation.setter
    def orientation(self, value):
        self._pose.orientation = value

    @property
    def observation_space(self):
        return gym.spaces.Box(np.inf, -np.inf, shape=(6,), dtype=np.float32)

    def reset(self):
        init_pos = self._initial_pose.position
        init_ori = self._initial_pose.orientation
        self.position = Point(x=init_pos.x, y=init_pos.y, z=init_pos.z)
        self.orientation = Quaternion(x=init_ori.x, y=init_ori.y, z=init_ori.z, w=init_ori.w)
        return self.initial_pose

    def get_observation(self):
        pos = np.array(
            [self.position.x, self.position.y, self.position.z])
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

    @property
    def box(self):
        return self._box

    @property
    def peg(self):
        return self._peg

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
                Pose(position=Point(x=0.856, y=-0.042, z=0.007)),
                osp.join(World.MODEL_DIR, 'box_with_lid/model.urdf'))
            box = BoxWithLid(
                name='box',
                init_pose=Pose(Point(0.856, -0.042, 0.007), Quaternion(0, 0, 0, 1)),
                lid_name='lid',
                lid_init_pose=Pose(Point(0.851, -0.037, 0.130), Quaternion(0, 0, 0, 1)),
                hole_name='hole',                
                resource=osp.join(World.MODEL_DIR, 'box_with_lid/model.urdf'))
            self._box = box

            # Block Peg
            Gazebo.load_gazebo_model(
                'peg',
                Pose(position=Point(x=0.793, y=0.293, z=0.092)),
                osp.join(World.MODEL_DIR, 'peg/model.urdf'))
            peg = BlockPeg(
                name='peg',
                init_pose=Pose(Point((0.793, 0.293, 0.092), Quaternion(0, 0, 0, 1))), 
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
              
            box_init_pose = self.get_box_pose(box_name)
            lid_init_pose = self.get_lid_pose(lid_name)
            box = BoxWithLid(box_name, box_init_pose, lid_name, lid_init_pose, hole_name)
            self._box = box

            peg_init_pose = self.get_peg_pose(peg_name)               
            peg = BlockPeg(peg_name, peg_init_pose)
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
            obj.orientation = obj_pose.orientation

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
                new_pos = obj.reset().position
                logger.log('position for {} is x = {}, y = {}, z = {}'.
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
                        box_pose = self.get_box_pose()
                        self._box.position = box_pose.position
                        self._box.orientation = box_pose.orientation
                        
                        lid_pose = self.get_lid_pose()
                        self._box.lid.position = lid_pose.position
                        self._box.lid.orientation = lid_pose.orientation

            #Update peg position using apriltags
                        peg_pose = self.get_peg_pose()
                        self._peg.position = peg_pose.position
                        self._peg.orientation = peg_pose.orientation

                
        obs = {}
        objs = [self._box, self._peg]
        for obj in objs:
            obs = {**obs, **obj.get_observation()}
        return obs

    def get_box_pose(self, box_frame=None):
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
        
        box_pos = Point(box_pos[0], box_pos[1], box_pos[2])
        box_ori = Quaternion(box_ori[0], box_ori[1], box_ori[2], box_ori[3])
        return Pose(box_pos, box_ori)

    def get_lid_pose(self, lid_frame=None):        
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
        
        lid_pos = Point(lid_pos[0], lid_pos[1], lid_pos[2])
        lid_ori = Quaternion(lid_ori[0], lid_ori[1], lid_ori[2], lid_ori[3])
        return Pose(lid_pos, lid_ori)

    def get_peg_pose(self, peg_frame=None):        
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

        peg_pos = Point(peg_pos[0], peg_pos[1], peg_pos[2])
        peg_ori = Quaternion(peg_ori[0], peg_ori[1], peg_ori[2], peg_ori[3])
        return Pose(peg_pos, peg_ori)

    def get_lid_hole_pose(self, hole_frame=None):
        if hole_frame is None:
            hole_frame = self._box.lid.hole_name
        self._tf_listener.waitForTransform(
            self._base_frame, hole_frame, rospy.Time(0), rospy.Duration(2))        
        hole_pos, hole_ori = self._tf_listener.lookupTransform(
            self._base_frame, hole_frame, rospy.Time(0))
        
        hole_pos = Point(hole_pos[0], hole_pos[1], hole_pos[2])
        hole_ori = Quaternion(hole_ori[0], hole_ori[1], hole_ori[2], hole_ori[3])
        return Pose(hole_pos, hole_ori)

    @property
    def observation_space(self):
        spaces = [self._box.observation_space, self._peg.observation_space]

        high = np.concatenate([sp.high for sp in spaces]).ravel()
        low = np.concatenate([sp.low for sp in spaces]).ravel()
        return gym.spaces.Box(high, low, dtype=np.float32)
