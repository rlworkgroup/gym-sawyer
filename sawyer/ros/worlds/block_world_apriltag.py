import collections
import copy
import os.path as osp

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
import gym
from moveit_msgs.msg import CollisionObject
import numpy as np
import rospy
from tf import TransformListener
from tf.msg import tfMessage

from sawyer.ros.worlds.gazebo import Gazebo
from sawyer.ros.worlds.world import World
from sawyer.ros.worlds.moveit_planningscene_controller import MoveitPlanningSceneController
import sawyer.garage.misc.logger as logger

class Block:
    def __init__(self,
                 name,
                 size,
                 initial_pos,
                 random_delta_range,
                 resource=None):
        """
        Task Object interface
        :param name: str
        :param size: [float]
                [x, y, z]
        :param initial_pos: geometry_msgs.msg.Point
                object's original position. Use this for
                training from scratch on ros/gazebo.
        :param random_delta_range: [float, float, float]
                positive, the range that would be used in
                sampling object' new start position for every episode.
                Set it as 0, if you want to keep the
                object's initial_pos for every episode.
                Use this for training from scratch on ros/gazebo
        :param resource: str
                the model path(str) for simulation training or ros
                topic name for real robot training
        """
        self._name = name
        self._resource = resource
        self._size = size
        self._initial_pos = Point(
            x=initial_pos[0], y=initial_pos[1], z=initial_pos[2])
        self._random_delta_range = random_delta_range
        self._position = Point(
            x=initial_pos[0], y=initial_pos[1], z=initial_pos[2])
        self._orientation = Quaternion(x=0., y=0., z=0., w=1.)
        # If it's first smoothed, set data directly.
        self.first_smoothed = True

    @property
    def size(self):
        return self._size

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
        """Position with reference to robot frame."""
        return self._position

    @position.setter
    def position(self, value):
        """Position with reference to robot frame."""
        self._position = value

    @property
    def orientation(self):
        """Orientation with reference to robot frame."""
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        """Orientation with reference to robot frame."""
        self._orientation = value


class BlockWorld(World):
    def __init__(self, moveit_scene, frame_id, simulated=False, num_of_blocks=1):
        """Users use this to manage world and get world state."""
        self._blocks = []
        self._simulated = simulated
        self._block_states_subs = []
        self._moveit_scene = moveit_scene
        self._frame_id = frame_id
        # Use this to move collision object in moveit.
        self._moveit_col_obj_pub = rospy.Publisher(
            'collision_object', CollisionObject, queue_size=10)
        self._lowpass_alpha = 1
        self._moveit_scene_controller = MoveitPlanningSceneController(frame_id)
        self._tf_listener = TransformListener()
        self._num_of_blocks = num_of_blocks

    def initialize(self):
        """Initialize the block world."""
        if self._simulated:
            Gazebo.load_gazebo_model(
                'table',
                Pose(position=Point(x=0.75, y=0.0, z=0.0)),
                osp.join(World.MODEL_DIR, 'cafe_table/model.sdf'))
            Gazebo.load_gazebo_model(
                'block',
                Pose(position=Point(x=0.5725, y=0.1265, z=0.05)),
                osp.join(World.MODEL_DIR, 'block/model.urdf'))
            block_name = 'block_{}'.format(len(self._blocks))
            block = Block(
                name=block_name,
                size=[0.11, 0.11, 0.13],
                initial_pos=(0.5725, 0.1265, 0.05),
                random_delta_range=0.15,
                resource=osp.join(World.MODEL_DIR, 'block/model.urdf'))
            try:
                block_states_sub = rospy.Subscriber(
                    '/gazebo/model_states', ModelStates,
                    self._gazebo_update_block_states)
                self._block_states_subs.append(block_states_sub)
                # Must get the first msg from gazebo.
                rospy.wait_for_message(
                    '/gazebo/model_states', ModelStates, timeout=2)
            except rospy.ROSException as e:
                print(
                    "Topic /gazebo/model_states is not available, aborting...")
                print("Error message: ", e)
                exit()
            self._blocks.append(block)
        else:
            tf_topic = "/tf"            
            try:
                block_state_sub = rospy.Subscriber(tf_topic, tfMessage,
                    self._apriltag_update_block_state)
                self._block_states_subs.append(block_state_sub)
                # Must get the first msg.
                rospy.wait_for_message(tf_topic, tfMessage, timeout=2)
            except rospy.ROSException as e:
                print("Topic {} is not available, aborting...".format(tf_topic))
                print("Error message: ", e)
                exit()

            for i in range(self._num_of_blocks):
                block_name = 'block_{}'.format(i)
                block = Block(
                    name=block_name,
                    size=[0.11, 0.11, 0.13],
                    initial_pos=(0.55, 0., 0.03),
                    random_delta_range=0.15,
                    resource=block_name)
                self._blocks.append(block)

        # Add table to moveit
        # moveit needs a sleep before adding object
        rospy.sleep(1)
        pose_stamped_table = PoseStamped()
        pose_stamped_table.header.frame_id = self._frame_id
        pose_stamped_table.pose.position.x = 0.655
        pose_stamped_table.pose.position.y = 0
        # Leave redundant space
        pose_stamped_table.pose.position.z = -0.05
        pose_stamped_table.pose.orientation.x = 0
        pose_stamped_table.pose.orientation.y = 0
        pose_stamped_table.pose.orientation.z = 0
        pose_stamped_table.pose.orientation.w = 1.0
        self._moveit_scene.add_box('table', pose_stamped_table,
                                   (1.0, 0.9, 0.1))
        # Add calibration marker to moveit
        rospy.sleep(1)
        pose_stamped_marker = PoseStamped()
        pose_stamped_marker.header.frame_id = self._frame_id
        pose_stamped_marker.pose.position.x = 1.055
        pose_stamped_marker.pose.position.y = -0.404
        # Leave redundant space
        pose_stamped_marker.pose.position.z = 0.02
        pose_stamped_marker.pose.orientation.x = 0
        pose_stamped_marker.pose.orientation.y = 0
        pose_stamped_marker.pose.orientation.z = 0
        pose_stamped_marker.pose.orientation.w = 1.0
        self._moveit_scene.add_box('marker', pose_stamped_marker,
                                   (0.09, 0.08, 0.04))
        # Add blocks to moveit
        for block in self._blocks:
            rospy.sleep(1)
            pose_stamped_block = PoseStamped()
            pose_stamped_block.header.frame_id = self._frame_id
            pos = block.position
            pos.z += 0.0
            pose_stamped_block.pose.position = pos
            orientation = block.orientation
            pose_stamped_block.pose.orientation = orientation
            self._moveit_scene.add_box(
                block.name, pose_stamped_block,
                (block.size[0], block.size[1], block.size[2]))
            # add a top block to protect block
            # protect_block = copy.deepcopy(pose_stamped_block)
            # protect_block.pose.position.z += block.size[2] / 2 + 0.005 / 2
            # self._moveit_scene.add_box(
            #     block.name + '_protect', protect_block,
            #     (block.size[0] * 0.9, block.size[1] * 0.9, 0.005))
            # add the block to the allowed collision matrix
            rospy.sleep(1)
            self._moveit_scene_controller.add_object_to_acm(block.name)

    def _gazebo_update_block_states(self, data):
        model_states = data
        model_names = model_states.name
        for block in self._blocks:
            block_idx = model_names.index(block.name)
            block_pose = model_states.pose[block_idx]
            block.position = block_pose.position
            block.orientation = block_pose.orientation

            self._moveit_update_block(block)

    def _apriltag_update_block_state(self, data):
        robot_frame = "robot"

        if self._tf_listener.frameExists(robot_frame):     
            for block in self._blocks:                            
                block_frame = block.resource
                if self._tf_listener.frameExists(block_frame): 
                    translation_wrt_robot, orientation_wrt_robot = self._tf_listener.lookupTransform(
                        block_frame, robot_frame, rospy.Time(0))

                    translation_wrt_robot = Point(
                        x=translation_wrt_robot[0],
                        y=translation_wrt_robot[1],
                        z=translation_wrt_robot[2])

                    orientation_wrt_robot = Quaternion(
                        x=orientation_wrt_robot[0],
                        y=orientation_wrt_robot[1],
                        z=orientation_wrt_robot[2],
                        w=orientation_wrt_robot[3])
                    
                    # Use low pass filter to smooth data.
                    if block.first_smoothed:
                        block.position = translation_wrt_robot
                        block.position.x -= 0.035
                        block.position.y += 0.035
                        block.position.z = 0.065
                        block.orientation = orientation_wrt_robot
                        block.first_smoothed = False
                    else:
                        block.position.x = self._lowpass_filter(
                            translation_wrt_robot.x, block.position.x) - 0.035
                        block.position.y = self._lowpass_filter(
                            translation_wrt_robot.y, block.position.y) + 0.035
                        # block.position.z = self._lowpass_filter(
                        #     translation_wrt_robot.z, block.position.z)
                        block.position.z = 0.065
                        block.orientation.x = self._lowpass_filter(
                            orientation_wrt_robot.x, block.orientation.x)
                        block.orientation.y = self._lowpass_filter(
                            orientation_wrt_robot.y, block.orientation.y)
                        block.orientation.z = self._lowpass_filter(
                            orientation_wrt_robot.z, block.orientation.z)
                        block.orientation.w = self._lowpass_filter(
                            orientation_wrt_robot.w, block.orientation.w)
                    self._moveit_update_block(block)

    def _lowpass_filter(self, observed_value_p1, estimated_value):
        estimated_value_p1 = estimated_value + self._lowpass_alpha * (
            observed_value_p1 - estimated_value)
        return estimated_value_p1

    def _moveit_update_block(self, block):
        move_object = CollisionObject()
        move_object.id = block.name
        move_object.header.frame_id = self._frame_id
        pose = Pose()
        pose.position = block.position
        pose.position.z += 0.0
        pose.orientation = block.orientation
        move_object.primitive_poses.append(pose)
        move_object.operation = move_object.MOVE

        self._moveit_col_obj_pub.publish(move_object)
        #
        # move_object = CollisionObject()
        # move_object.id = block.name + '_protect'
        # move_object.header.frame_id = self._frame_id
        # pose = Pose()
        # pose.position = block.position
        # pose.position.z += block.size[2] / 2 + 0.005 / 2
        # pose.orientation = block.orientation
        # move_object.primitive_poses.append(pose)
        # move_object.operation = move_object.MOVE
        #
        # self._moveit_col_obj_pub.publish(move_object)

    def reset(self):
        if self._simulated:
            self._reset_sim()
        else:
            self._reset_real()

    def _reset_sim(self):
        """
        reset the simulation
        """
        # Randomize start position of blocks
        for block in self._blocks:
            block_random_delta = np.zeros(2)
            while np.linalg.norm(block_random_delta) < 0.1:
                block_random_delta = np.random.uniform(
                    -block.random_delta_range,
                    block.random_delta_range,
                    size=2)
            Gazebo.set_model_pose(
                block.name,
                new_pose=Pose(
                    position=Point(
                        x=block.initial_pos.x + block_random_delta[0],
                        y=block.initial_pos.y + block_random_delta[1],
                        z=block.initial_pos.z)))

    def _reset_real(self):
        """
        reset the real
        """
        # randomize start position of blocks
        for block in self._blocks:
            block_random_delta = np.zeros(2)
            new_pos = block.initial_pos
            while np.linalg.norm(block_random_delta) < 0.1:
                block_random_delta = np.random.uniform(
                    -block.random_delta_range,
                    block.random_delta_range,
                    size=2)
            new_pos.x += block_random_delta[0]
            new_pos.y += block_random_delta[1]
            logger.log('new position for {} is x = {}, y = {}, z = {}'.format(
                block.name, new_pos.x, new_pos.y, new_pos.z))
            ready = False
            while not ready:
                ans = input(
                    'Have you finished setting up {}?[Yes/No]\n'.format(
                        block.name))
                if ans.lower() == 'yes' or ans.lower() == 'y':
                    ready = True

    def terminate(self):
        for sub in self._block_states_subs:
            sub.unregister()

        self._moveit_col_obj_pub.unregister()

        if self._simulated:
            for block in self._blocks:
                Gazebo.delete_gazebo_model(block.name)
            Gazebo.delete_gazebo_model('table')
        else:
            ready = False
            while not ready:
                ans = input('Are you ready to exit?[Yes/No]\n')
                if ans.lower() == 'yes' or ans.lower() == 'y':
                    ready = True

        self._moveit_scene.remove_world_object('table')
        self._moveit_scene.remove_world_object('marker')

    def get_observation(self):
        blocks_pos = np.array([])
        blocks_ori = np.array([])

        for block in self._blocks:
            pos = np.array(
                [block.position.x, block.position.y, block.position.z])
            ori = np.array([
                block.orientation.x, block.orientation.y, block.orientation.z,
                block.orientation.w
            ])
            blocks_pos = np.concatenate((blocks_pos, pos))
            blocks_ori = np.concatenate((blocks_ori, ori))

        achieved_goal = np.squeeze(blocks_pos)

        obs = np.concatenate((blocks_pos, blocks_ori))

        Observation = collections.namedtuple('Observation',
                                             'obs achieved_goal')

        observation = Observation(obs=obs, achieved_goal=achieved_goal)

        return observation

    def get_block_orientation(self, name):
        for block in self._blocks:
            if block.name == name:
                return block.orientation
        raise NameError('No block named {}'.format(name))

    def get_block_position(self, name):
        for block in self._blocks:
            if block.name == name:
                return block.position
        raise NameError('No block named {}'.format(name))

    def get_blocks_orientation(self):
        orientations = []

        for block in self._blocks:
            orientations.append(block.orientation)

        return orientations

    def get_blocks_position(self):
        poses = []

        for block in self._blocks:
            poses.append(block.position)

        return poses

    @property
    def observation_space(self):
        return gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=self.get_observation().obs.shape,
            dtype=np.float32)

    def add_block(self, block):
        if self._simulated:
            Gazebo.load_gazebo_model(
                block.name, Pose(position=block.initial_pos), block.resource)
            # Waiting model to be loaded
            rospy.sleep(1)
        else:
            tf_topic = "/tf"
            self._block_states_subs.append(
                rospy.Subscriber(tf_topic, tfMessage,
                                 self._apriltag_update_block_state))
        self._blocks.append(block)
