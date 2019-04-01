"""Sawyer Interface."""

import gym
from intera_core_msgs.msg import JointLimits
from geometry_msgs.msg import Pose, Point, Quaternion
import intera_interface
import moveit_msgs.msg
import numpy as np
import rospy
from tf import TransformListener

from sawyer.ros.robots.kinematics_interfaces import StateValidity
from sawyer.ros.robots.robot import Robot

INITIAL_JOINT_STATE = {
    'right_j0': -0.140923828125,
    'right_j1': -1.2789248046875,
    'right_j2': -3.043166015625,
    'right_j3': -2.139623046875,
    'right_j4': -0.047607421875,
    'right_j5': -0.7052822265625,
    'right_j6': -1.4102060546875,
}

class Sawyer(Robot):
    """Sawyer class."""

    def __init__(self,
                 moveit_group,
                 initial_joint_pos=INITIAL_JOINT_STATE,
                 control_mode='position',
                 extended_finger=True):
        """
        Sawyer class.

        :param initial_joint_pos: {str: float}
                        {'joint_name': position_value}, and also
                        initial_joint_pos should include all of the
                        joints that user wants to control and observe.
        :param moveit_group: str
                        Use this to check safety
        :param control_mode: string
                        robot control mode: 'position' or velocity
                        or effort
        """
        Robot.__init__(self)
        self._limb = intera_interface.Limb('right')
        self._gripper = intera_interface.Gripper()
        self._initial_joint_pos = initial_joint_pos
        self._control_mode = control_mode
        self._used_joints = []
        for joint in initial_joint_pos:
            self._used_joints.append(joint)
        self._joint_limits = rospy.wait_for_message('/robot/joint_limits',
                                                    JointLimits)
        self._moveit_group = moveit_group
        self._tf_listener = TransformListener()
        self._base_frame = "base"
        if extended_finger:
            self._tip_frame = "right_gripper_tip_ex"
        else:
            self._tip_frame = "right_gripper_tip"

        self._sv = StateValidity()

    def safety_check(self):
        """
        If robot is in safe state.

        :return safe: Bool
                if robot is safe.
        """
        rs = moveit_msgs.msg.RobotState()
        current_joint_angles = self._limb.joint_angles()
        for joint in current_joint_angles:
            rs.joint_state.name.append(joint)
            rs.joint_state.position.append(current_joint_angles[joint])
        result = self._sv.get_state_validity(rs, self._moveit_group)
        return result.valid

    def safety_predict(self, joint_angles):
        """
        Will robot be in safe state.

        :param joint_angles: {'': float}
        :return safe: Bool
                    if robot is safe.
        """
        rs = moveit_msgs.msg.RobotState()
        for joint in joint_angles:
            rs.joint_state.name.append(joint)
            rs.joint_state.position.append(joint_angles[joint])
        result = self._sv.get_state_validity(rs, self._moveit_group)
        return result.valid

    def in_collision_state(self):
        return self._limb.has_collided()

    @property
    def enabled(self):
        """
        If robot is enabled.

        :return: if robot is enabled.
        """
        return intera_interface.RobotEnable(
            intera_interface.CHECK_VERSION).state().enabled

    def reset(self):
        """Reset sawyer."""
        self._move_to_start_position()

    def get_observation(self):
        """
        Get robot observation.

        :return: robot observation
        """
        return {
            'gripper_position': np.array(self.gripper_pose['position']),
            'gripper_state': np.array([self.gripper_state]),
            'gripper_orientation': np.array(self.gripper_pose['orientation']),
            'gripper_lvel': np.array(self._limb.endpoint_velocity()['linear']),
            'gripper_avel': np.array(self._limb.endpoint_velocity()['angular']),
            'gripper_force': np.array(self._limb.endpoint_effort()['force']),
            'gripper_torque': np.array(self._limb.endpoint_effort()['torque']),
            'robot_joint_angles': np.array(list(self._limb.joint_angles().values())),
            'robot_joint_velocities': np.array(list(self._limb.joint_velocities().values())),
            'robot_joint_efforts': np.array(list(self._limb.joint_efforts().values()))
        }

    @property
    def observation_space(self):
        """
        Observation space.

        :return: gym.spaces
                    observation space
        """
        return gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(len(self.get_observation()),),
            dtype=np.float32)

    def send_command(self, commands):
        """
        Send command to sawyer.

        :param commands: [float]
                    list of command for different joints and gripper
        """
        is_joint_commands = (len(commands) == (len(self._used_joints) + 1)) # +1 for gripper
        commands = np.clip(commands, self.action_space.low, self.action_space.high)

        if is_joint_commands:
            i = 0
            joint_commands = {}
            for joint in self._used_joints:
                joint_commands[joint] = commands[i]
                i += 1

            if self._control_mode == 'position':
                self._set_limb_joint_positions(joint_commands)
            elif self._control_mode == 'velocity':
                self._set_limb_joint_velocities(joint_commands)
            elif self._control_mode == 'effort':
                self._set_limb_joint_torques(joint_commands)
            
            self._set_gripper_state(commands[7])

        else: # Position command
            if self._control_mode == 'task_space':                
                self._set_gripper_end_pose(commands[:3])
            
            # Rescale gripper state
            gripper_state =  self._gripper.MAX_POSITION if commands[3] > 0 else self._gripper.MIN_POSITION
            self._set_gripper_state(gripper_state)                    

    @property
    def gripper_pose(self):
        """
        Get the gripper pose.

        :return: gripper pose
        """
        gripper_pos, gripper_ori = self._get_tf_between(self._base_frame, self._tip_frame)
        gripper_pose = { 'position': gripper_pos, 'orientation': gripper_ori }
        return gripper_pose

    @property
    def action_space(self):
        """
        Return a Space object.

        :return: action space
        """
        lower_bounds = np.array([])
        upper_bounds = np.array([])
        action_space = None
        if self._control_mode == 'task_space':
            limit = np.array([0.01, 0.01, 0.01, 1.])
            lower_bounds = np.concatenate((lower_bounds, -limit)) 
            upper_bounds = np.concatenate((upper_bounds, limit))
            action_space = gym.spaces.Box(lower_bounds, upper_bounds, dtype=np.float32)
        else:
            for joint in self._used_joints:
                joint_idx = self._joint_limits.joint_names.index(joint)
                if self._control_mode == 'position':
                    lower_bounds = np.concatenate(
                        (lower_bounds,
                         np.array(self._joint_limits.position_lower[
                             joint_idx:joint_idx + 1])))
                    upper_bounds = np.concatenate(
                        (upper_bounds,
                         np.array(self._joint_limits.position_upper[
                             joint_idx:joint_idx + 1])))
                elif self._control_mode == 'velocity':
                    velocity_limit = np.array(
                        self._joint_limits.velocity[joint_idx:joint_idx + 1]) * 0.1
                    lower_bounds = np.concatenate((lower_bounds, -velocity_limit))
                    upper_bounds = np.concatenate((upper_bounds, velocity_limit))
                elif self._control_mode == 'effort':
                    effort_limit = np.array(
                        self._joint_limits.effort[joint_idx:joint_idx + 1])
                    lower_bounds = np.concatenate((lower_bounds, -effort_limit))
                    upper_bounds = np.concatenate((upper_bounds, effort_limit))
                else:
                    raise ValueError(
                        'Control mode %s is not known!' % self._control_mode)

                action_space = gym.spaces.Box(
                    np.concatenate((lower_bounds, np.array([0]))),
                    np.concatenate((upper_bounds, np.array([100]))),
                    dtype=np.float32)

        return action_space

    @property
    def gripper_state(self):
        ori_position = self._gripper.get_position()
        if self._control_mode == 'task_space':
            return self._rescale_value(ori_position, self._gripper.MIN_POSITION, self._gripper.MAX_POSITION,
                self.action_space.low[3], self.action_space.high[3])
        return ori_position

    def _set_limb_joint_positions(self, joint_angle_cmds):
        # limit joint angles cmd
        current_joint_angles = self._limb.joint_angles()
        for joint in joint_angle_cmds:
            joint_cmd_delta = joint_angle_cmds[joint] - \
                              current_joint_angles[joint]
            joint_angle_cmds[
                joint] = current_joint_angles[joint] + joint_cmd_delta * 0.1

        if self.safety_predict(joint_angle_cmds):
            self._limb.set_joint_positions(joint_angle_cmds)

    def _set_limb_joint_velocities(self, joint_angle_cmds):
        self._limb.set_joint_velocities(joint_angle_cmds)

    def _set_limb_joint_torques(self, joint_angle_cmds):
        self._limb.set_joint_torques(joint_angle_cmds)

    def _set_gripper_state(self, position):
        self._gripper.set_position(position)
    
    def _set_gripper_end_pose(self, gripper_pose_delta):
        cur_pos, cur_ori = self._get_tf_between(self._base_frame, self._tip_frame)
        new_pos = cur_pos + np.array(gripper_pose_delta)
        
        # ik_request returns valid joint positions if exists, 
        # otherwise returns False.        
        pose_msg = Pose()
        pose_msg.position = Point(new_pos[0], new_pos[1], new_pos[2])
        pose_msg.orientation = Quaternion(cur_ori[0], cur_ori[1], cur_ori[2], cur_ori[3])        
        joint_angles = self._limb.ik_request(pose_msg, self._tip_frame)
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles, timeout=0.001)

    def _move_to_start_position(self):
        if rospy.is_shutdown():
            return
        self._limb.move_to_joint_positions(self._initial_joint_pos, timeout=60.0)
        self._gripper.close()
        rospy.sleep(1.0)

    def _rescale_value(self, value, cur_min, cur_max, new_range_min, new_range_max):
        rescaled_value = (((new_range_max - new_range_min) * (
                            value - cur_min)) / (cur_max - cur_min)) + new_range_min

        return rescaled_value 

    def _get_tf_between(self, frame1, frame2):
        self._tf_listener.waitForTransform(
            frame1, frame2, rospy.Time(0), rospy.Duration(2))
        pos, ori = self._tf_listener.lookupTransform(
            frame1, frame2, rospy.Time(0))
        return pos, ori
