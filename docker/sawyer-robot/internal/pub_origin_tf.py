#!/usr/bin/python

import rospy
import tf2_ros
import roslaunch
import geometry_msgs.msg
import intera_interface
from tf import TransformListener

INITIAL_ROBOT_JOINT_POS = {
    'right_j0': -0.79328125, 
    'right_j1': 0.4875126953125,
    'right_j2': -2.848359375, 
    'right_j3': 0.9184482421875,
    'right_j4': -2.932041015625,
    'right_j5': -0.0460625,
    'right_j6': 0.0387646484375,
}

print("\nInitializing ROS Node\n")
rospy.init_node("pub_base_origin_tf")

print("\nMoving arm to detect origin\n")
limb = intera_interface.Limb('right')
limb.set_joint_position_speed(0.1)
limb.move_to_joint_positions(INITIAL_ROBOT_JOINT_POS, timeout=60.0)

hand_camera_frame = "right_hand_camera"
base_frame = "base"
base_duplicate_frame = "base_d"
origin_frame = "origin"
launch_file = "/root/origin_detection.launch"

print("\nLaunching apriltag detection\n")
uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_file])
launch.start()

print("\nWaiting to detect origin\n")
tf_listener = TransformListener()
tf_listener.waitForTransform(origin_frame, base_frame, rospy.Time(0), rospy.Duration(1000))
trans, quat = tf_listener.lookupTransform(origin_frame, base_frame, rospy.Time(0))

print("\nTerminating apriltag detection\n")
launch.shutdown()

print("Publishing transform between base and origin")
broadcaster = tf2_ros.StaticTransformBroadcaster()
static_transformStamped = geometry_msgs.msg.TransformStamped()

static_transformStamped.header.stamp = rospy.Time.now()
static_transformStamped.header.frame_id = origin_frame
static_transformStamped.child_frame_id = base_duplicate_frame #we donot use "base" frame to avoid changing sawyer's internal tf tree

static_transformStamped.transform.translation.x = trans[0]
static_transformStamped.transform.translation.y = trans[1]
static_transformStamped.transform.translation.z = trans[2]
static_transformStamped.transform.rotation.x = quat[0]
static_transformStamped.transform.rotation.y = quat[1]
static_transformStamped.transform.rotation.z = quat[2]
static_transformStamped.transform.rotation.w = quat[3]

broadcaster.sendTransform(static_transformStamped)
rospy.spin()

