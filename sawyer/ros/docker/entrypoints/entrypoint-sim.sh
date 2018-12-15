#!/bin/bash
export DOCKER_INTERNAL=/root/code/gym-sawyer/sawyer/ros/docker/internal

ln -fs /usr/bin/python2.7 /usr/bin/python
cp $DOCKER_INTERNAL/intera.sh $ROS_WS
cd $ROS_WS

#start gazebo sawyer
./intera.sh sim "export PYTHONPATH=/opt/ros/kinetic/lib/python2.7/dist-packages:$ROS_WS/devel/lib/python3/dist-packages && roslaunch sawyer_gazebo sawyer_learning.launch &"
sleep 5

#start sawyer moveit
./intera.sh sim "roslaunch sawyer_moveit_config sawyer_moveit.launch electric_gripper:=true &"
sleep 5

#change default python version to 3.6
ln -fs /usr/bin/python3.6 /usr/bin/python

#start sawyer moveit trajectory server
cp $DOCKER_INTERNAL/robot_enable.py $ROS_WS/src/intera_sdk/intera_interface/src/intera_interface/
./intera.sh sim "rosrun intera_interface enable_robot.py -e && rosrun intera_interface joint_trajectory_action_server.py &"
sleep 5

source "/home/$USER/.bashrc"

cd "/home/$USER"

bash
