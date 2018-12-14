#!/usr/bin/env bash

cd ~/ros_ws

LAUNCH="sawyer_gazebo sawyer_world.launch electric_gripper:=true &"
./intera.sh sim "roslaunch ${LAUNCH}"

LAUNCH="sawyer_moveit_config sawyer_moveit.launch electric_gripper:=true &"
./intera.sh sim "roslaunch --wait ${LAUNCH}"

./intera.sh sim
