
#!/bin/bash
export DOCKER_INTERNAL=/root/code/gym-sawyer/sawyer/ros/docker/internal

#change default python version to 3.6
ln -fs /usr/bin/python3.6 /usr/bin/python
cp $DOCKER_INTERNAL/intera.sh $ROS_WS
cd $ROS_WS

#start sawyer moveit trajectory server
cp $DOCKER_INTERNAL/robot_enable.py $ROS_WS/src/intera_sdk/intera_interface/src/intera_interface/
./intera.sh "rosrun intera_interface enable_robot.py -e && rosrun intera_interface joint_trajectory_action_server.py &"
sleep 5

ln -fs /usr/bin/python2.7 /usr/bin/python

#start sawyer moveit
./intera.sh "roslaunch sawyer_moveit_config sawyer_moveit.launch electric_gripper:=true &"
sleep 5

#remap from /robot/joint_states to /joint_states
./intera.sh "rosrun topic_tools relay /robot/joint_states /joint_states &"

source "/home/$USER/.bashrc"

cd "/home/$USER"

bash
