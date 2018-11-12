USER=sawyer_docker
ROS_WS=/home/$USER/ros_ws

#Copy test script inside docker container
docker cp "${1}" sawyer-ros-docker:/tmp;

#Run the test script
docker exec -it sawyer-ros-docker bash -c "cd $ROS_WS; $ROS_WS/intera.sh sim 'python3.6 /tmp/$(basename $1)'" > /dev/null
