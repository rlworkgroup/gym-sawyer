ARG PARENT_IMAGE=ubuntu:16.04
FROM $PARENT_IMAGE

# Build Source
RUN ["/bin/bash", "-c", \
 "source /opt/ros/kinetic/setup.bash && \
  cd ~/ros_ws && \
  catkin_make"]

# Copy the modified intera script configured for ros-kinetic
COPY intera.sh /root/ros_ws/intera.sh

# Install and Build Sawyer Moveit Repo. Instructions obtained from:
# http://sdk.rethinkrobotics.com/intera/MoveIt_Tutorial

RUN ["/bin/bash", "-c", \
  "cd ~/ros_ws/ && \
  ./intera.sh && \
  cd ~/ros_ws/src && \
  wstool init . && \
  wstool merge https://raw.githubusercontent.com/RethinkRobotics/sawyer_moveit/becef615db853e156b8444bb343127d99d97fddc/sawyer_moveit.rosinstall && \
  wstool update && \
  cd ~/ros_ws/ && \
  source /opt/ros/kinetic/setup.bash && \
  catkin_make"]

COPY docker-entrypoint.sh /root/
COPY sawyer-robot.launch /root/

ENTRYPOINT ["/root/docker-entrypoint.sh"]
