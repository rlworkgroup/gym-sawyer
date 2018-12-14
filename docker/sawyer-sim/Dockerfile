ARG PARENT_IMAGE=ubuntu:16.04
FROM $PARENT_IMAGE

# Install Gazebo. Instructions obtained from:
# http://sdk.rethinkrobotics.com/intera/Gazebo_Tutorial
RUN DEBIAN_FRONTEND=noninteractive apt update && apt -y install \
  gazebo7 \
  ros-kinetic-qt-build \
  ros-kinetic-gazebo-ros-control \
  ros-kinetic-gazebo-ros-pkgs \
  ros-kinetic-ros-control \
  ros-kinetic-control-toolbox \
  ros-kinetic-realtime-tools \
  ros-kinetic-ros-controllers \
  ros-kinetic-xacro \
  python-wstool \
  ros-kinetic-tf-conversions \
  ros-kinetic-kdl-parser \
  ros-kinetic-sns-ik-lib

# Install Sawyer Simulator.
RUN /bin/bash -c 'cd ~/ros_ws/src &&\
  git clone https://github.com/RethinkRobotics/sawyer_simulator.git &&\
  cd ~/ros_ws/src &&\
  wstool init . &&\
  wstool merge sawyer_simulator/sawyer_simulator.rosinstall &&\
  wstool update'

# Build Source
RUN /bin/bash -c 'source /opt/ros/kinetic/setup.bash &&\
  cd ~/ros_ws &&\
  catkin_make'

# Install and Build Sawyer Moveit Repo. Instructions obtained from:
# http://sdk.rethinkrobotics.com/intera/MoveIt_Tutorial
RUN /bin/bash -c 'cd ~/ros_ws/ &&\
  ./intera.sh &&\
  cd ~/ros_ws/src &&\
  wstool merge https://raw.githubusercontent.com/RethinkRobotics/sawyer_moveit/becef615db853e156b8444bb343127d99d97fddc/sawyer_moveit.rosinstall &&\
  wstool update &&\
  cd ~/ros_ws/ &&\
  source /opt/ros/kinetic/setup.bash &&\
  catkin_make'

COPY docker-entrypoint.sh /root/

ENTRYPOINT ["/root/docker-entrypoint.sh"]
