ARG PARENT_IMAGE=rlworkgroup/garage-headless-ros-intera-py2:latest
FROM $PARENT_IMAGE

# Packages required in garage to run rospy and intera_interface
RUN /bin/bash -c 'source activate garage && \
  pip install \
    catkin_pkg \
    empy \
    rospkg'

# Install intera and geometry. The latter is mainly required by the
# ROS TF (transform) package.
COPY docker/garage.rosinstall /root/ros_ws/src
RUN /bin/bash -c 'cd /root/ros_ws/src && \
  wstool init . && \
  wstool merge garage.rosinstall && \
  wstool update'

# The target demo causes compilation errors and it cannot be disabled using
# EXCLUDE_FROM_ALL with CMake, so we're manually disabling it by commenting it
# from the corresponding CMakeList file.
RUN sed -i '/demo/s/^/# /g' "/root/ros_ws/src/moveit/moveit_ros/planning_interface/move_group_interface/CMakeLists.txt"

# Set libboost to python 3 before compiling ROS libraries
RUN /bin/bash -c 'rm /usr/lib/x86_64-linux-gnu/libboost_python.so && \
  ln -s /usr/lib/x86_64-linux-gnu/libboost_python-py35.so /usr/lib/x86_64-linux-gnu/libboost_python.so'

# Install any dependencies for intera, move it and geometry, and then compile
# them with the Python executable used by garage.
RUN /bin/bash -c 'cd /root/ros_ws/ && \
  apt -qy update && \
  rosdep update -qy && \
  rosdep install -qy --from-paths src/ --ignore-src --rosdistro kinetic && \
  source activate garage && \
  source /opt/ros/kinetic/setup.bash && \
  catkin_make --cmake-args -DPYTHON_EXECUTABLE="$(which python)" -DCATKIN_BLACKLIST_PACKAGES="moveit_setup_assistant" -DCMAKE_BUILD_TYPE=Release'

# cv2.so is installed with ros-kinetic-opencv3, but it's compiled for Python2,
# causing compatibility issues with Python3 in garage. We cannot  uninstall
# ros-kinetic-opencv3 since other packages rely on it, nor we can remove it
# from the PATH since other packages are imported from it. Since Python3 comes
# with its own compiled cv2, we're deleting the cv2 compiled for Python2.
RUN /bin/bash -c 'if [[ -f /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so ]]; then \
  rm /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so; fi'

# Although the intera and other libraries are now compiled for Python3, they
# still have the Python2 syntax. This script sets them all to Python3.
RUN /bin/bash -c 'packages=(moveit moveit_msgs intera_sdk); \
  for package in ${packages[@]}; do \
    python_scripts="$(find /root/ros_ws/src/${package} -name "*.py")"; \
    for script in ${python_scripts[@]}; do \
      2to3 -w ${script} --no-diffs > /dev/null 2>&1; \
    done; \
  done'

# Setup repo
WORKDIR /root/code/gym-sawyer
# Add code stub last
COPY . /root/code/gym-sawyer
