#!/bin/bash

ORANGE='\033[0;33m'
WHITE='\033[1;37m'

cd "/root/"
mkdir dev
cd dev
pushd .

# Installs a CMake project from a git repository.
# Args: rel_target_folder_name print_name repository_url
function install_from_git {
	if [ ! -d "/root/dev/$1/build" ] && [ ! -f "/root/dev/$1/build" ]; then
		echo -e "${ORANGE}Building and installing $2.${WHITE}"
		git clone $3 $1 \
			&& cd $1 \
			&& mkdir build && cd build \
			&& cmake .. && make -j4 && make install
		popd
		pushd .
	else
		echo -e "${ORANGE}$2 already exists, just need to install.${WHITE}"
		cd "/root/dev/$1/build" && make install
		popd
		pushd .
	fi
}

# Installs a CMake project from a tar.gz archive.
# Args: rel_target_folder_name print_name tar_url
function install_from_targz {
	if [ ! -d "/root/dev/$1/build" ] && [ ! -f "/root/dev/$1/build" ]; then
		echo -e "${ORANGE}Building and installing $2.${WHITE}"
		wget -O $1.tar.gz $3 \
			&& tar xzf $1.tar.gz \
			&& rm $1.tar.gz \
			&& cd $1 \
			&& mkdir build && cd build \
			&& cmake .. && make -j4 && make install
		popd
		pushd .
	else
		echo -e "${ORANGE}$2 already exists, just need to install.${WHITE}"
		cd "/root/dev/$1/build" && make install
		popd
		pushd .
	fi
}

# Installs a CMake project from a zip archive.
# Args: rel_target_folder_name print_name tar_url
function install_from_zip {
	if [ ! -d "/root/dev/$1/build" ] && [ ! -f "/root/dev/$1/build" ]; then
		echo -e "${ORANGE}Building and installing $2.${WHITE}"
		wget -O $1.zip $3 \
			&& unzip $1.zip \
			&& rm $1.zip \
			&& cd $1 \
			&& mkdir build && cd build \
			&& cmake .. && make -j4 && make install
		popd
		pushd .
	else
		echo -e "${ORANGE}$2 already exists, just need to install.${WHITE}"
		cd "/root/dev/$1/build" && make install
		popd
		pushd .
	fi
}

# Install Eigen
install_from_targz "eigen-3.3.5" "Eigen 3.3.5" "http://bitbucket.org/eigen/eigen/get/3.3.5.tar.gz"

# Install libccd
install_from_git "libccd" "libccd" "https://github.com/danfis/libccd.git"

# Install octomap
install_from_git "octomap" "Octomap" "https://github.com/OctoMap/octomap.git"

# Install fcl
if [ ! -d "/root/dev/fcl/build" ] && [ ! -f "/root/dev/fcl/build" ]; then
	echo -e "${ORANGE}Building and installing fcl.${WHITE}"
	git clone https://github.com/flexible-collision-library/fcl.git \
		&& cd fcl && git checkout fcl-0.5 \
		&& mkdir build && cd build \
		&& cmake .. && make -j4 && make install
	popd
	pushd .
else
	echo -e "${ORANGE}fcl already exists, just need to install.${WHITE}"
	cd "/root/dev/fcl/build" && make install
	popd
	pushd .
fi

# Install OMPL
install_from_targz "ompl-1.3.1" "OMPL 1.3.1" "https://github.com/ompl/ompl/archive/1.3.1.tar.gz"

# Get source
if [ ! -d "root/ros_ws/src/object_recognition_msgs" ] && [ ! -f "/root/ros_ws/src/object_recognition_msgs" ]; then
  cd "/root/ros_ws/src"
  git clone https://github.com/wg-perception/object_recognition_msgs.git
fi

if [ ! -d "/root/ros_ws/src/octomap_msgs" ] && [ ! -f "/root/ros_ws/src/octomap_msgs" ]; then
  cd "/root/ros_ws/src"
  git clone https://github.com/OctoMap/octomap_msgs.git
  cd octomap_msgs
  git checkout indigo-devel
fi

if [ ! -d "/root/ros_ws/src/urdf_parser_py" ] && [ ! -f "/root/ros_ws/src/urdf_parser_py" ]; then
  cd "/root/ros_ws/src"
  git clone https://github.com/ros/urdf_parser_py.git
  cd urdf_parser_py
  git checkout indigo-devel
fi

if [ ! -d "/root/ros_ws/src/warehouse_ros" ] && [ ! -f "/root/ros_ws/src/warehouse_ros" ]; then
  cd "/root/ros_ws/src"
  git clone https://github.com/ros-planning/warehouse_ros.git
  cd warehouse_ros
  git checkout kinetic-devel
fi

if [ ! -d "/root/ros_ws/src/rviz_visual_tools" ] && [ ! -f "/root/ros_ws/src/rviz_visual_tools" ]; then
  cd "/root/ros_ws/ros_ws/src"
  git clone https://github.com/PickNikRobotics/rviz_visual_tools.git
  cd rviz_visual_tools
  git checkout kinetic-devel
fi

if [ ! -d "/root/ros_ws/src/graph_msgs" ] && [ ! -f "/root/ros_ws/src/graph_msgs" ]; then
  cd "/root/ros_ws/src"
  git clone https://github.com/davetcoleman/graph_msgs.git
  cd graph_msgs
  git checkout indigo-devel
fi

if [ ! -d "/root/ros_ws/src/moveit" ] && [ ! -f "/root/ros_ws/src/moveit" ]; then
  cd "/root/ros_ws/src"
  wstool merge moveit.rosinstall
  wstool update
  #rosdep install -y --from-paths src --ignore-src --rosdistro kinetic
  # Have to exclude some targets
  # See https://github.com/ros-planning/moveit/issues/697
  sed -i '/demo/s/^/# /g' "/root/ros_ws/src/moveit/moveit_ros/planning_interface/move_group_interface/CMakeLists.txt"
fi

# convert python2 code to python3
# /root/sawyer_2to3.sh "/root/ros_ws/src/"
# source /opt/ros/kinetic/setup.bash

export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH

# See https://github.com/osmcode/pyosmium/issues/52
# We need boost_python-py35
rm /usr/lib/x86_64-linux-gnu/libboost_python.so
ln -s /usr/lib/x86_64-linux-gnu/libboost_python-py35.so /usr/lib/x86_64-linux-gnu/libboost_python.so
