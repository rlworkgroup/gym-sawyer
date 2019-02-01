# gym-sawyer
Sawyer environments for reinforcement learning that use the OpenAI Gym
interface, as well as Dockerfiles with ROS to communicate with the real robot
or a simulated one with Gazebo.

This repository is under development, so all code is still experimental.

## Docker containers

### Sawyer Simulation

We use Gazebo to simulate Sawyer, so a dedicated GPU is required
([see System Requirements](http://gazebosim.org/tutorials?tut=guided_b1&cat=)).
Currently only NVIDIA GPUs are supported.

#### NVIDIA GPU

This section contains instructions to build the docker image and run the docker
container for the simulated Sawyer in the ROS environment using an NVIDIA GPU.

##### Prerequisites

- Install [Docker CE](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce)
- Install [Docker Compose](https://docs.docker.com/compose/install/#install-compose)
- Install the latest NVIDIA driver, tested
  on [nvidia-390](https://tecadmin.net/install-latest-nvidia-drivers-ubuntu/)
- [Install nvidia-docker2](https://github.com/NVIDIA/nvidia-docker#ubuntu-140416041804-debian-jessiestretch)
- Clone this repository in your local workspace

Tested on Ubuntu 16.04

##### Instructions

1. In the root folder of your cloned repository build the image by running:
  ```
  $ make build-nvidia-sawyer-sim
  ```
2. After the image is built, run the container:
  ```
  $ make run-nvidia-sawyer-sim
  ```
3. Gazebo and MoveIt! should open with Sawyer in them

4. To exit the container, type `sudo docker stop sawyer-sim` in a new terminal.

##### Control Sim Sawyer

1. Open a new terminal in the container by running:
  ```
  $ docker exec -it sawyer-sim bash
  ```
2. Once inside the terminal, run the following commands to execute a keyboard
  controller:
  ```
  $ cd ~/ros_ws
  $ ./intera.sh sim
  $ rosrun intera_examples joint_position_keyboard.py
  ```
3. The following message should appear:
  ```
  Initializing node...
  Getting robot state...
  [INFO] [1544554222.728405, 33.526000]: Enabling robot...
  [INFO] [1544554222.729527, 33.527000]: Robot Enabled
  Controlling joints. Press ? for help, Esc to quit.
  ```
4. Type ? to get the keys that control sawyer.

##### View the header camera in Sim Sawyer

1. Perform the first step from the previous section
2. Once in the container shell, run the following command:
  ```
  rosrun image_view image_view image:=/io/internal_camera/head_camera/image_raw
  ```
  
 ### Sawyer Robot  
A dedicated GPU is recommended for rviz and other visualization tools. Currently only NVIDIA GPUs are supported.

#### NVIDIA GPU

This section contains instructions to build the docker image and run the docker
container for the Sawyer robot in the ROS environment using an NVIDIA GPU.

##### Prerequisites

- Same as sawyer simulation.

##### Instructions

1. Export sawyer hostname, sawyer ip address, and workstation ip address.
  ```
  $ export SAWYER_HOSTNAME=__sawyerhostname__  
  $ export SAWYER_IP=__sawyerip__ 
  $ export WORKSTATION_IP=__workip__ 
  ```
2. In the root folder of your cloned repository build the image by running:
  ```
  $ make build-nvidia-sawyer-robot 
  ```
3. After the image is built, run the container:
  ```
  $ make run-nvidia-sawyer-robot
  ```
3. Rviz should open with Sawyer in it. Now you can plan and execute trajectories through rviz.

4. To exit the container, type `sudo docker stop sawyer-robot` in a new terminal.

### Garage-ROS-Intera

To run Reinforcement Learning algorithms along with the Sawyer robot, we use
the [garage](https://github.com/rlworkgroup/garage) docker images that include
an extensive library of utilities and primitives for RL experiments.

On top of the garage images, we add the layers for ROS and Intera that work
with Python3 (garage runs with Python3), so we're able to communicate with
Sawyer through ROS communication using the convenient libraries for Python
`rospy`, `intera_interface`, `moveit_commander` and `moveit_msgs`.

Under this schema, two docker containers are needed: one for the simulated or
real sawyer, and another with garage-ros-intera. The former creates the ROS
master while the latter subscribes to the ROS topics to control and visualize
what Sawyer is doing through Reinforcement Learning algorithms.

##### Prerequisites

- Install [Docker CE](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce). Tested
  on version 18.09.0.
- Install [Docker Compose](https://docs.docker.com/compose/install/#install-compose). Tested
  on version 1.23.2.

Tested on Ubuntu 16.04. It's recommended to use the versions indicated above
for docker-ce and docker-compose.

##### Instructions

In the root of the gym-sawyer repository execute:
```
$ make run-nvidia-sawyer-<type>
```
Where type can be `sim` (run simulated Sawyer on Gazebo) or `robot` (you're
connected to a real Sawyer).

Once the container is up and running (make sure Gazebo is fully initialized if
running simulated Sawyer). Then run:
```
$ make run-garage-<type>-ros RUN_CMD="examples/hello_world_sawyer.py"
```
Where type can be:
  - headless: garage without environment visualization.
  - nvidia: garage with environment visualization using an NVIDIA graphics
    card.
If your computer has an NVIDIA GPU, use this image to render the
environments in garage, and the pre-requisites are the same as for the Sawyer
Simulation image.

You should see Sawyer moving to neutral position and then waving its arm three
times.

The command to execute in the image is specified in the variable `RUN_CMD`.

###### Run your local repository of garage with ros-intera

If you're working with garage in your local repository and would like to
include your latests changes, follow these instructions.

1. Make sure to run build your docker image for garage first. For
  further information, visit [garage](ihttps://github.com/rlworkgroup/garage/blob/master/docker/README.md).
2. Then rebuild and run the garage-ros image:
  ```
  $ make run-garage-<type>-ros RUN_CMD="..."
  ```
