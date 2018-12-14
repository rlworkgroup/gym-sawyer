# gym-sawyer
Sawyer environments for reinforcement learning that use the OpenAI Gym
interface, as well as Dockerfiles with ROS to communicate with the real robot
or a simulated one with Gazebo.

This repository is under development, so all code is still experimental.

## Docker containers

### Simulate Sawyer

We use Gazebo to simulate Sawyer, so a dedicated GPU is required
([see System Requirements](http://gazebosim.org/tutorials?tut=guided_b1&cat=)).
Currently only NVIDIA GPUs are supported.

#### NVIDIA GPU

This section contains instructions to build the docker image and run the docker
container for the simulated Sawyer in the ROS environment using an NVIDIA GPU.

##### Prerequisites

- Ubuntu 16.04.
- Install [Docker CE](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce)
- Install [Docker Compose](https://docs.docker.com/compose/install/#install-compose)
- Install the latest NVIDIA driver, tested
  on [nvidia-390](https://tecadmin.net/install-latest-nvidia-drivers-ubuntu/)
- [Install nvidia-docker2](https://github.com/NVIDIA/nvidia-docker#ubuntu-140416041804-debian-jessiestretch)
- Clone this repository in your local workspace

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

4. To exit the container, type `exit` in the container shell

##### Control Sim Sawyer

1. Open a new terminal and find the NAME property of the container with:
  ```
  $ docker container ps
  ```
2. Let's open a new terminal in the container by running:
  ```
  $ docker exec -it <container_name> bash
  ```
3. Once inside the terminal, run the following commands to execute a keyboard
  controller:
  ```
  $ cd ~/ros_ws
  $ ./intera.sh sim
  $ rosrun intera_examples joint_position_keyboard.py
  ```
4. The following message should appear:
  ```
  Initializing node...
  Getting robot state...
  [INFO] [1544554222.728405, 33.526000]: Enabling robot...
  [INFO] [1544554222.729527, 33.527000]: Robot Enabled
  Controlling joints. Press ? for help, Esc to quit.
  ```
5. Type ? to get the keys that control sawyer.

##### View the header camera in Sim Sawyer

1. Perform the first two steps from the previous section
2. Once in the container shell, run the following command:
  ```
  rosrun image_view image_view image:=/io/internal_camera/head_camera/image_raw
  ```
