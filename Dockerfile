ARG PARENT_IMAGE=rlworkgroup/garage-nvidia-ros-intera:latest
FROM $PARENT_IMAGE

RUN mkdir /root/code/gym-sawyer
COPY . /root/code/gym-sawyer

ENV PYTHONPATH=$PYTHONPATH:/root/code/gym-sawyer

# TEMP
RUN DEBIAN_FRONTEND=noninteractive apt -y install \
  tmux \
  vim

COPY ./docker/sawyer-robot/intera.sh /root/ros_ws
COPY ./docker/sawyer-robot/internal/temp/get_task_srv /root/ros_ws/src
RUN /bin/bash -c 'cd /root/ros_ws/ && \
  source activate garage && \
  source /opt/ros/kinetic/setup.bash && \
  catkin_make -DPYTHON_EXECUTABLE="$(which python)" --pkg get_task_srv'

