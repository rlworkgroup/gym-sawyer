#!/bin/sh

USER_UID=$(id -u)
USER_GID=$(id -g)

xhost +local:root

DOCKER_VISUAL="-v /tmp/.X11-unix:/tmp/.X11-unix"
SAWYER_HOSTNAME="021707CP00056.local"
SAWYER_IP="192.168.33.7"


docker run \
    -it \
    --rm \
    --init \
    $DOCKER_VISUAL \
    --net="host" \
    --add-host="${SAWYER_HOSTNAME}:${SAWYER_IP}" \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --cap-add SYS_ADMIN \
    --cap-add MKNOD \
    --device /dev/fuse \
    --name "sawyer-ros-docker" \
    --security-opt apparmor:unconfined \
sawyer-ros-docker:cpu bash;

xhost -local:root
