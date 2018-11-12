#!/bin/sh

USER_UID=$(id -u)
USER_GID=$(id -g)

xhost +local:root

if [ -z ${NVIDIA_DRIVER+x} ]; then
	NVIDIA_DRIVER=$(nvidia-settings -q NvidiaDriverVersion | head -2 | tail -1 | sed 's/.*\([0-9][0-9][0-9]\)\..*/\1/') ;
fi
if [ -z ${NVIDIA_DRIVER+x} ]; then
	echo "Error: Could not determine NVIDIA driver version number. Please specify your driver version number manually in $0." 1>&2 ;
	exit ;
else
	echo "Linking to NVIDIA driver version $NVIDIA_DRIVER..." ;
fi

DOCKER_VISUAL_NVIDIA="-v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/nvidia0 --device /dev/nvidiactl"

docker run \
	-it \
	--rm \
	--runtime=nvidia \
	--init \
	$DOCKER_VISUAL_NVIDIA \
	--env="DISPLAY" \
	--env="QT_X11_NO_MITSHM=1" \
	--cap-add SYS_ADMIN \
	--cap-add MKNOD \
	--device /dev/fuse \
	--name "sawyer-ros-docker" \
	--security-opt apparmor:unconfined \
sawyer-ros-docker:gpu bash;

xhost -local:root
