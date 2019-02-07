.PHONY: build-garage-headless-ros run-garage-headless-ros \
	build-garage-nvidia-ros run-garage-nvidia-ros \
	build-nvidia-sawyer-sim run-nvidia-sawyer-sim \
	build-nvidia-sawyer-robot run-nvidia-sawyer-robot

# Path in host where the experiment data obtained in the container is stored
DATA_PATH ?= $(shell pwd)/data
# Set the environment variable MJKEY with the contents of the file specified by
# MJKEY_PATH.
MJKEY_PATH ?= ~/.mujoco/mjkey.txt
# Sets the add-host argument used to connect to the Sawyer ROS master
SAWYER_NET = "$(SAWYER_HOSTNAME):$(SAWYER_IP)"
ifneq (":", $(SAWYER_NET))
	ADD_HOST=--add-host=$(SAWYER_NET)
endif

build-garage-headless-ros: docker/docker-compose-garage-headless-ros.yml
	docker-compose -f docker/docker-compose-garage-headless-ros.yml build

build-garage-nvidia-ros: docker/docker-compose-garage-nvidia-ros.yml
	docker-compose -f docker/docker-compose-garage-nvidia-ros.yml build

build-nvidia-sawyer-sim: docker/docker-compose-nv-sim.yml docker/get_intera.sh
	docker/get_intera.sh --sim
	docker-compose -f docker/docker-compose-nv-sim.yml build

build-nvidia-sawyer-robot: docker/docker-compose-nv-robot.yml docker/get_intera.sh
	docker/get_intera.sh
	docker-compose -f docker/docker-compose-nv-robot.yml build nvidia-sawyer-robot

build-nvidia-robot-apriltag: docker/docker-compose-nv-robot.yml docker/get_intera.sh
	docker/get_intera.sh
	docker-compose -f docker/docker-compose-nv-robot.yml build nvidia-robot-apriltag

run-garage-headless-ros: CONTAINER_NAME ?= garage-headless-ros
run-garage-headless-ros: build-garage-headless-ros
	docker run \
		--init \
		-it \
		--rm \
		--net="host" \
		$(ADD_HOST) \
		-v $(DATA_PATH)/$(CONTAINER_NAME):/root/code/garage/data \
		-e MJKEY="$$(cat $(MJKEY_PATH))" \
		--name $(CONTAINER_NAME) \
		$(ADD_ARGS) \
		rlworkgroup/garage-headless-ros $(RUN_CMD)

run-garage-nvidia-ros: CONTAINER_NAME ?= garage-nvidia-ros
run-garage-nvidia-ros: build-garage-nvidia-ros
	xhost +local:docker
	docker run \
		--init \
		-it \
		--rm \
		--runtime=nvidia \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY="${DISPLAY}" \
		-e QT_X11_NO_MITSHM=1 \
		--net="host" \
		$(ADD_HOST) \
		-v $(DATA_PATH)/$(CONTAINER_NAME):/root/code/garage/data \
		-e MJKEY="$$(cat $(MJKEY_PATH))" \
		--name $(CONTAINER_NAME) \
		$(ADD_ARGS) \
		rlworkgroup/garage-nvidia-ros $(RUN_CMD)

run-nvidia-sawyer-sim: build-nvidia-sawyer-sim
	xhost +local:docker
	docker run \
		--init \
		-t \
		--rm \
		--runtime=nvidia \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY="${DISPLAY}" \
		-e QT_X11_NO_MITSHM=1 \
		--net="host" \
		--name "sawyer-sim" \
		gym-sawyer/nvidia-sawyer-sim

run-nvidia-sawyer-robot: build-nvidia-sawyer-robot
ifeq (,$(ADD_HOST))
	$(error Set the environment variables SAWYER_HOST and SAWYER_IP)
endif
	xhost +local:docker
	docker run \
		--init \
		-t \
		--rm \
		--runtime=nvidia \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY="${DISPLAY}" \
		-e QT_X11_NO_MITSHM=1 \
		--net="host" \
		$(ADD_HOST) \
		--name "sawyer-robot" \
		gym-sawyer/nvidia-sawyer-robot

run-nvidia-robot-apriltag: build-nvidia-robot-apriltag
ifeq (,$(ADD_HOST))
	$(error Set the environment variables SAWYER_HOST and SAWYER_IP)
endif
	xhost +local:docker
	docker run \
		--init \
		-t \
		--rm \
        --privileged \
		--runtime=nvidia \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY="${DISPLAY}" \
		-e QT_X11_NO_MITSHM=1 \
		--net="host" \
		$(ADD_HOST) \
        --volume=/dev/bus/usb:/dev/bus/usb:ro \
		--name "sawyer-robot-apriltag" \
		gym-sawyer/nvidia-robot-apriltag
