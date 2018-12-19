.PHONY: build-nvidia-sawyer-sim run-nvidia-sawyer-sim build-nvidia-sawyer-robot run-nvidia-sawyer-robot

build-nvidia-sawyer-sim: docker/docker-compose-nv-sim.yml docker/get_intera.sh
	docker/get_intera.sh --sim
	docker-compose -f docker/docker-compose-nv-sim.yml build

build-nvidia-sawyer-robot: docker/docker-compose-nv-robot.yml docker/get_intera.sh
	docker/get_intera.sh
	docker-compose -f docker/docker-compose-nv-robot.yml build

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
		--name "sawyer-sim" \
		gym-sawyer/nvidia-sawyer-sim

run-nvidia-sawyer-robot: build-nvidia-sawyer-robot
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
		--add-host="${SAWYER_HOSTNAME}:${SAWYER_IP}" \
		--name "sawyer-robot" \
		gym-sawyer/nvidia-sawyer-robot
