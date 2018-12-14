.PHONY: build-nvidia-sawyer-sim run-nvidia-sawyer-sim

build-nvidia-sawyer-sim: docker/docker-compose-nv-sim.yml docker/get_intera_sim.sh
	docker/get_intera_sim.sh
	docker-compose -f docker/docker-compose-nv-sim.yml build

run-nvidia-sawyer-sim: build-nvidia-sawyer-sim
	xhost +local:docker
	docker run \
		-it \
		--rm \
		--runtime=nvidia \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY="${DISPLAY}" \
		-e QT_X11_NO_MITSHM=1 \
		gym-sawyer/nvidia-sawyer-sim
