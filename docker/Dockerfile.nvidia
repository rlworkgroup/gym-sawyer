# Install NVIDIA/OpenGL image. For more information see:
# https://hub.docker.com/r/nvidia/opengl/
FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu16.04

ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES},display

RUN apt-get update && apt-get install -y --no-install-recommends \
        mesa-utils && \
    rm -rf /var/lib/apt/lists/*
