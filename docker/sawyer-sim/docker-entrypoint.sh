#!/usr/bin/env bash

cd ~/ros_ws

./intera.sh sim "roslaunch /root/sawyer-sim.launch &"

./intera.sh sim
