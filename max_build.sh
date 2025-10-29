#!/bin/bash

echo "RUNNING BUILD!"

sudo apt update --fix-missing

rosdep update && rosdep install --from-paths ${ISAAC_ROS_WS}/src --ignore-src -y

colcon build  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=False
