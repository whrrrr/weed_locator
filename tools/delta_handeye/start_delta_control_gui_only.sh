#!/usr/bin/env bash
set -eo pipefail

cd /home/wyy/gpt_dev_ws
source /opt/ros/humble/setup.bash
source /home/wyy/gpt_dev_ws/install/setup.bash
export ROS_LOG_DIR=/tmp/ros_logs

exec ros2 run weed_locator delta_safe_control_gui
