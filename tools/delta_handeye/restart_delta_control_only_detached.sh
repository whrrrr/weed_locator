#!/usr/bin/env bash
set -eo pipefail

cd /home/wyy/gpt_dev_ws

kill_ros_processes() {
  local patterns=(
    '[d]elta_safe_control_gui'
    '[d]elta_gcode_bridge'
    '[l]ocal_delta_hand_eye.launch.py'
    '[d]elta_charuco_calibration'
    '[a]stra_camera_node'
    '[c]olored_pointcloud_from_rgbd'
    '[d]epth_image_visualizer'
    '[r]viz2'
    '[s]howimage'
  )

  for pattern in "${patterns[@]}"; do
    pkill -TERM -f "$pattern" 2>/dev/null || true
  done
  sleep 1
  for pattern in "${patterns[@]}"; do
    pkill -KILL -f "$pattern" 2>/dev/null || true
  done

  for executable in \
    delta_safe_control_gui \
    delta_gcode_bridge
  do
    pgrep -f "/home/wyy/gpt_dev_ws/install/weed_locator/lib/weed_locator/${executable}" \
      | xargs -r kill -9 2>/dev/null || true
  done
}

kill_ros_processes

setsid -f bash -lc 'cd /home/wyy/gpt_dev_ws; source /opt/ros/humble/setup.bash; source install/setup.bash; export ROS_LOG_DIR=/tmp/ros_logs; ros2 run weed_locator delta_gcode_bridge --ros-args -p port:=/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0 > /tmp/delta_gcode_bridge_control_only.log 2>&1'
sleep 3
setsid -f bash -lc 'cd /home/wyy/gpt_dev_ws; source /opt/ros/humble/setup.bash; source install/setup.bash; export ROS_LOG_DIR=/tmp/ros_logs; ros2 run weed_locator delta_safe_control_gui > /tmp/delta_safe_control_gui.out 2>&1'

echo 'delta control-only stack restarted'
