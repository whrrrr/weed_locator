#!/usr/bin/env bash
set -eo pipefail

cd /home/wyy/gpt_dev_ws

kill_ros_processes() {
  local patterns=(
    '[d]elta_safe_control_gui'
    '[l]ocal_delta_hand_eye.launch.py'
    '[d]elta_gcode_bridge'
    '[a]stra_camera_node'
    '[r]viz2'
    '[s]howimage'
    '[c]olored_pointcloud_from_rgbd'
    '[d]epth_image_visualizer'
  )

  for pattern in "${patterns[@]}"; do
    pkill -TERM -f "$pattern" 2>/dev/null || true
  done
  sleep 1
  for pattern in "${patterns[@]}"; do
    pkill -KILL -f "$pattern" 2>/dev/null || true
  done

  for executable in \
    delta_charuco_calibration \
    delta_safe_control_gui \
    delta_gcode_bridge \
    colored_pointcloud_from_rgbd \
    depth_image_visualizer
  do
    pgrep -f "/home/wyy/gpt_dev_ws/install/weed_locator/lib/weed_locator/${executable}" \
      | xargs -r kill -9 2>/dev/null || true
  done
}

kill_ros_processes

setsid -f bash -lc 'cd /home/wyy/gpt_dev_ws; source /opt/ros/humble/setup.bash; source install/setup.bash; export ROS_LOG_DIR=/tmp/ros_logs; ros2 launch weed_locator local_delta_hand_eye.launch.py port:=/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0 start_calibration:=true > /tmp/delta_handeye_launch.log 2>&1'
sleep 7
setsid -f bash -lc 'cd /home/wyy/gpt_dev_ws; source /opt/ros/humble/setup.bash; source install/setup.bash; export ROS_LOG_DIR=/tmp/ros_logs; ros2 run weed_locator delta_safe_control_gui > /tmp/delta_safe_control_gui.out 2>&1'

echo 'delta hand-eye stack restarted'
