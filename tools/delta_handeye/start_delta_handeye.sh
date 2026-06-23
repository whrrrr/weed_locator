#!/usr/bin/env bash
set -eo pipefail

cd /home/wyy/gpt_dev_ws

echo "[1/3] stopping old ROS/camera/viewer processes..."
pkill -f "local_delta_hand_eye.launch.py" 2>/dev/null || true
pkill -f "astra_camera_node" 2>/dev/null || true
pkill -f "delta_charuco_calibration" 2>/dev/null || true
pkill -f "delta_gcode_bridge" 2>/dev/null || true
pkill -f "colored_pointcloud_from_rgbd" 2>/dev/null || true
pkill -f "depth_image_visualizer" 2>/dev/null || true
pkill -f "opencv_camera_publisher" 2>/dev/null || true
pkill -f "rviz2" 2>/dev/null || true
pkill -f "showimage" 2>/dev/null || true
sleep 1

echo "[2/3] sourcing ROS workspace..."
source /opt/ros/humble/setup.bash
source /home/wyy/gpt_dev_ws/install/setup.bash
export ROS_LOG_DIR=/tmp/ros_logs

echo "[3/3] launching camera, Delta bridge, colored point cloud, RViz, OpenCV debug view, depth view..."
exec ros2 launch weed_locator local_delta_hand_eye.launch.py port:=/dev/ttyUSB0
