#!/usr/bin/env bash
set -eo pipefail

cd /home/wyy/gpt_dev_ws

set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
set -u
export ROS_LOG_DIR=/tmp/ros_logs

HOME_SETTLE_SEC="${HOME_SETTLE_SEC:-6}"

echo "[1/3] making sure chess test stack is running..."
nodes="$(timeout 5 ros2 node list --no-daemon 2>/dev/null || true)"
if ! grep -qx "/chess_detector" <<<"$nodes" || ! grep -qx "/chess_handeye_target" <<<"$nodes"; then
  ./start_chess_test_detached.sh
  sleep 5
fi

echo "[2/3] homing delta arm..."
ros2 topic pub --once /delta_arm/home std_msgs/msg/Empty "{}" >/tmp/chess_home_command.out
sleep "$HOME_SETTLE_SEC"

echo "[3/3] moving to current selected chess target..."
exec ./reach_chess_once.sh
