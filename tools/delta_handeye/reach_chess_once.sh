#!/usr/bin/env bash
set -eo pipefail

cd /home/wyy/gpt_dev_ws

set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
set -u
export ROS_LOG_DIR=/tmp/ros_logs

PIXEL_TIMEOUT_SEC="${PIXEL_TIMEOUT_SEC:-8}"
TARGET_TIMEOUT_SEC="${TARGET_TIMEOUT_SEC:-5}"
CAMERA_OUT="/tmp/chess_camera_point_once.txt"
TARGET_OUT="/tmp/chess_delta_target_once.txt"
STATUS_OUT="/tmp/chess_move_status_once.txt"

echo "[1/4] checking chess detector and hand-eye target nodes..."
nodes="$(timeout 5 ros2 node list --no-daemon 2>/dev/null || true)"
if ! grep -qx "/chess_detector" <<<"$nodes" || ! grep -qx "/chess_handeye_target" <<<"$nodes"; then
  echo "chess nodes are not running; starting local chess test stack..."
  ./start_chess_test_detached.sh
  sleep 4
fi

echo "[2/4] waiting for current selected chess center..."
if ! timeout "$PIXEL_TIMEOUT_SEC" ros2 topic echo --once /chess/selected_pixel_center >/tmp/chess_selected_pixel_once.txt 2>/tmp/chess_selected_pixel_once.err; then
  echo "no selected chess center yet; restarting local chess test stack once..."
  ./start_chess_test_detached.sh
  sleep 5
fi

if ! timeout "$PIXEL_TIMEOUT_SEC" ros2 topic echo --once /chess/selected_pixel_center >/tmp/chess_selected_pixel_once.txt 2>/tmp/chess_selected_pixel_once.err; then
  echo "ERROR: no selected chess center yet."
  echo "Look at the show_chess_detection window first. It must draw a box/center and SELECT on the chess piece."
  echo "Recent chess log:"
  tail -80 /home/wyy/gpt_dev_ws/local_chess_test.log 2>/dev/null || true
  exit 2
fi

echo "selected pixel:"
sed -n '1,12p' /tmp/chess_selected_pixel_once.txt

echo "[3/4] sending go command and waiting for converted delta target..."
rm -f "$CAMERA_OUT"
rm -f "$TARGET_OUT"
rm -f "$STATUS_OUT"
timeout "$TARGET_TIMEOUT_SEC" ros2 topic echo --once /chess/camera_point >"$CAMERA_OUT" 2>/tmp/chess_camera_point_once.err &
camera_pid=$!
timeout "$TARGET_TIMEOUT_SEC" ros2 topic echo --once /chess/delta_target >"$TARGET_OUT" 2>/tmp/chess_delta_target_once.err &
echo_pid=$!
timeout "$TARGET_TIMEOUT_SEC" ros2 topic echo --once /chess/move_status >"$STATUS_OUT" 2>/tmp/chess_move_status_once.err &
status_pid=$!

sleep 0.8
ros2 topic pub --once /chess/handeye_command std_msgs/msg/String "{data: go}" >/tmp/chess_go_command.out

if wait "$camera_pid"; then
  echo "camera point:"
  cat "$CAMERA_OUT"
else
  echo "WARNING: no /chess/camera_point was published."
fi

target_ok=1
if ! wait "$echo_pid"; then
  target_ok=0
fi

status_ok=1
if wait "$status_pid"; then
  echo "[4/4] move status:"
  cat "$STATUS_OUT"
else
  status_ok=0
  echo "[4/4] WARNING: no /chess/move_status was published."
fi

if [ "$target_ok" -eq 0 ] && [ "$status_ok" -eq 0 ]; then
  echo "ERROR: chess center existed, but no /chess/delta_target or /chess/move_status was published."
  echo "Likely causes: no valid depth at that pixel, missing camera_info, or invalid hand-eye file."
  echo "Recent chess log:"
  tail -80 /home/wyy/gpt_dev_ws/local_chess_test.log 2>/dev/null || true
  exit 3
fi

if grep -q "blocked:" "$STATUS_OUT" 2>/dev/null; then
  echo "Move was blocked by safety limits. Delta target was:"
  if [ "$target_ok" -eq 1 ]; then
    cat "$TARGET_OUT"
  else
    echo "(missed /chess/delta_target echo; see target_mm in move status above)"
  fi
  if grep -q "X .* outside" "$STATUS_OUT" 2>/dev/null; then
    echo
    echo "Hint: X is outside the safe range. Move the chess piece in the camera view:"
    echo "  X too negative -> move the piece to the RIGHT in the image."
    echo "  X too positive -> move the piece to the LEFT in the image."
  fi
  exit 4
fi

echo "Delta target was:"
if [ "$target_ok" -eq 1 ]; then
  cat "$TARGET_OUT"
else
  echo "(missed /chess/delta_target echo; move status above is authoritative)"
fi
