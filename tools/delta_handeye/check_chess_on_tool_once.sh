#!/usr/bin/env bash
set -eo pipefail

cd /home/wyy/gpt_dev_ws

set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
set -u
export ROS_LOG_DIR=/tmp/ros_logs

EXPECT_X_MM="${EXPECT_X_MM:-0.0}"
EXPECT_Y_MM="${EXPECT_Y_MM:-80.0}"
EXPECT_Z_MM="${EXPECT_Z_MM:--230.0}"
SAFE_Z_MM="${SAFE_Z_MM:--210.0}"
FEEDRATE="${FEEDRATE:-80.0}"
PIXEL_TIMEOUT_SEC="${PIXEL_TIMEOUT_SEC:-30}"
TARGET_TIMEOUT_SEC="${TARGET_TIMEOUT_SEC:-6}"
SETTLE_SEC="${SETTLE_SEC:-2.0}"
WAIT_FOR_ENTER="${WAIT_FOR_ENTER:-1}"
CHESS_X_OFFSET_MM="${CHESS_X_OFFSET_MM:-0.0}"
CHESS_Y_OFFSET_MM="${CHESS_Y_OFFSET_MM:-0.0}"
CHESS_Z_OFFSET_MM="${CHESS_Z_OFFSET_MM:-0.0}"

CAMERA_OUT="/tmp/chess_tool_camera_point_once.txt"
TARGET_OUT="/tmp/chess_tool_delta_target_once.txt"
STATUS_OUT="/tmp/chess_tool_status_once.txt"

ensure_delta_bridge() {
  local nodes
  nodes="$(timeout 5 ros2 node list --no-daemon 2>/dev/null || true)"
  if grep -qx "/delta_gcode_bridge" <<<"$nodes"; then
    return
  fi

  echo "delta_gcode_bridge is not running; starting it..."
  setsid -f bash -lc '
    cd /home/wyy/gpt_dev_ws
    set +u
    source /opt/ros/humble/setup.bash
    source install/setup.bash
    export ROS_LOG_DIR=/tmp/ros_logs
    ros2 run weed_locator delta_gcode_bridge --ros-args \
      -p port:=/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0 \
      > /home/wyy/gpt_dev_ws/delta_gcode_bridge_tool_check.log 2>&1
  '
  sleep 4
}

send_gcode_line() {
  local line="$1"
  echo "  $line"
  ros2 topic pub --once /delta_arm/gcode_raw std_msgs/msg/String "{data: '$line'}" >/tmp/chess_tool_move.out
  sleep 0.4
}

echo "[1/5] restarting chess test stack with offsets: X=${CHESS_X_OFFSET_MM} Y=${CHESS_Y_OFFSET_MM} Z=${CHESS_Z_OFFSET_MM}"
CHESS_X_OFFSET_MM="$CHESS_X_OFFSET_MM" CHESS_Y_OFFSET_MM="$CHESS_Y_OFFSET_MM" CHESS_Z_OFFSET_MM="$CHESS_Z_OFFSET_MM" ./start_chess_test_detached.sh
sleep 5

echo "[2/5] moving Delta to known visible pose: X=${EXPECT_X_MM} Y=${EXPECT_Y_MM} Z=${EXPECT_Z_MM}"
ensure_delta_bridge
send_gcode_line "G90"
send_gcode_line "G1 Z${SAFE_Z_MM} F${FEEDRATE}"
send_gcode_line "G1 X${EXPECT_X_MM} Y${EXPECT_Y_MM} Z${SAFE_Z_MM} F${FEEDRATE}"
send_gcode_line "G1 X${EXPECT_X_MM} Y${EXPECT_Y_MM} Z${EXPECT_Z_MM} F${FEEDRATE}"

echo "Put/fix the chess piece on the Delta end effector now if it is not already there."
if [ "$WAIT_FOR_ENTER" = "1" ]; then
  echo "Adjust the chess piece until show_chess_detection draws a box with SELECT."
  read -r -p "Press Enter after SELECT is visible..."
else
  echo "Waiting ${SETTLE_SEC}s..."
  sleep "$SETTLE_SEC"
fi

echo "[3/5] waiting for selected chess center..."
if ! timeout "$PIXEL_TIMEOUT_SEC" ros2 topic echo --once /chess/selected_pixel_center >/tmp/chess_tool_selected_pixel_once.txt 2>/tmp/chess_tool_selected_pixel_once.err; then
  echo "ERROR: no selected chess center."
  echo "Recent chess log:"
  tail -80 /home/wyy/gpt_dev_ws/local_chess_test.log 2>/dev/null || true
  exit 2
fi

echo "selected pixel:"
sed -n '1,12p' /tmp/chess_tool_selected_pixel_once.txt

echo "[4/5] computing hand-eye target without moving..."
rm -f "$CAMERA_OUT" "$TARGET_OUT" "$STATUS_OUT"
timeout "$TARGET_TIMEOUT_SEC" ros2 topic echo --once /chess/camera_point >"$CAMERA_OUT" 2>/tmp/chess_tool_camera_point_once.err &
camera_pid=$!
timeout "$TARGET_TIMEOUT_SEC" ros2 topic echo --once /chess/delta_target >"$TARGET_OUT" 2>/tmp/chess_tool_delta_target_once.err &
target_pid=$!
sleep 0.8
ros2 topic pub --once /chess/handeye_command std_msgs/msg/String "{data: capture}" >/tmp/chess_tool_capture_command.out

if wait "$camera_pid"; then
  echo "camera point:"
  cat "$CAMERA_OUT"
else
  echo "WARNING: no /chess/camera_point was published."
fi

if ! wait "$target_pid"; then
  echo "ERROR: no /chess/delta_target was published."
  echo "Recent chess log:"
  tail -80 /home/wyy/gpt_dev_ws/local_chess_test.log 2>/dev/null || true
  exit 3
fi

echo "computed delta target:"
cat "$TARGET_OUT"

echo "[5/5] error against known Delta pose:"
python3 - "$TARGET_OUT" "$EXPECT_X_MM" "$EXPECT_Y_MM" "$EXPECT_Z_MM" <<'PY'
import math
import re
import sys
from pathlib import Path

target_path = Path(sys.argv[1])
expected = [float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])]
text = target_path.read_text(encoding='utf-8')

values = {}
current = None
for line in text.splitlines():
    stripped = line.strip()
    if stripped in ("x:", "y:", "z:"):
        current = stripped[0]
        continue
    match = re.match(r"([xyz]):\s*([-+0-9.eE]+)", stripped)
    if match:
        values[match.group(1)] = float(match.group(2))

if not all(axis in values for axis in ("x", "y", "z")):
    raise SystemExit("could not parse delta target from " + str(target_path))

actual = [values["x"], values["y"], values["z"]]
err = [actual[i] - expected[i] for i in range(3)]
norm = math.sqrt(sum(v * v for v in err))

print("expected_mm=[%.1f, %.1f, %.1f]" % tuple(expected))
print("actual_mm=[%.1f, %.1f, %.1f]" % tuple(actual))
print("error_mm=[%.1f, %.1f, %.1f]  norm=%.1f" % (err[0], err[1], err[2], norm))
print("Suggested offsets to cancel this error:")
print("CHESS_X_OFFSET_MM=%.1f CHESS_Y_OFFSET_MM=%.1f CHESS_Z_OFFSET_MM=%.1f" % (-err[0], -err[1], -err[2]))
PY
