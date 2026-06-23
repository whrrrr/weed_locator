#!/usr/bin/env bash
set -eo pipefail

cd /home/wyy/gpt_dev_ws

cleanup_patterns=(
  "[r]os2 launch weed_locator local_chess_test.launch.py"
  "[r]os2 run weed_locator chess_test_gui"
  "bash -lc .*local_chess_test.launch.py"
  "bash -lc .*chess_test_gui"
  "[/]weed_locator/chess_detector"
  "[/]weed_locator/chess_handeye_target"
  "[/]weed_locator/delta_gcode_bridge"
  "[/]weed_locator/chess_test_gui"
  "[a]stra_camera_node"
  "[g]emini.launch.xml"
  "[s]how_chess_detection"
  "[/]image_tools/showimage"
)

for pattern in "${cleanup_patterns[@]}"; do
  pkill -TERM -f "$pattern" 2>/dev/null || true
done
sleep 1
for pattern in "${cleanup_patterns[@]}"; do
  pkill -KILL -f "$pattern" 2>/dev/null || true
done

set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
set -u
export ROS_LOG_DIR=/tmp/ros_logs

if [ "${CHESS_X_OFFSET_MM+x}" ]; then
  CHESS_X_OFFSET_MM_EXPLICIT="$CHESS_X_OFFSET_MM"
fi
if [ "${CHESS_Y_OFFSET_MM+x}" ]; then
  CHESS_Y_OFFSET_MM_EXPLICIT="$CHESS_Y_OFFSET_MM"
fi
if [ "${CHESS_Z_OFFSET_MM+x}" ]; then
  CHESS_Z_OFFSET_MM_EXPLICIT="$CHESS_Z_OFFSET_MM"
fi

if [ -f /home/wyy/gpt_dev_ws/chess_offsets.env ]; then
  # shellcheck disable=SC1091
  source /home/wyy/gpt_dev_ws/chess_offsets.env
fi

if [ "${CHESS_X_OFFSET_MM_EXPLICIT+x}" ]; then
  CHESS_X_OFFSET_MM="$CHESS_X_OFFSET_MM_EXPLICIT"
fi
if [ "${CHESS_Y_OFFSET_MM_EXPLICIT+x}" ]; then
  CHESS_Y_OFFSET_MM="$CHESS_Y_OFFSET_MM_EXPLICIT"
fi
if [ "${CHESS_Z_OFFSET_MM_EXPLICIT+x}" ]; then
  CHESS_Z_OFFSET_MM="$CHESS_Z_OFFSET_MM_EXPLICIT"
fi

CHESS_X_OFFSET_MM="${CHESS_X_OFFSET_MM:-0.0}"
CHESS_Y_OFFSET_MM="${CHESS_Y_OFFSET_MM:-0.0}"
CHESS_Z_OFFSET_MM="${CHESS_Z_OFFSET_MM:-0.0}"
CHESS_TARGET_Z_MM="${CHESS_TARGET_Z_MM:--230.0}"
DELTA_PORT="${DELTA_PORT:-/dev/ttyUSB0}"
DELTA_BAUDRATE="${DELTA_BAUDRATE:-115200}"

if [ -z "${CHESS_HANDEYE_PATH:-}" ]; then
  latest_run="$(find /home/wyy/gpt_dev_ws/calibration_targets/recalibration_runs -mindepth 1 -maxdepth 1 -type d -name 'handeye_5layer_*' 2>/dev/null | sort | tail -n 1)"
  if [ -n "$latest_run" ] && [ -f "$latest_run/z230/delta_hand_eye.yaml" ]; then
    CHESS_HANDEYE_PATH="$latest_run/z230/delta_hand_eye.yaml"
  else
    CHESS_HANDEYE_PATH="/home/wyy/gpt_dev_ws/calibration_targets/delta_hand_eye.yaml"
  fi
fi

setsid -f bash -lc '
  cd /home/wyy/gpt_dev_ws
  set +u
  source /opt/ros/humble/setup.bash
  source install/setup.bash
  export ROS_LOG_DIR=/tmp/ros_logs
  ros2 launch weed_locator local_chess_test.launch.py \
    start_camera:=true \
    start_delta_bridge:=true \
    port:='"$DELTA_PORT"' \
    baudrate:='"$DELTA_BAUDRATE"' \
    handeye_path:='"$CHESS_HANDEYE_PATH"' \
    target_z_override_mm:='"$CHESS_TARGET_Z_MM"' \
    target_x_offset_mm:='"$CHESS_X_OFFSET_MM"' \
    target_y_offset_mm:='"$CHESS_Y_OFFSET_MM"' \
    target_z_offset_mm:='"$CHESS_Z_OFFSET_MM"' \
    > /home/wyy/gpt_dev_ws/local_chess_test.log 2>&1
'

sleep 3
setsid -f bash -lc '
  cd /home/wyy/gpt_dev_ws
  set +u
  source /opt/ros/humble/setup.bash
  source install/setup.bash
  export ROS_LOG_DIR=/tmp/ros_logs
  ros2 run weed_locator chess_test_gui > /home/wyy/gpt_dev_ws/local_chess_test_gui.log 2>&1
'

echo "chess test started"
echo "handeye: ${CHESS_HANDEYE_PATH}"
echo "target z: ${CHESS_TARGET_Z_MM}mm"
echo "offsets: X=${CHESS_X_OFFSET_MM}mm Y=${CHESS_Y_OFFSET_MM}mm Z=${CHESS_Z_OFFSET_MM}mm"
echo "delta port: ${DELTA_PORT} @ ${DELTA_BAUDRATE}"
