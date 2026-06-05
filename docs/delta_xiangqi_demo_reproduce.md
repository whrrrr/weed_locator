# Delta Xiangqi Demo Reproduce

Last known good setup: 2026-06-05

Purpose: use Gemini depth + YOLO xiangqi detection + empirical camera-to-Delta model. Put the xiangqi in view, wait for `READY`, then trigger once to move the Delta arm.

## Terminal 1: Delta Bridge

```bash
cd /home/whr/cc_ws/tros_ws
source install/setup.bash

ros2 launch weed_locator delta_gcode_bridge.launch.py \
  port:=/dev/ttyUSB0 \
  default_feedrate:=150.0
```

If the port changes, check it with:

```bash
ls /dev/ttyUSB*
```

## Terminal 2: Gemini Camera

```bash
cd /home/whr/cc_ws/tros_ws
source install/setup.bash

ros2 launch astra_camera gemini.launch.xml
```

Expected camera topics:

```bash
ros2 topic list | grep -E 'camera|depth|image'
```

The demo expects:

```text
/camera/color/image_raw
/camera/color/camera_info
/camera/depth/image_raw
```

## Terminal 3: Visual Pick Demo

This is the successful command to keep for reproduction:

```bash
cd /home/whr/cc_ws/tros_ws
source install/setup.bash

ros2 run weed_locator delta_visual_pick_demo --ros-args \
  -p detector:=yolo \
  -p yolo_model_path:="/home/whr/文档/xwechat_files/wxid_mc7cj27h4kzg22_bc6c/msg/file/2026-05/best.pt" \
  -p yolo_class_name:=xiangqi \
  -p yolo_conf:=0.2 \
  -p use_depth:=true \
  -p depth_sample_mode:=bbox_near_mean \
  -p depth_percentile:=20.0 \
  -p depth_bbox_shrink:=0.25 \
  -p depth_temporal_window:=8 \
  -p use_depth_for_z:=true \
  -p use_empirical_model:=true \
  -p empirical_model_path:=/home/whr/cc_ws/tros_ws/calibration_targets/delta_hand_eye_z210_300_poly2_model.yaml \
  -p offset_x_mm:=120.0 \
  -p offset_y_mm:=-80.0 \
  -p offset_z_mm:=20.0 \
  -p depth_z_offset_mm:=0.0 \
  -p use_staged_motion:=false \
  -p feedrate:=150.0 \
  -p enforce_workspace_limits:=false
```

Important successful parameters:

```text
depth_sample_mode=bbox_near_mean
depth_percentile=20.0
depth_bbox_shrink=0.25
depth_temporal_window=8
empirical_model=delta_hand_eye_z210_300_poly2_model.yaml
offset=(120.0, -80.0, 20.0) mm
feedrate=150.0
```

## Terminal 4: Trigger

Only trigger after the demo window says `READY` and `cmd xyz` looks reasonable.

```bash
cd /home/whr/cc_ws/tros_ws
source install/setup.bash

ros2 topic pub --once /delta_visual_pick_demo/trigger std_msgs/msg/Empty "{}"
```

## Optional: Home First

Use this if the Delta coordinate frame may be stale. Wait until homing finishes before triggering the demo.

```bash
cd /home/whr/cc_ws/tros_ws
source install/setup.bash

ros2 topic pub --once /delta_arm/home std_msgs/msg/Empty "{}"
```

## Notes

- This version uses the polynomial empirical model from the 2026-06-05 calibration:
  `/home/whr/cc_ws/tros_ws/calibration_targets/delta_hand_eye_z210_300_poly2_model.yaml`
- The earlier rigid global fit across z=-210..-300 was poor, so do not use `delta_hand_eye_z210_300.yaml` directly for demo movement.
- Depth should change with target height. If z stops changing, first check the overlay values for `raw`, `smooth`, and `mode=bbox_near_mean`.
