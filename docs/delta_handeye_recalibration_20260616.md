# Delta hand-eye recalibration record, 2026-06-16

This is a sanitized technical record of the current Delta hand-eye calibration work.
The raw Codex conversation was not committed because it contained credentials,
login details, and machine-specific secrets.

## Workspace

- ROS workspace: `/home/wyy/gpt_dev_ws`
- Main package repository: `/home/wyy/gpt_dev_ws/src/weed_locator`
- Camera: Orbbec/Astra-compatible RGB-D camera over USB
- Delta arm controller: USB serial G-code bridge
- Calibration board: ChArUco, `DICT_4X4_50`, 8 x 5 squares, square length `0.028 m`, marker length `0.020 m`

The camera was moved during testing, so old discovery heat maps, rejected zones,
and previous calibration transforms were treated as invalid. A local backup of
the pre-move calibration files was made outside this repository at:

`/home/wyy/gpt_dev_ws/calibration_targets/backup_camera_moved_20260616_164035`

The current committed snapshot lives in this repository under:

`calibration_targets/`

## Main Files

- `scripts/delta_charuco_calibration.py`: automatic ChArUco discovery, stable capture, heat map, bad zones, fitting, and progress dashboard.
- `launch/local_delta_hand_eye.launch.py`: one-shot local launch for camera, Delta bridge, calibration node, RGB-D point cloud, depth visualizer, RViz, and OpenCV debug image.
- `launch/local_delta_hand_eye_cv.launch.py`: camera/debug-focused launch variant.
- `rviz/delta_rgbd_pointcloud.rviz`: RViz layout for RGB-D point cloud inspection.
- `scripts/depth_image_visualizer.py`: converts depth images into a visible mono/BGR image for `showimage`.
- `scripts/colored_pointcloud_from_rgbd.py`: creates colored point cloud from RGB-D topics.
- `weed_locator/delta_gcode_bridge.py`: Delta serial bridge updates used by the GUI and calibration flow.
- `scripts/delta_safe_control_gui.py`: safety-oriented Delta jog/control GUI.
- `scripts/delta_safe_calibration_control.py`: calibration control helper.
- `launch/local_chess_test.launch.py`: local chess detection and hand-eye target launch.
- `scripts/chess_detector.py`: YOLO chess detection with selected target pixel output and enhanced dark-image preprocessing.
- `scripts/chess_handeye_target.py`: converts selected chess pixel plus depth into Delta target coordinates using the hand-eye calibration.
- `tools/delta_handeye/*.sh`: local helper scripts copied from the workspace root for repeatable startup/testing.

## Current Sampling Strategy

Current automatic calibration strategy is intentionally strict:

```yaml
discover_target_waypoints: 25
discover_max_probes: 260
discover_min_corners: 14
discover_grid_step_xy_mm: 10.0
discover_z_mm: -230.0
discover_z_levels_down: 1
discover_z_step_mm: 20.0
discover_bad_zone_radius_mm: 5.0
discover_bad_zone_max_corners: 11
discover_adaptive_radius_mm: 25.0
stable_detection_frames: 5
stable_detection_tolerance_mm: 1.0
post_move_detect_timeout_sec: 5.0
home_before_discovery: true
home_between_samples: true
home_settle_sec: 4.0
safe_xy_z_mm: -210.0
workspace_min_x_mm: -90.0
workspace_max_x_mm: 90.0
workspace_min_y_mm: -90.0
workspace_max_y_mm: 90.0
workspace_min_z_mm: -320.0
workspace_max_z_mm: 0.0
```

Motion policy:

- Home before auto discovery.
- For each sample, move through a safe Z before XY motion.
- Home between samples to reduce accumulated step loss.
- Accept a sample only when the OpenCV/ChArUco corner count is at least 14.
- If a probe has too few corners, mark a small local bad zone around that XY location.

## Latest One-Layer Result

The latest single-layer run after the camera moved produced:

```text
Target samples: 25
Accepted samples: 25 / 25
Rejected probes: 2
Z layer: -230 mm
Accepted corner count range: 14-17
```

Fit metrics from `calibration_targets/delta_hand_eye.yaml`:

```text
Planar affine camera_xyz -> delta_xy:
  RMSE 1.386 mm
  mean 1.197 mm
  median 1.058 mm
  max 2.888 mm

Planar affine camera_xy -> delta_xy:
  RMSE 1.960 mm
  mean 1.703 mm
  median 1.315 mm
  max 3.730 mm

Full 3D rigid:
  RMSE 9.045 mm
  mean 8.670 mm
  median 8.272 mm
  max 14.777 mm
```

Interpretation:

- The one-layer planar XY calibration is usable for same-height chess-position testing.
- The full 3D rigid metric is not the right accuracy number for a one-Z-layer data set.
- For reliable XYZ mapping, collect multiple Z layers after verifying that every sample homes correctly and does not enter mechanical dead zones.

## Camera And OpenCV Notes

- The OpenCV debug image should show detected marker/corner overlays. If the image is black or no board is detected, restart the camera/launch and verify the RGB topic.
- Low image resolution, long camera distance, motion blur, and low exposure all make the ChArUco corner pose jitter more.
- The depth image and colored point cloud are useful for sanity checks, but the current hand-eye fit uses OpenCV ChArUco board pose/corners, not raw depth as the primary calibration signal.
- For YOLO chess testing, `chess_detector.py` can brighten and contrast-enhance the RGB frame before inference. This affects YOLO recognition, not the physical camera calibration.

## Handoff

To restart the local hand-eye stack from this repository checkout:

```bash
cd /home/wyy/gpt_dev_ws
tools/delta_handeye/restart_delta_handeye_detached.sh
```

To run one chess reach attempt after the chess stack is active:

```bash
cd /home/wyy/gpt_dev_ws
tools/delta_handeye/reach_chess_once.sh
```

Keep raw credential-bearing chat logs out of public commits. Use this file as the
portable summary for another Codex/GPT session.
