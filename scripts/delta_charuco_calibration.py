#!/usr/bin/env python3
"""Interactive Delta eye-to-hand calibration with a ChArUco board."""

import json
import math
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Empty, String


def make_transform(r_mat, t_vec):
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = r_mat
    transform[:3, 3] = t_vec.reshape(3)
    return transform


def fit_rigid_transform(source_points, target_points):
    """Return R, t for target ~= R @ source + t, with no scale."""
    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if source.shape != target.shape or source.shape[1] != 3:
        raise ValueError('source and target must both be Nx3 arrays')
    if source.shape[0] < 4:
        raise ValueError('at least 4 samples are required')

    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid

    h_mat = source_centered.T @ target_centered
    u_mat, _s_vals, vt_mat = np.linalg.svd(h_mat)
    r_mat = vt_mat.T @ u_mat.T
    if np.linalg.det(r_mat) < 0.0:
        vt_mat[-1, :] *= -1.0
        r_mat = vt_mat.T @ u_mat.T
    t_vec = target_centroid - r_mat @ source_centroid
    return r_mat, t_vec


def rotation_error_deg(r_mat):
    value = (np.trace(r_mat) - 1.0) / 2.0
    value = max(-1.0, min(1.0, float(value)))
    return math.degrees(math.acos(value))


class DeltaCharucoCalibration(Node):
    def __init__(self):
        super().__init__('delta_charuco_calibration')

        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('delta_move_topic', '/delta_arm/move_to')
        self.declare_parameter('delta_home_topic', '/delta_arm/home')
        self.declare_parameter('save_path', '/home/whr/cc_ws/tros_ws/calibration_targets/delta_hand_eye.yaml')
        self.declare_parameter('validation_path', '/home/whr/cc_ws/tros_ws/calibration_targets/delta_hand_eye_filtered.yaml')
        self.declare_parameter('waypoint_path', '/home/whr/cc_ws/tros_ws/calibration_targets/delta_calibration_waypoints.yaml')
        self.declare_parameter('boundary_path', '/home/whr/cc_ws/tros_ws/calibration_targets/delta_workspace_slices.yaml')
        self.declare_parameter('debug_image_path', '/tmp/delta_charuco_debug.png')
        self.declare_parameter('squares_x', 6)
        self.declare_parameter('squares_y', 5)
        self.declare_parameter('square_length_m', 0.028)
        self.declare_parameter('marker_length_m', 0.020)
        self.declare_parameter('dictionary', 'DICT_4X4_50')
        self.declare_parameter('min_charuco_corners', 8)
        self.declare_parameter('status_period_sec', 2.0)
        self.declare_parameter('move_settle_sec', 2.0)
        self.declare_parameter('use_staged_motion', False)
        self.declare_parameter('travel_z_mm', -170.0)
        self.declare_parameter('feedrate', 80.0)
        self.declare_parameter('auto_start_index', 1)
        self.declare_parameter('auto_end_index', 0)
        self.declare_parameter('max_consecutive_skips', 3)
        self.declare_parameter('post_move_detect_timeout_sec', 5.0)
        self.declare_parameter('stable_detection_frames', 3)
        self.declare_parameter('stable_detection_tolerance_mm', 3.0)
        self.declare_parameter('grid_step_xy_mm', 20.0)
        self.declare_parameter('home_x_mm', 0.0)
        self.declare_parameter('home_y_mm', 0.0)
        self.declare_parameter('home_z_mm', -140.0)

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_image = None
        self.latest_detection = None
        self.image_seq = 0
        self.latest_detection_seq = -1
        self.latest_detect_debug = {
            'markers': 0,
            'charuco_corners': 0,
            'reason': 'no image yet',
        }
        self.validation_transform = None
        self.validation_path_loaded = None
        self.samples = []
        self.waypoints = []
        self.current_waypoint_index = -1
        self.boundary_layers = {}
        self.last_status_time = 0.0

        self.delta_xyz_mm = np.array([
            float(self.get_parameter('home_x_mm').value),
            float(self.get_parameter('home_y_mm').value),
            float(self.get_parameter('home_z_mm').value),
        ], dtype=float)

        dictionary_name = str(self.get_parameter('dictionary').value)
        dictionary_id = getattr(cv2.aruco, dictionary_name)
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.board = cv2.aruco.CharucoBoard(
            (
                int(self.get_parameter('squares_x').value),
                int(self.get_parameter('squares_y').value),
            ),
            float(self.get_parameter('square_length_m').value),
            float(self.get_parameter('marker_length_m').value),
            self.aruco_dictionary,
        )
        self.detector = cv2.aruco.CharucoDetector(self.board)

        self.create_subscription(
            CameraInfo,
            str(self.get_parameter('camera_info_topic').value),
            self.on_camera_info,
            10,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter('image_topic').value),
            self.on_image,
            10,
        )
        self.create_subscription(
            Point,
            str(self.get_parameter('delta_move_topic').value),
            self.on_delta_move,
            10,
        )
        self.create_subscription(
            Empty,
            str(self.get_parameter('delta_home_topic').value),
            self.on_delta_home,
            10,
        )
        self.delta_move_pub = self.create_publisher(
            Point,
            str(self.get_parameter('delta_move_topic').value),
            10,
        )
        self.delta_home_pub = self.create_publisher(
            Empty,
            str(self.get_parameter('delta_home_topic').value),
            10,
        )
        self.raw_gcode_pub = self.create_publisher(
            String,
            '/delta_arm/gcode_raw',
            10,
        )

        self.board_pose_pub = self.create_publisher(
            PoseStamped,
            '/delta_charuco/board_pose_camera',
            10,
        )
        self.board_point_pub = self.create_publisher(
            PointStamped,
            '/delta_charuco/board_origin_camera',
            10,
        )
        self.predicted_delta_pub = self.create_publisher(
            PointStamped,
            '/delta_charuco/predicted_delta_point',
            10,
        )

        self.load_validation_transform()
        self.load_waypoints(log_missing=False)
        self.load_boundary_layers(log_missing=False)
        self.get_logger().info('Delta ChArUco calibration ready')
        self.print_help()

    def print_help(self):
        print('')
        print('Delta ChArUco calibration')
        print('  space: save current sample')
        print('  c: compute calibration from saved samples')
        print('  w: write debug image to disk')
        print('  l: list saved samples')
        print('  u: undo last sample')
        print('  p: save current delta pose as a waypoint')
        print('  y: list waypoints')
        print('  o: write waypoints to disk')
        print('  r: load waypoints from disk')
        print('  n: move to next waypoint')
        print('  b: move to previous waypoint')
        print('  a: auto-run all waypoints and save samples')
        print('  k: save current pose as boundary point for current Z layer')
        print('  j: list boundary points for current Z layer')
        print('  t: save boundary layers to disk')
        print('  i: load boundary layers from disk')
        print('  g: generate grid waypoints from current Z boundary layer')
        print('  q: quit')
        print('')

    def load_validation_transform(self):
        path = Path(os.path.expanduser(str(self.get_parameter('validation_path').value)))
        if not path.exists():
            self.get_logger().warning(f'validation transform not found: {path}')
            return False
        try:
            with path.open('r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
            self.validation_transform = np.array(data['T_delta_camera'], dtype=float)
            self.validation_path_loaded = str(path)
            err = data.get('error', {})
            self.get_logger().info(
                'loaded validation transform: %s (rmse=%.2f mm, samples=%s)'
                % (
                    path,
                    float(err.get('rmse_m', 0.0)) * 1000.0,
                    err.get('count', '?'),
                )
            )
            return True
        except Exception as exc:
            self.get_logger().error(f'failed to load validation transform {path}: {exc}')
            return False

    def waypoint_path(self):
        return Path(os.path.expanduser(str(self.get_parameter('waypoint_path').value)))

    def boundary_path(self):
        return Path(os.path.expanduser(str(self.get_parameter('boundary_path').value)))

    def move_settle_sec(self):
        return float(self.get_parameter('move_settle_sec').value)

    def use_staged_motion(self):
        return bool(self.get_parameter('use_staged_motion').value)

    def travel_z_mm(self):
        return float(self.get_parameter('travel_z_mm').value)

    def feedrate(self):
        return float(self.get_parameter('feedrate').value)

    def auto_start_index(self):
        return max(1, int(self.get_parameter('auto_start_index').value))

    def auto_end_index(self):
        return int(self.get_parameter('auto_end_index').value)

    def max_consecutive_skips(self):
        return max(1, int(self.get_parameter('max_consecutive_skips').value))

    def post_move_detect_timeout_sec(self):
        return max(0.0, float(self.get_parameter('post_move_detect_timeout_sec').value))

    def stable_detection_frames(self):
        return max(1, int(self.get_parameter('stable_detection_frames').value))

    def stable_detection_tolerance_mm(self):
        return max(0.0, float(self.get_parameter('stable_detection_tolerance_mm').value))

    def grid_step_xy_mm(self):
        return float(self.get_parameter('grid_step_xy_mm').value)

    @staticmethod
    def z_layer_key(z_mm):
        return int(round(float(z_mm)))

    def load_waypoints(self, log_missing=True):
        path = self.waypoint_path()
        if not path.exists():
            if log_missing:
                self.get_logger().warning(f'waypoint file not found: {path}')
            return False
        try:
            with path.open('r', encoding='utf-8') as file:
                data = yaml.safe_load(file) or {}
            loaded = data.get('waypoints', [])
            self.waypoints = []
            for item in loaded:
                xyz = np.array(
                    [
                        float(item['x_mm']),
                        float(item['y_mm']),
                        float(item['z_mm']),
                    ],
                    dtype=float,
                )
                self.waypoints.append({
                    'name': str(item.get('name', f'pt_{len(self.waypoints) + 1:02d}')),
                    'xyz_mm': xyz,
                })
            self.current_waypoint_index = 0 if self.waypoints else -1
            self.get_logger().info(f'loaded {len(self.waypoints)} waypoints from {path}')
            return True
        except Exception as exc:
            self.get_logger().error(f'failed to load waypoints from {path}: {exc}')
            return False

    def save_waypoints(self):
        path = self.waypoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'description': 'Delta calibration waypoints in delta base frame (mm)',
            'waypoints': [
                {
                    'name': waypoint['name'],
                    'x_mm': float(waypoint['xyz_mm'][0]),
                    'y_mm': float(waypoint['xyz_mm'][1]),
                    'z_mm': float(waypoint['xyz_mm'][2]),
                }
                for waypoint in self.waypoints
            ],
        }
        with path.open('w', encoding='utf-8') as file:
            yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
        self.get_logger().info(f'saved {len(self.waypoints)} waypoints to {path}')

    def load_boundary_layers(self, log_missing=True):
        path = self.boundary_path()
        if not path.exists():
            if log_missing:
                self.get_logger().warning(f'boundary layer file not found: {path}')
            return False
        try:
            with path.open('r', encoding='utf-8') as file:
                data = yaml.safe_load(file) or {}
            self.boundary_layers = {}
            for layer in data.get('layers', []):
                z_key = self.z_layer_key(layer['z_mm'])
                points = []
                for item in layer.get('points', []):
                    points.append(np.array([float(item[0]), float(item[1])], dtype=float))
                self.boundary_layers[z_key] = points
            self.get_logger().info(f'loaded {len(self.boundary_layers)} boundary layers from {path}')
            return True
        except Exception as exc:
            self.get_logger().error(f'failed to load boundary layers from {path}: {exc}')
            return False

    def save_boundary_layers(self):
        path = self.boundary_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        layers = []
        for z_key in sorted(self.boundary_layers):
            layers.append({
                'z_mm': int(z_key),
                'points': [
                    [float(point[0]), float(point[1])]
                    for point in self.boundary_layers[z_key]
                ],
            })
        data = {
            'description': 'Safe XY boundary points grouped by Z slice (mm)',
            'layers': layers,
        }
        with path.open('w', encoding='utf-8') as file:
            yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
        self.get_logger().info(f'saved {len(layers)} boundary layers to {path}')

    def add_boundary_point_from_current_pose(self):
        z_key = self.z_layer_key(self.delta_xyz_mm[2])
        xy = self.delta_xyz_mm[:2].astype(float).copy()
        self.boundary_layers.setdefault(z_key, []).append(xy)
        self.get_logger().info(
            'saved boundary point for z=%d mm: %s'
            % (z_key, np.round(xy, 1).tolist())
        )

    def list_boundary_points(self):
        z_key = self.z_layer_key(self.delta_xyz_mm[2])
        points = self.boundary_layers.get(z_key, [])
        if not points:
            self.get_logger().info(f'no boundary points recorded for z={z_key} mm')
            return
        self.get_logger().info(f'boundary layer z={z_key} mm has {len(points)} points')
        for index, point in enumerate(points, start=1):
            self.get_logger().info(
                '%02d xy_mm=%s'
                % (index, np.round(point, 1).tolist())
            )

    @staticmethod
    def convex_hull(points_xy):
        pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts, clockwise=False)
        return hull.reshape(-1, 2).astype(float)

    def generate_grid_waypoints_from_current_layer(self):
        z_key = self.z_layer_key(self.delta_xyz_mm[2])
        points = self.boundary_layers.get(z_key, [])
        if len(points) < 3:
            self.get_logger().warning(
                f'need at least 3 boundary points for z={z_key} mm, got {len(points)}'
            )
            return

        hull = self.convex_hull(points)
        hull_cv = hull.astype(np.float32).reshape(-1, 1, 2)
        x_min = float(np.min(hull[:, 0]))
        x_max = float(np.max(hull[:, 0]))
        y_min = float(np.min(hull[:, 1]))
        y_max = float(np.max(hull[:, 1]))
        step = max(1.0, self.grid_step_xy_mm())

        generated = []
        y = y_min
        row_index = 0
        while y <= y_max + 1e-6:
            xs = np.arange(x_min, x_max + 0.5 * step, step)
            if row_index % 2 == 1:
                xs = xs[::-1]
            for x in xs:
                inside = cv2.pointPolygonTest(hull_cv, (float(x), float(y)), False)
                if inside >= 0.0:
                    xyz = np.array([float(x), float(y), float(z_key)], dtype=float)
                    generated.append({
                        'name': f'z{z_key}_pt_{len(generated) + 1:02d}',
                        'xyz_mm': xyz,
                    })
            y += step
            row_index += 1

        if not generated:
            self.get_logger().warning(f'no grid waypoints generated for z={z_key} mm')
            return

        self.waypoints = [wp for wp in self.waypoints if self.z_layer_key(wp['xyz_mm'][2]) != z_key]
        self.waypoints.extend(generated)
        self.current_waypoint_index = 0 if self.waypoints else -1
        self.get_logger().info(
            'generated %d waypoints for z=%d mm using step %.1f mm'
            % (len(generated), z_key, step)
        )

    def add_waypoint_from_current_pose(self):
        xyz = self.delta_xyz_mm.astype(float).copy()
        name = f'pt_{len(self.waypoints) + 1:02d}'
        self.waypoints.append({
            'name': name,
            'xyz_mm': xyz,
        })
        self.current_waypoint_index = len(self.waypoints) - 1
        self.get_logger().info(
            'saved waypoint %s at %s mm'
            % (name, np.round(xyz, 1).tolist())
        )

    def list_waypoints(self):
        if not self.waypoints:
            self.get_logger().info('no waypoints saved')
            return
        for index, waypoint in enumerate(self.waypoints, start=1):
            marker = '*' if index - 1 == self.current_waypoint_index else ' '
            self.get_logger().info(
                '%s%02d %s delta_mm=%s'
                % (
                    marker,
                    index,
                    waypoint['name'],
                    np.round(waypoint['xyz_mm'], 1).tolist(),
                )
            )

    def publish_direct_move(self, xyz_mm):
        msg = Point()
        msg.x = float(xyz_mm[0])
        msg.y = float(xyz_mm[1])
        msg.z = float(xyz_mm[2])
        self.delta_move_pub.publish(msg)
        self.get_logger().info(
            'sent delta move to %s mm'
            % np.round(np.array([msg.x, msg.y, msg.z]), 1).tolist()
        )

    def publish_staged_move(self, xyz_mm):
        target = np.array(xyz_mm, dtype=float)
        current = np.array(self.delta_xyz_mm, dtype=float)
        travel_z = self.travel_z_mm()
        feedrate = self.feedrate()
        staged_points = [
            (current[0], current[1], travel_z),
            (target[0], target[1], travel_z),
            (target[0], target[1], target[2]),
        ]

        lines = ['G90']
        for x_mm, y_mm, z_mm in staged_points:
            lines.append('G1 X%.2f Y%.2f Z%.2f F%.2f' % (x_mm, y_mm, z_mm, feedrate))

        msg = String()
        msg.data = '\n'.join(lines)
        self.raw_gcode_pub.publish(msg)
        self.delta_xyz_mm = target
        self.get_logger().info(
            'sent staged delta move: current=(%.1f, %.1f, %.1f) -> travel_z=%.1f -> target=%s mm'
            % (
                current[0],
                current[1],
                current[2],
                travel_z,
                np.round(target, 1).tolist(),
            )
        )

    def publish_move(self, xyz_mm):
        if self.use_staged_motion():
            self.publish_staged_move(xyz_mm)
        else:
            self.publish_direct_move(xyz_mm)

    def move_to_waypoint(self, index):
        if not self.waypoints:
            self.get_logger().warning('no waypoints available')
            return False
        index = max(0, min(index, len(self.waypoints) - 1))
        waypoint = self.waypoints[index]
        self.current_waypoint_index = index
        self.publish_move(waypoint['xyz_mm'])
        self.get_logger().info(f'moving to waypoint {index + 1}/{len(self.waypoints)}: {waypoint["name"]}')
        return True

    def wait_with_spin(self, duration_sec):
        end_time = time.time() + max(0.0, duration_sec)
        while rclpy.ok() and time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for_stable_detection(self, min_image_seq):
        timeout_sec = self.post_move_detect_timeout_sec()
        stable_needed = self.stable_detection_frames()
        tolerance_m = self.stable_detection_tolerance_mm() / 1000.0
        end_time = time.time() + timeout_sec
        last_detection_seq = -1
        last_tvec = None
        stable_count = 0

        while rclpy.ok() and time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.latest_detection is None:
                stable_count = 0
                last_tvec = None
                continue
            if self.latest_detection_seq <= min_image_seq:
                continue
            if self.latest_detection_seq == last_detection_seq:
                continue

            tvec = self.latest_detection['tvec'].reshape(3).astype(float)
            if last_tvec is None:
                stable_count = 1
            else:
                shift = float(np.linalg.norm(tvec - last_tvec))
                stable_count = stable_count + 1 if shift <= tolerance_m else 1
            last_tvec = tvec
            last_detection_seq = self.latest_detection_seq

            if stable_count >= stable_needed:
                return True

        return False

    def auto_run_waypoints(self):
        if not self.waypoints:
            self.get_logger().warning('no waypoints to auto-run')
            return
        self.get_logger().info(
            'auto-running waypoints %d..%d/%d with settle %.2f sec, max_consecutive_skips=%d'
            % (
                self.auto_start_index(),
                self.auto_end_index() if self.auto_end_index() > 0 else len(self.waypoints),
                len(self.waypoints),
                self.move_settle_sec(),
                self.max_consecutive_skips(),
            )
        )
        saved_before = len(self.samples)
        start_index = self.auto_start_index() - 1
        end_index = self.auto_end_index()
        if end_index <= 0:
            end_index = len(self.waypoints)
        end_index = min(end_index, len(self.waypoints))
        consecutive_skips = 0
        for index in range(start_index, end_index):
            if not rclpy.ok():
                break
            image_seq_before_move = self.image_seq
            self.move_to_waypoint(index)
            self.wait_with_spin(self.move_settle_sec())
            if not self.wait_for_stable_detection(image_seq_before_move):
                consecutive_skips += 1
                self.get_logger().warning(
                    'waypoint %d skipped: ChArUco board was not stable after move (%d/%d consecutive skips)'
                    % (index + 1, consecutive_skips, self.max_consecutive_skips())
                )
                if consecutive_skips >= self.max_consecutive_skips():
                    self.get_logger().error(
                        'auto-run stopped after %d consecutive skipped waypoints; camera cannot see the board in this region'
                        % consecutive_skips
                    )
                    break
                continue
            consecutive_skips = 0
            self.save_sample()
        self.get_logger().info(
            'auto-run finished: saved %d new samples'
            % (len(self.samples) - saved_before)
        )

    def log_box(self, title, lines, level='info'):
        width = 72
        border = '=' * width
        text_lines = [border, f'{title}', '-' * width]
        text_lines.extend(str(line) for line in lines)
        text_lines.append(border)
        text = '\n'.join(text_lines)
        if level == 'warning':
            self.get_logger().warning(text)
        elif level == 'error':
            self.get_logger().error(text)
        else:
            self.get_logger().info(text)

    def on_camera_info(self, msg):
        self.camera_matrix = np.array(msg.k, dtype=float).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=float).reshape(-1, 1)

    def on_delta_move(self, msg):
        self.delta_xyz_mm = np.array([msg.x, msg.y, msg.z], dtype=float)

    def on_delta_home(self, _msg):
        self.delta_xyz_mm = np.array([
            float(self.get_parameter('home_x_mm').value),
            float(self.get_parameter('home_y_mm').value),
            float(self.get_parameter('home_z_mm').value),
        ], dtype=float)

    def on_image(self, msg):
        if self.camera_matrix is None:
            return
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warning(f'failed to convert image: {exc}')
            return

        self.latest_image = image
        self.image_seq += 1
        self.latest_detection = self.detect_charuco(image, msg.header.frame_id)
        if self.latest_detection is not None:
            self.latest_detection_seq = self.image_seq
        now = time.time()
        status_period = float(self.get_parameter('status_period_sec').value)
        if now - self.last_status_time > status_period:
            self.last_status_time = now
            if self.latest_detection is None:
                self.log_box(
                    'CHARUCO STATUS: NO BOARD',
                    [
                        f'reason: {self.latest_detect_debug.get("reason", "unknown")}',
                        f'markers detected: {self.latest_detect_debug.get("markers", 0)}',
                        f'charuco corners: {self.latest_detect_debug.get("charuco_corners", 0)}',
                        f'min required corners: {int(self.get_parameter("min_charuco_corners").value)}',
                        f'samples: {len(self.samples)}',
                        f'delta_xyz_mm: {np.round(self.delta_xyz_mm, 1).tolist()}',
                        'try: improve light, reduce glare, show more white border, or move closer',
                    ],
                    level='warning',
                )
            else:
                point_m = self.latest_detection['tvec'].reshape(3)
                corner_count = int(self.latest_detection['corner_count'])
                self.log_box(
                    'CHARUCO STATUS: BOARD OK',
                    self.make_status_lines(point_m, corner_count),
                )
                self.publish_board_pose(msg.header, self.latest_detection)

    def make_status_lines(self, camera_xyz_m, corner_count):
        lines = [
                        f'corners: {corner_count}  (>=8 ok, more is better)',
            f'camera_xyz_m: {np.round(camera_xyz_m, 4).tolist()}',
                        f'delta_xyz_mm: {np.round(self.delta_xyz_mm, 1).tolist()}',
                        f'samples saved: {len(self.samples)}',
        ]
        if self.validation_transform is not None:
            camera_h = np.append(camera_xyz_m.reshape(3), 1.0)
            predicted_m = (self.validation_transform @ camera_h)[:3]
            predicted_mm = predicted_m * 1000.0
            error_mm = predicted_mm - self.delta_xyz_mm
            error_norm = float(np.linalg.norm(error_mm))
            lines.extend([
                f'predicted_delta_mm: {np.round(predicted_mm, 1).tolist()}',
                f'validation_error_mm: {np.round(error_mm, 1).tolist()} | norm={error_norm:.1f}',
            ])
            self.publish_predicted_delta(predicted_m)
        lines.append('press SPACE to save sample, or move to another pose for validation')
        return lines

    def publish_predicted_delta(self, predicted_m):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'delta_base'
        msg.point.x = float(predicted_m[0])
        msg.point.y = float(predicted_m[1])
        msg.point.z = float(predicted_m[2])
        self.predicted_delta_pub.publish(msg)

    def detect_charuco(self, image, frame_id):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(gray)
        min_corners = int(self.get_parameter('min_charuco_corners').value)
        marker_count = 0 if marker_ids is None else int(len(marker_ids))
        charuco_count = 0 if charuco_ids is None else int(len(charuco_ids))
        self.latest_detect_debug = {
            'markers': marker_count,
            'charuco_corners': charuco_count,
            'reason': 'ok',
        }
        if charuco_corners is None or charuco_ids is None or len(charuco_ids) < min_corners:
            if marker_count == 0:
                reason = 'no ArUco markers detected'
            else:
                reason = f'only {charuco_count} ChArUco corners; need at least {min_corners}'
            self.latest_detect_debug = {
                'markers': marker_count,
                'charuco_corners': charuco_count,
                'reason': reason,
            }
            return None

        all_corners = np.asarray(self.board.getChessboardCorners(), dtype=np.float32)
        ids = charuco_ids.reshape(-1).astype(int)
        object_points = all_corners[ids].reshape(-1, 1, 3)
        image_points = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2)

        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        return {
            'rvec': rvec.reshape(3, 1),
            'tvec': tvec.reshape(3, 1),
            'corner_count': len(ids),
            'charuco_corners': charuco_corners,
            'charuco_ids': charuco_ids,
            'marker_corners': marker_corners,
            'marker_ids': marker_ids,
            'frame_id': frame_id,
        }

    def publish_board_pose(self, header, detection):
        r_mat, _ = cv2.Rodrigues(detection['rvec'])
        quat = self.rotation_matrix_to_quaternion(r_mat)

        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.pose.position.x = float(detection['tvec'][0])
        pose_msg.pose.position.y = float(detection['tvec'][1])
        pose_msg.pose.position.z = float(detection['tvec'][2])
        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])
        self.board_pose_pub.publish(pose_msg)

        point_msg = PointStamped()
        point_msg.header = header
        point_msg.point.x = pose_msg.pose.position.x
        point_msg.point.y = pose_msg.pose.position.y
        point_msg.point.z = pose_msg.pose.position.z
        self.board_point_pub.publish(point_msg)

    @staticmethod
    def rotation_matrix_to_quaternion(r_mat):
        trace = np.trace(r_mat)
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (r_mat[2, 1] - r_mat[1, 2]) / scale
            qy = (r_mat[0, 2] - r_mat[2, 0]) / scale
            qz = (r_mat[1, 0] - r_mat[0, 1]) / scale
        else:
            index = int(np.argmax(np.diag(r_mat)))
            if index == 0:
                scale = math.sqrt(1.0 + r_mat[0, 0] - r_mat[1, 1] - r_mat[2, 2]) * 2.0
                qw = (r_mat[2, 1] - r_mat[1, 2]) / scale
                qx = 0.25 * scale
                qy = (r_mat[0, 1] + r_mat[1, 0]) / scale
                qz = (r_mat[0, 2] + r_mat[2, 0]) / scale
            elif index == 1:
                scale = math.sqrt(1.0 + r_mat[1, 1] - r_mat[0, 0] - r_mat[2, 2]) * 2.0
                qw = (r_mat[0, 2] - r_mat[2, 0]) / scale
                qx = (r_mat[0, 1] + r_mat[1, 0]) / scale
                qy = 0.25 * scale
                qz = (r_mat[1, 2] + r_mat[2, 1]) / scale
            else:
                scale = math.sqrt(1.0 + r_mat[2, 2] - r_mat[0, 0] - r_mat[1, 1]) * 2.0
                qw = (r_mat[1, 0] - r_mat[0, 1]) / scale
                qx = (r_mat[0, 2] + r_mat[2, 0]) / scale
                qy = (r_mat[1, 2] + r_mat[2, 1]) / scale
                qz = 0.25 * scale
        return np.array([qx, qy, qz, qw], dtype=float)

    def save_sample(self):
        if self.latest_detection is None:
            self.get_logger().warning('cannot save sample: ChArUco board is not detected')
            return
        camera_xyz_m = self.latest_detection['tvec'].reshape(3).astype(float)
        delta_xyz_m = self.delta_xyz_mm.astype(float) / 1000.0
        sample = {
            'camera_xyz_m': camera_xyz_m.tolist(),
            'delta_xyz_mm': self.delta_xyz_mm.astype(float).tolist(),
            'delta_xyz_m': delta_xyz_m.tolist(),
            'corner_count': int(self.latest_detection['corner_count']),
        }
        self.samples.append(sample)
        self.log_box(
            f'SAVED SAMPLE {len(self.samples)}',
            [
                'camera_m: %s' % (
                    np.round(camera_xyz_m, 4).tolist(),
                ),
                'delta_mm: %s' % (
                    np.round(self.delta_xyz_mm, 1).tolist(),
                ),
                f'corners: {sample["corner_count"]}',
                'move to the next pose, wait until stable, then press SPACE again',
            ],
        )

    def undo_sample(self):
        if not self.samples:
            self.get_logger().warning('no sample to undo')
            return
        removed = self.samples.pop()
        self.get_logger().info(f'removed sample {len(self.samples) + 1}: {removed}')

    def list_samples(self):
        if not self.samples:
            self.get_logger().info('no samples saved')
            return
        for index, sample in enumerate(self.samples, start=1):
            self.get_logger().info(
                '%02d camera_m=%s delta_mm=%s corners=%d'
                % (
                    index,
                    np.round(sample['camera_xyz_m'], 4).tolist(),
                    np.round(sample['delta_xyz_mm'], 1).tolist(),
                    sample['corner_count'],
                )
            )

    def compute_and_save(self):
        if len(self.samples) < 4:
            self.get_logger().error(f'at least 4 samples required, got {len(self.samples)}')
            return False

        camera_points = np.array([s['camera_xyz_m'] for s in self.samples], dtype=float)
        delta_points = np.array([s['delta_xyz_m'] for s in self.samples], dtype=float)
        r_mat, t_vec = fit_rigid_transform(camera_points, delta_points)
        predicted = (r_mat @ camera_points.T).T + t_vec
        errors_m = np.linalg.norm(predicted - delta_points, axis=1)

        transform = make_transform(r_mat, t_vec)
        inverse = np.linalg.inv(transform)
        save_path = Path(os.path.expanduser(str(self.get_parameter('save_path').value)))
        save_path.parent.mkdir(parents=True, exist_ok=True)

        result = {
            'description': 'Delta eye-to-hand calibration. delta_m = R * camera_m + t',
            'camera_frame': 'camera_color_optical_frame',
            'delta_frame': 'delta_base',
            'board': {
                'type': 'charuco',
                'dictionary': str(self.get_parameter('dictionary').value),
                'squares_x': int(self.get_parameter('squares_x').value),
                'squares_y': int(self.get_parameter('squares_y').value),
                'square_length_m': float(self.get_parameter('square_length_m').value),
                'marker_length_m': float(self.get_parameter('marker_length_m').value),
            },
            'T_delta_camera': transform.tolist(),
            'T_camera_delta': inverse.tolist(),
            'translation_delta_camera_m': t_vec.tolist(),
            'rotation_delta_camera': r_mat.tolist(),
            'samples': self.samples,
            'error': {
                'count': int(len(errors_m)),
                'rmse_m': float(np.sqrt(np.mean(errors_m ** 2))),
                'mean_m': float(np.mean(errors_m)),
                'max_m': float(np.max(errors_m)),
                'per_sample_m': errors_m.tolist(),
            },
        }

        with save_path.open('w', encoding='utf-8') as file:
            yaml.safe_dump(result, file, sort_keys=False, allow_unicode=True)

        self.get_logger().info(f'saved calibration: {save_path}')
        self.get_logger().info('T_delta_camera:\n' + np.array2string(transform, precision=6, suppress_small=True))
        self.get_logger().info(
            'fit error: rmse=%.2f mm, mean=%.2f mm, max=%.2f mm'
            % (
                result['error']['rmse_m'] * 1000.0,
                result['error']['mean_m'] * 1000.0,
                result['error']['max_m'] * 1000.0,
            )
        )
        self.get_logger().info('rotation magnitude %.2f deg' % rotation_error_deg(r_mat))
        return True

    def write_debug_image(self):
        if self.latest_image is None:
            self.get_logger().warning('no image received yet')
            return
        debug = self.latest_image.copy()
        if self.latest_detection is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                debug,
                self.latest_detection['charuco_corners'],
                self.latest_detection['charuco_ids'],
            )
            cv2.drawFrameAxes(
                debug,
                self.camera_matrix,
                self.dist_coeffs,
                self.latest_detection['rvec'],
                self.latest_detection['tvec'],
                0.05,
            )
        path = Path(os.path.expanduser(str(self.get_parameter('debug_image_path').value)))
        cv2.imwrite(str(path), debug)
        self.get_logger().info(f'wrote debug image: {path}')

    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            if select.select([sys.stdin], [], [], 0.05)[0]:
                return sys.stdin.read(1)
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def spin_keyboard(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.02)
            key = self.get_key()
            if key is None:
                continue
            if key == ' ':
                self.save_sample()
            elif key == 'c':
                self.compute_and_save()
            elif key == 'w':
                self.write_debug_image()
            elif key == 'l':
                self.list_samples()
            elif key == 'u':
                self.undo_sample()
            elif key == 'p':
                self.add_waypoint_from_current_pose()
            elif key == 'y':
                self.list_waypoints()
            elif key == 'o':
                self.save_waypoints()
            elif key == 'r':
                self.load_waypoints()
            elif key == 'n':
                if self.waypoints:
                    next_index = (self.current_waypoint_index + 1) % len(self.waypoints)
                    self.move_to_waypoint(next_index)
                else:
                    self.get_logger().warning('no waypoints available')
            elif key == 'b':
                if self.waypoints:
                    prev_index = (self.current_waypoint_index - 1) % len(self.waypoints)
                    self.move_to_waypoint(prev_index)
                else:
                    self.get_logger().warning('no waypoints available')
            elif key == 'a':
                self.auto_run_waypoints()
            elif key == 'k':
                self.add_boundary_point_from_current_pose()
            elif key == 'j':
                self.list_boundary_points()
            elif key == 't':
                self.save_boundary_layers()
            elif key == 'i':
                self.load_boundary_layers()
            elif key == 'g':
                self.generate_grid_waypoints_from_current_layer()
            elif key == 'q' or key == '\x03':
                break


def main(args=None):
    rclpy.init(args=args)
    node = DeltaCharucoCalibration()
    try:
        node.spin_keyboard()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
