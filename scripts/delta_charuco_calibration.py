#!/usr/bin/env python3
"""Interactive Delta eye-to-hand calibration with a ChArUco board."""

import json
import math
import os
import select
import sys
import termios
import threading
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
from std_msgs.msg import Bool, Empty, String


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


def fit_affine(source_points, target_points):
    """Return least-squares affine coefficients for target ~= [source, 1] @ coeffs."""
    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if source.shape[0] != target.shape[0]:
        raise ValueError('source and target must have the same number of rows')
    design = np.hstack([source, np.ones((source.shape[0], 1), dtype=float)])
    coeffs, _residuals, _rank, _singular = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ coeffs
    errors = np.linalg.norm(predicted - target, axis=1)
    return coeffs, predicted, errors


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
        self.declare_parameter('manual_rotation_path', '/home/whr/cc_ws/tros_ws/calibration_targets/delta_manual_board_rotations.yaml')
        self.declare_parameter('bad_zone_path', '/home/whr/cc_ws/tros_ws/calibration_targets/delta_bad_discovery_zones.yaml')
        self.declare_parameter('corner_observation_path', '/home/whr/cc_ws/tros_ws/calibration_targets/delta_corner_observations.yaml')
        self.declare_parameter('discovery_progress_path', '/home/whr/cc_ws/tros_ws/calibration_targets/delta_discovery_progress.yaml')
        self.declare_parameter('discovery_dashboard_path', '/home/whr/cc_ws/tros_ws/calibration_targets/delta_discovery_dashboard.txt')
        self.declare_parameter('debug_image_path', '/tmp/delta_charuco_debug.png')
        self.declare_parameter('squares_x', 6)
        self.declare_parameter('squares_y', 5)
        self.declare_parameter('square_length_m', 0.028)
        self.declare_parameter('marker_length_m', 0.020)
        self.declare_parameter('dictionary', 'DICT_4X4_50')
        self.declare_parameter('min_charuco_corners', 14)
        self.declare_parameter('max_reprojection_error_px', 1.5)
        self.declare_parameter('calibration_outlier_threshold_mm', 20.0)
        self.declare_parameter('status_period_sec', 2.0)
        self.declare_parameter('move_settle_sec', 2.0)
        self.declare_parameter('home_before_discovery', True)
        self.declare_parameter('home_between_samples', False)
        self.declare_parameter('home_settle_sec', 4.0)
        self.declare_parameter('use_staged_motion', False)
        self.declare_parameter('travel_z_mm', -170.0)
        self.declare_parameter('feedrate', 80.0)
        self.declare_parameter('auto_start_index', 1)
        self.declare_parameter('auto_end_index', 8)
        self.declare_parameter('manual_rotation_waypoint_index', 9)
        self.declare_parameter('max_consecutive_skips', 3)
        self.declare_parameter('post_move_detect_timeout_sec', 5.0)
        self.declare_parameter('stable_detection_frames', 3)
        self.declare_parameter('stable_detection_tolerance_mm', 3.0)
        self.declare_parameter('grid_step_xy_mm', 20.0)
        self.declare_parameter('home_x_mm', 0.0)
        self.declare_parameter('home_y_mm', 0.0)
        self.declare_parameter('home_z_mm', 0.0)
        self.declare_parameter('hold_after_auto_run', True)
        self.declare_parameter('motion_safety_enabled', True)
        self.declare_parameter('safe_xy_z_mm', -210.0)
        self.declare_parameter('workspace_min_x_mm', -90.0)
        self.declare_parameter('workspace_max_x_mm', 90.0)
        self.declare_parameter('workspace_min_y_mm', -90.0)
        self.declare_parameter('workspace_max_y_mm', 90.0)
        self.declare_parameter('workspace_min_z_mm', -320.0)
        self.declare_parameter('workspace_max_z_mm', 0.0)
        self.declare_parameter('jog_step_xy_mm', 5.0)
        self.declare_parameter('jog_step_z_mm', 5.0)
        self.declare_parameter('debug_image_topic', '/delta_charuco/debug_image')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('auto_discover_on_start', False)
        self.declare_parameter('discover_target_waypoints', 25)
        self.declare_parameter('discover_min_corners', 14)
        self.declare_parameter('discover_grid_step_xy_mm', 10.0)
        self.declare_parameter('discover_z_mm', -230.0)
        self.declare_parameter('discover_z_levels_down', 1)
        self.declare_parameter('discover_z_step_mm', 20.0)
        self.declare_parameter('discover_bounds_margin_mm', 0.0)
        self.declare_parameter('discover_save_samples', True)
        self.declare_parameter('discover_compute_after', True)
        self.declare_parameter('discover_bad_zone_radius_mm', 5.0)
        self.declare_parameter('discover_bad_zone_max_corners', 11)
        self.declare_parameter('discover_adaptive_radius_mm', 25.0)
        self.declare_parameter('discover_max_probes', 260)

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_image = None
        self.latest_detection = None
        self.latest_raw_detection = None
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
        self.manual_rotation_samples = []
        self.waypoints = []
        self.current_waypoint_index = -1
        self.boundary_layers = {}
        self.bad_zones = []
        self.corner_observations = []
        self.last_status_time = 0.0
        self.long_task_running = False
        self.cancel_long_task = False
        self.spin_thread_id = None

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
        self.create_subscription(
            String,
            '/delta_charuco/command',
            self.on_command,
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
        self.motor_enable_pub = self.create_publisher(
            Bool,
            '/delta_arm/motor_enable',
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
        self.debug_image_pub = self.create_publisher(
            Image,
            str(self.get_parameter('debug_image_topic').value),
            10,
        )

        self.load_validation_transform()
        self.load_waypoints(log_missing=False)
        self.load_boundary_layers(log_missing=False)
        self.load_bad_zones(log_missing=False)
        self.load_corner_observations(log_missing=False)
        self.get_logger().info('Delta ChArUco calibration ready')
        if bool(self.get_parameter('auto_discover_on_start').value):
            self.create_timer(2.0, self.auto_discover_timer_once)
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
        print('  R: move to manual rotation waypoint and hold motors')
        print('  0: home with G28')
        print('  I/K: jog Y +/-')
        print('  J/L: jog X -/+')
        print('  U/O: jog Z up/down')
        print('  m: clear all saved waypoints')
        print('  h: hold current delta pose with motors enabled')
        print('  e: enable delta motors')
        print('  d: disable delta motors')
        print('  v: save current board pose as a manual rotation sample')
        print('  x: write manual rotation samples to disk')
        print('  k: save current pose as boundary point for current Z layer')
        print('  j: list boundary points for current Z layer')
        print('  t: save boundary layers to disk')
        print('  i: load boundary layers from disk')
        print('  g: generate grid waypoints from current Z boundary layer')
        print('  ROS command topic /delta_charuco/command:')
        print('    discover: scan safe grid and save visible waypoints')
        print('    discover_calibrate: scan, save samples, compute calibration')
        print('    auto: run saved waypoints and save samples')
        print('    compute: compute calibration from saved samples')
        print('    mark_bad: mark current delta pose as a bad discovery zone')
        print('    mark_bad_stop: mark current pose bad and stop current auto discovery')
        print('    stop: stop current auto discovery after current motion settles')
        print('  q: quit')
        print('')

    def auto_discover_timer_once(self):
        if getattr(self, '_auto_discover_started', False):
            return
        self._auto_discover_started = True
        self.start_long_task('auto_discover_on_start', self.discover_visible_waypoints)

    def on_command(self, msg):
        command = msg.data.strip().lower()
        if command == 'discover':
            self.start_long_task(command, self.discover_visible_waypoints)
        elif command == 'discover_calibrate':
            self.start_long_task(command, self.discover_visible_waypoints, True)
        elif command == 'auto':
            self.start_long_task(command, self.auto_run_waypoints)
        elif command == 'compute':
            self.start_long_task(command, self.compute_and_save)
        elif command == 'list':
            self.list_waypoints()
            self.list_samples()
        elif command in ('save_rotation', 'rotation_sample'):
            self.save_manual_rotation_sample()
        elif command in ('write_rotation', 'rotation_write'):
            self.write_manual_rotation_samples()
        elif command == 'mark_bad':
            self.mark_current_bad_zone()
        elif command == 'mark_bad_stop':
            self.mark_current_bad_zone()
            self.cancel_long_task = True
            self.get_logger().warning('requested stop after manual bad-zone mark')
        elif command == 'stop':
            self.cancel_long_task = True
            self.get_logger().warning('requested stop for current long task')
        else:
            self.get_logger().warning(
                'unknown /delta_charuco/command "%s"; use discover, discover_calibrate, auto, compute, save_rotation, write_rotation, mark_bad, mark_bad_stop, stop, or list'
                % command
            )

    def start_long_task(self, name, func, *args):
        if self.long_task_running:
            self.get_logger().warning('busy: long task already running; ignored command %s' % name)
            return False

        def work():
            self.long_task_running = True
            self.cancel_long_task = False
            try:
                self.get_logger().info('starting long task: %s' % name)
                func(*args)
            except Exception as exc:
                self.get_logger().error('long task %s failed: %s' % (name, exc))
            finally:
                self.long_task_running = False
                self.cancel_long_task = False
                self.get_logger().info('finished long task: %s' % name)

        threading.Thread(target=work, daemon=True).start()
        return True

    def mark_current_bad_zone(self):
        latest_corners = int(self.latest_detect_debug.get('charuco_corners', 0))
        self.add_bad_zone(self.delta_xyz_mm, latest_corners)
        self.save_bad_zones()
        self.add_corner_observation(
            self.delta_xyz_mm,
            latest_corners,
            markers=int(self.latest_detect_debug.get('markers', 0)),
            accepted=False,
        )
        self.get_logger().warning(
            'MANUAL BAD ZONE: marked current delta pose %s mm; radius=%.1f mm; bad_zones=%d'
            % (
                np.round(self.delta_xyz_mm, 1).tolist(),
                self.discover_bad_zone_radius_mm(),
                len(self.bad_zones),
            )
        )

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

    def manual_rotation_path(self):
        return Path(os.path.expanduser(str(self.get_parameter('manual_rotation_path').value)))

    def bad_zone_path(self):
        return Path(os.path.expanduser(str(self.get_parameter('bad_zone_path').value)))

    def corner_observation_path(self):
        return Path(os.path.expanduser(str(self.get_parameter('corner_observation_path').value)))

    def discovery_progress_path(self):
        return Path(os.path.expanduser(str(self.get_parameter('discovery_progress_path').value)))

    def discovery_dashboard_path(self):
        return Path(os.path.expanduser(str(self.get_parameter('discovery_dashboard_path').value)))

    def hold_after_auto_run(self):
        return bool(self.get_parameter('hold_after_auto_run').value)

    def motion_safety_enabled(self):
        return bool(self.get_parameter('motion_safety_enabled').value)

    def safe_xy_z_mm(self):
        return float(self.get_parameter('safe_xy_z_mm').value)

    def jog_step_xy_mm(self):
        return float(self.get_parameter('jog_step_xy_mm').value)

    def jog_step_z_mm(self):
        return float(self.get_parameter('jog_step_z_mm').value)

    def workspace_bounds(self):
        return {
            'x_min': float(self.get_parameter('workspace_min_x_mm').value),
            'x_max': float(self.get_parameter('workspace_max_x_mm').value),
            'y_min': float(self.get_parameter('workspace_min_y_mm').value),
            'y_max': float(self.get_parameter('workspace_max_y_mm').value),
            'z_min': float(self.get_parameter('workspace_min_z_mm').value),
            'z_max': float(self.get_parameter('workspace_max_z_mm').value),
        }

    def move_settle_sec(self):
        return float(self.get_parameter('move_settle_sec').value)

    def home_between_samples(self):
        return bool(self.get_parameter('home_between_samples').value)

    def home_before_discovery(self):
        return bool(self.get_parameter('home_before_discovery').value)

    def home_settle_sec(self):
        return float(self.get_parameter('home_settle_sec').value)

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

    def manual_rotation_waypoint_index(self):
        return max(1, int(self.get_parameter('manual_rotation_waypoint_index').value))

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

    def load_bad_zones(self, log_missing=True):
        path = self.bad_zone_path()
        if not path.exists():
            if log_missing:
                self.get_logger().warning(f'bad-zone file not found: {path}')
            return False
        try:
            with path.open('r', encoding='utf-8') as file:
                data = yaml.safe_load(file) or {}
            zones = []
            for item in data.get('bad_zones', []):
                zones.append({
                    'x_mm': float(item['x_mm']),
                    'y_mm': float(item['y_mm']),
                    'z_mm': float(item.get('z_mm', self.discover_z_mm())),
                    'radius_mm': float(item.get('radius_mm', self.discover_bad_zone_radius_mm())),
                    'max_corners': int(item.get('max_corners', 0)),
                    'count': int(item.get('count', 1)),
                    'time_sec': float(item.get('time_sec', 0.0)),
                })
            self.bad_zones = zones
            self.get_logger().info(f'loaded {len(self.bad_zones)} bad discovery zones from {path}')
            return True
        except Exception as exc:
            self.get_logger().error(f'failed to load bad zones from {path}: {exc}')
            return False

    def save_bad_zones(self):
        path = self.bad_zone_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'description': 'Delta ChArUco discovery zones that failed detection; candidate generation skips nearby XY points.',
            'bad_zones': self.bad_zones,
        }
        with path.open('w', encoding='utf-8') as file:
            yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
        self.get_logger().info(f'saved {len(self.bad_zones)} bad discovery zones to {path}')

    def load_corner_observations(self, log_missing=True):
        path = self.corner_observation_path()
        if not path.exists():
            if log_missing:
                self.get_logger().warning(f'corner-observation file not found: {path}')
            return False
        try:
            with path.open('r', encoding='utf-8') as file:
                data = yaml.safe_load(file) or {}
            observations = []
            for item in data.get('observations', []):
                observations.append({
                    'x_mm': float(item['x_mm']),
                    'y_mm': float(item['y_mm']),
                    'z_mm': float(item.get('z_mm', self.discover_z_mm())),
                    'corners': int(item.get('corners', 0)),
                    'markers': int(item.get('markers', 0)),
                    'accepted': bool(item.get('accepted', False)),
                    'time_sec': float(item.get('time_sec', 0.0)),
                })
            self.corner_observations = observations
            self.get_logger().info(f'loaded {len(self.corner_observations)} corner observations from {path}')
            return True
        except Exception as exc:
            self.get_logger().error(f'failed to load corner observations from {path}: {exc}')
            return False

    def save_corner_observations(self):
        path = self.corner_observation_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'description': 'Delta ChArUco discovery observations. Higher corners means this region is more useful.',
            'observations': self.corner_observations,
        }
        with path.open('w', encoding='utf-8') as file:
            yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)

    def add_corner_observation(self, xyz, corners, markers=0, accepted=False):
        point = np.array(xyz, dtype=float)
        self.corner_observations.append({
            'x_mm': float(point[0]),
            'y_mm': float(point[1]),
            'z_mm': float(point[2]),
            'corners': int(corners),
            'markers': int(markers),
            'accepted': bool(accepted),
            'time_sec': float(time.time()),
        })
        # Keep the file useful without letting it grow forever.
        self.corner_observations = self.corner_observations[-2000:]
        self.save_corner_observations()

    def progress_bar(self, done, total, width=24):
        total = max(1, int(total))
        done = max(0, min(int(done), total))
        filled = int(round(width * done / total))
        return '[' + '#' * filled + '-' * (width - filled) + ']'

    def accepted_sample_pairs(self):
        pairs = []
        for index, sample in enumerate(self.samples, start=1):
            pairs.append({
                'index': index,
                'camera_xyz_m': sample.get('camera_xyz_m', []),
                'delta_xyz_mm': sample.get('delta_xyz_mm', []),
                'corner_count': int(sample.get('corner_count', 0)),
            })
        return pairs

    def corner_heatmap_summary(self, cell_mm=20.0):
        cells = {}
        for item in self.corner_observations:
            x = float(item['x_mm'])
            y = float(item['y_mm'])
            z = float(item.get('z_mm', self.discover_z_mm()))
            key = (
                int(round(x / cell_mm) * cell_mm),
                int(round(y / cell_mm) * cell_mm),
                int(round(z)),
            )
            cell = cells.setdefault(key, {
                'x_mm': key[0],
                'y_mm': key[1],
                'z_mm': key[2],
                'count': 0,
                'accepted_count': 0,
                'corner_sum': 0,
                'max_corners': 0,
                'min_corners': None,
            })
            corners = int(item.get('corners', 0))
            cell['count'] += 1
            cell['corner_sum'] += corners
            cell['max_corners'] = max(cell['max_corners'], corners)
            cell['min_corners'] = corners if cell['min_corners'] is None else min(cell['min_corners'], corners)
            if bool(item.get('accepted', False)):
                cell['accepted_count'] += 1

        summaries = []
        for cell in cells.values():
            avg = float(cell['corner_sum']) / max(1, int(cell['count']))
            summaries.append({
                'center_xyz_mm': [cell['x_mm'], cell['y_mm'], cell['z_mm']],
                'count': int(cell['count']),
                'accepted_count': int(cell['accepted_count']),
                'avg_corners': round(avg, 2),
                'max_corners': int(cell['max_corners']),
                'min_corners': int(cell['min_corners'] or 0),
            })
        hot = sorted(summaries, key=lambda item: (item['max_corners'], item['avg_corners'], item['count']), reverse=True)[:12]
        cold = sorted(summaries, key=lambda item: (item['max_corners'], item['avg_corners'], -item['count']))[:12]
        return {
            'cell_mm': cell_mm,
            'observed_cell_count': len(summaries),
            'hot_regions': hot,
            'cold_regions': cold,
        }

    def write_discovery_dashboard(self, progress_data):
        path = self.discovery_dashboard_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        total = int(progress_data.get('max_probes', progress_data.get('total_candidates', 0)))
        probed = int(progress_data.get('probed', 0))
        accepted = int(progress_data.get('accepted', 0))
        rejected = int(progress_data.get('rejected', 0))
        target = int(progress_data.get('target_waypoints', self.discovery_target_count()))
        lines = [
            'Delta ChArUco Discovery Dashboard',
            '=================================',
            'state: %s' % progress_data.get('state', 'unknown'),
            'progress: %s %d/%d (%s%%)' % (
                self.progress_bar(probed, max(1, total), width=32),
                probed,
                total,
                progress_data.get('progress_percent', 0),
            ),
            'visible waypoints: %d/%d   rejected probes: %d   bad zones: %d' % (
                accepted,
                target,
                rejected,
                int(progress_data.get('bad_zones', len(self.bad_zones))),
            ),
            'current xyz mm: %s' % np.round(np.array(progress_data.get('current_xyz_mm', self.delta_xyz_mm), dtype=float), 1).tolist(),
            'latest corners: %s   latest markers: %s' % (
                progress_data.get('latest_corners', self.latest_detect_debug.get('charuco_corners', 0)),
                progress_data.get('latest_markers', self.latest_detect_debug.get('markers', 0)),
            ),
            'accepted by z: %s' % progress_data.get('accepted_by_z', {}),
            '',
            'Accepted coordinate pairs for matrix fit',
            '---------------------------------------',
        ]
        next_action = str(progress_data.get('next_action', '')).strip()
        if next_action:
            lines.extend([
                'NEXT ACTION',
                '-----------',
                next_action,
                '',
            ])
        pairs = self.accepted_sample_pairs()
        if not pairs:
            lines.append('(none yet)')
        for pair in pairs[-20:]:
            lines.append(
                '%02d corners=%d  camera_m=%s  delta_mm=%s'
                % (
                    pair['index'],
                    pair['corner_count'],
                    np.round(np.array(pair['camera_xyz_m'], dtype=float), 4).tolist(),
                    np.round(np.array(pair['delta_xyz_mm'], dtype=float), 1).tolist(),
                )
            )

        heatmap = self.corner_heatmap_summary()
        lines.extend([
            '',
            'Corner heatmap: hottest regions',
            '-------------------------------',
        ])
        if not heatmap['hot_regions']:
            lines.append('(none yet)')
        for item in heatmap['hot_regions']:
            lines.append(
                'xyz=%s avg=%.2f max=%d count=%d accepted=%d'
                % (
                    item['center_xyz_mm'],
                    item['avg_corners'],
                    item['max_corners'],
                    item['count'],
                    item['accepted_count'],
                )
            )

        lines.extend([
            '',
            'Corner heatmap: cold/weak regions',
            '---------------------------------',
        ])
        if not heatmap['cold_regions']:
            lines.append('(none yet)')
        for item in heatmap['cold_regions']:
            lines.append(
                'xyz=%s avg=%.2f max=%d count=%d accepted=%d'
                % (
                    item['center_xyz_mm'],
                    item['avg_corners'],
                    item['max_corners'],
                    item['count'],
                    item['accepted_count'],
                )
            )

        lines.extend([
            '',
            'Files',
            '-----',
            'progress_yaml: %s' % self.discovery_progress_path(),
            'corner_observations_yaml: %s' % self.corner_observation_path(),
            'bad_zones_yaml: %s' % self.bad_zone_path(),
            'waypoints_yaml: %s' % self.waypoint_path(),
        ])
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def write_discovery_progress(self, **kwargs):
        path = self.discovery_progress_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        heatmap = self.corner_heatmap_summary()
        data = {
            'time_sec': float(time.time()),
            'accepted_sample_pairs': self.accepted_sample_pairs(),
            'corner_heatmap': heatmap,
            **kwargs,
        }
        with path.open('w', encoding='utf-8') as file:
            yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
        self.write_discovery_dashboard(data)

    def clear_waypoints(self):
        self.waypoints = []
        self.current_waypoint_index = -1
        self.get_logger().info('cleared all in-memory waypoints; press p to record your 9 safe calibration poses')

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

    def discovery_target_count(self):
        return max(1, int(self.get_parameter('discover_target_waypoints').value))

    def discover_min_corners(self):
        return max(1, int(self.get_parameter('discover_min_corners').value))

    def discover_z_mm(self):
        return float(self.get_parameter('discover_z_mm').value)

    def discover_z_levels_down(self):
        return max(1, int(self.get_parameter('discover_z_levels_down').value))

    def discover_z_step_mm(self):
        return max(1.0, float(self.get_parameter('discover_z_step_mm').value))

    def discover_z_values(self):
        return [
            self.discover_z_mm() - level * self.discover_z_step_mm()
            for level in range(self.discover_z_levels_down())
        ]

    def discover_grid_step_xy_mm(self):
        return max(1.0, float(self.get_parameter('discover_grid_step_xy_mm').value))

    def discover_bounds_margin_mm(self):
        return max(0.0, float(self.get_parameter('discover_bounds_margin_mm').value))

    def discover_bad_zone_radius_mm(self):
        return max(1.0, float(self.get_parameter('discover_bad_zone_radius_mm').value))

    def discover_bad_zone_max_corners(self):
        return max(0, int(self.get_parameter('discover_bad_zone_max_corners').value))

    def discover_adaptive_radius_mm(self):
        return max(1.0, float(self.get_parameter('discover_adaptive_radius_mm').value))

    def discover_max_probes(self):
        return max(1, int(self.get_parameter('discover_max_probes').value))

    def is_in_bad_zone(self, xyz):
        point = np.array(xyz, dtype=float)
        for zone in self.bad_zones:
            if abs(float(zone.get('z_mm', point[2])) - point[2]) > self.discover_z_step_mm() * 0.5:
                continue
            center = np.array([float(zone['x_mm']), float(zone['y_mm'])], dtype=float)
            radius = min(float(zone.get('radius_mm', self.discover_bad_zone_radius_mm())), self.discover_bad_zone_radius_mm())
            if float(np.linalg.norm(point[:2] - center)) <= radius:
                return True
        return False

    def add_bad_zone(self, xyz, max_corners):
        center = np.array(xyz, dtype=float)
        radius = self.discover_bad_zone_radius_mm()
        for zone in self.bad_zones:
            if abs(float(zone.get('z_mm', center[2])) - center[2]) > self.discover_z_step_mm() * 0.5:
                continue
            zone_center = np.array([float(zone['x_mm']), float(zone['y_mm'])], dtype=float)
            if float(np.linalg.norm(center[:2] - zone_center)) <= radius:
                zone['count'] = int(zone.get('count', 1)) + 1
                zone['max_corners'] = max(int(zone.get('max_corners', 0)), int(max_corners))
                zone['time_sec'] = float(time.time())
                return
        self.bad_zones.append({
            'x_mm': float(center[0]),
            'y_mm': float(center[1]),
            'z_mm': float(center[2]),
            'radius_mm': float(radius),
            'max_corners': int(max_corners),
            'count': 1,
            'time_sec': float(time.time()),
        })

    def discovery_candidate_waypoints(self):
        bounds = self.workspace_bounds()
        margin = self.discover_bounds_margin_mm()
        step = self.discover_grid_step_xy_mm()
        x_min = bounds['x_min'] + margin
        x_max = bounds['x_max'] - margin
        y_min = bounds['y_min'] + margin
        y_max = bounds['y_max'] - margin
        if x_min > x_max or y_min > y_max:
            self.get_logger().error('discovery bounds invalid after margin %.1f mm' % margin)
            return []

        candidates = []
        center = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], dtype=float)
        z_values = self.discover_z_values()
        for z_mm in z_values:
            y_values = np.arange(y_min, y_max + 0.5 * step, step)
            for row_index, y_mm in enumerate(y_values):
                x_values = np.arange(x_min, x_max + 0.5 * step, step)
                if row_index % 2 == 1:
                    x_values = x_values[::-1]
                for x_mm in x_values:
                    xyz = np.array([float(x_mm), float(y_mm), float(z_mm)], dtype=float)
                    if self.validate_target(xyz) and not self.is_in_bad_zone(xyz):
                        dist = float(np.linalg.norm(xyz[:2] - center)) + abs(z_mm - self.discover_z_mm()) * 0.3
                        candidates.append((dist, xyz))

        # Start near the camera-centered part of the workspace so good points are found early.
        candidates.sort(key=lambda item: item[0])
        return [xyz for _dist, xyz in candidates]

    def estimate_candidate_corner_score(self, xyz, observations):
        combined = []
        combined.extend(observations)
        for item in self.corner_observations:
            combined.append((
                np.array([float(item['x_mm']), float(item['y_mm']), float(item['z_mm'])], dtype=float),
                int(item.get('corners', 0)),
            ))
        if not combined:
            return 0.0
        point = np.array(xyz, dtype=float)
        radius = self.discover_adaptive_radius_mm()
        total_weight = 0.0
        weighted_score = 0.0
        for obs_xyz, corners in combined:
            obs = np.array(obs_xyz, dtype=float)
            dxy = float(np.linalg.norm(point[:2] - obs[:2]))
            dz = abs(float(point[2] - obs[2])) * 0.7
            distance = math.hypot(dxy, dz)
            if distance > radius:
                continue
            weight = 1.0 / (distance + 5.0)
            total_weight += weight
            weighted_score += float(corners) * weight
        if total_weight <= 0.0:
            return 0.0
        return weighted_score / total_weight

    def pop_next_discovery_candidate(self, candidates, observations, preferred_z_values=None):
        best_index = None
        best_score = None
        preferred = None
        if preferred_z_values:
            preferred = [float(z) for z in preferred_z_values]
            has_preferred = any(
                (not self.is_in_bad_zone(xyz))
                and any(abs(float(xyz[2]) - z) <= self.discover_z_step_mm() * 0.25 for z in preferred)
                for xyz in candidates
            )
            if not has_preferred:
                preferred = None
        bounds = self.workspace_bounds()
        center = np.array(
            [
                (bounds['x_min'] + bounds['x_max']) * 0.5,
                (bounds['y_min'] + bounds['y_max']) * 0.5,
                self.discover_z_mm(),
            ],
            dtype=float,
        )
        for index, xyz in enumerate(candidates):
            if self.is_in_bad_zone(xyz):
                continue
            if preferred is not None and not any(
                abs(float(xyz[2]) - z) <= self.discover_z_step_mm() * 0.25 for z in preferred
            ):
                continue
            corner_score = self.estimate_candidate_corner_score(xyz, observations)
            center_penalty = float(np.linalg.norm((np.array(xyz, dtype=float) - center) / np.array([90.0, 90.0, 80.0])))
            score = corner_score * 10.0 - center_penalty
            if best_score is None or score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            return None
        return candidates.pop(best_index)

    def wait_for_stable_detection_result(self, min_image_seq, min_corners=None):
        timeout_sec = self.post_move_detect_timeout_sec()
        stable_needed = self.stable_detection_frames()
        tolerance_m = self.stable_detection_tolerance_mm() / 1000.0
        required_corners = self.discover_min_corners() if min_corners is None else int(min_corners)
        end_time = time.time() + timeout_sec
        last_detection_seq = -1
        last_tvec = None
        stable_count = 0
        stable_detections = []

        while rclpy.ok() and time.time() < end_time:
            if threading.get_ident() == self.spin_thread_id:
                rclpy.spin_once(self, timeout_sec=0.05)
            else:
                time.sleep(0.05)
            detection = self.latest_detection
            if detection is None:
                stable_count = 0
                last_tvec = None
                continue
            if int(detection.get('corner_count', 0)) < required_corners:
                stable_count = 0
                last_tvec = None
                continue
            if self.latest_detection_seq <= min_image_seq:
                continue
            if self.latest_detection_seq == last_detection_seq:
                continue

            tvec = detection['tvec'].reshape(3).astype(float)
            if last_tvec is None:
                stable_count = 1
                stable_detections = [detection]
            else:
                shift = float(np.linalg.norm(tvec - last_tvec))
                if shift <= tolerance_m:
                    stable_count += 1
                    stable_detections.append(detection)
                else:
                    stable_count = 1
                    stable_detections = [detection]
            last_tvec = tvec
            last_detection_seq = self.latest_detection_seq

            if stable_count >= stable_needed:
                accepted_detection = dict(stable_detections[-1])
                stable_tvecs = np.array(
                    [item['tvec'].reshape(3).astype(float) for item in stable_detections[-stable_needed:]],
                    dtype=float,
                )
                accepted_detection['tvec'] = np.median(stable_tvecs, axis=0).reshape(3, 1)
                return {
                    'corner_count': int(accepted_detection['corner_count']),
                    'camera_xyz_m': accepted_detection['tvec'].reshape(3).astype(float).tolist(),
                    'image_seq': int(self.latest_detection_seq),
                    'detection': accepted_detection,
                }

        return None

    def discover_visible_waypoints(self, compute_after=None):
        target_count = self.discovery_target_count()
        min_corners = self.discover_min_corners()
        save_samples = bool(self.get_parameter('discover_save_samples').value)
        if compute_after is None:
            compute_after = bool(self.get_parameter('discover_compute_after').value)

        candidates = self.discovery_candidate_waypoints()
        if not candidates:
            self.get_logger().error('no discovery candidates generated')
            return False
        total_candidates = len(candidates)

        self.get_logger().info(
            'discovering up to %d visible calibration waypoints: candidates=%d, z=%.1f mm, step=%.1f mm, min_corners=%d'
            % (
                target_count,
                len(candidates),
                self.discover_z_mm(),
                self.discover_grid_step_xy_mm(),
                min_corners,
            )
        )
        self.publish_motor_enable(True)
        if self.home_before_discovery():
            self.home_and_wait('automatic discovery')
            if self.cancel_long_task:
                self.get_logger().warning('discovery cancelled during startup homing')
                return False
        discovered = []
        saved_before = len(self.samples)
        observations = []
        probe_index = 0
        rejected = 0
        max_probes = min(self.discover_max_probes(), total_candidates)
        z_values = self.discover_z_values()
        per_z_target = int(math.ceil(float(target_count) / max(1, len(z_values))))
        accepted_by_z = {str(int(round(z))): 0 for z in z_values}

        while candidates:
            if not rclpy.ok() or self.cancel_long_task or len(discovered) >= target_count or probe_index >= max_probes:
                break
            preferred_z_values = [
                z for z in z_values
                if accepted_by_z.get(str(int(round(z))), 0) < per_z_target
            ]
            xyz = self.pop_next_discovery_candidate(candidates, observations, preferred_z_values)
            if xyz is None:
                break
            if self.is_in_bad_zone(xyz):
                continue
            probe_index += 1

            image_seq_before_move = self.image_seq
            percent_done = int(round(100.0 * probe_index / max(1, max_probes)))
            self.get_logger().info(
                'DISCOVERY %s %d/%d (%d%%) accepted=%d/%d rejected=%d bad_zones=%d -> %s mm'
                % (
                    self.progress_bar(probe_index, max_probes),
                    probe_index,
                    max_probes,
                    percent_done,
                    len(discovered),
                    target_count,
                    rejected,
                    len(self.bad_zones),
                    np.round(xyz, 1).tolist(),
                )
            )
            self.write_discovery_progress(
                state='moving',
                total_candidates=total_candidates,
                max_probes=max_probes,
                probed=probe_index,
                accepted=len(discovered),
                rejected=rejected,
                target_waypoints=target_count,
                remaining_candidates=len(candidates),
                bad_zones=len(self.bad_zones),
                accepted_by_z=dict(accepted_by_z),
                current_xyz_mm=xyz.astype(float).tolist(),
                progress_percent=percent_done,
            )
            if not self.publish_move(xyz):
                continue
            self.wait_with_spin(self.move_settle_sec())
            result = self.wait_for_stable_detection_result(image_seq_before_move, min_corners)
            if result is None:
                rejected += 1
                latest_corners = int(self.latest_detect_debug.get('charuco_corners', 0))
                latest_markers = int(self.latest_detect_debug.get('markers', 0))
                observations.append((xyz.astype(float).copy(), latest_corners))
                self.add_corner_observation(xyz, latest_corners, markers=latest_markers, accepted=False)
                if latest_corners <= self.discover_bad_zone_max_corners():
                    self.add_bad_zone(xyz, latest_corners)
                    self.save_bad_zones()
                self.write_discovery_progress(
                    state='rejected',
                    total_candidates=total_candidates,
                    max_probes=max_probes,
                    probed=probe_index,
                    accepted=len(discovered),
                    rejected=rejected,
                    target_waypoints=target_count,
                    remaining_candidates=len(candidates),
                    bad_zones=len(self.bad_zones),
                    accepted_by_z=dict(accepted_by_z),
                    current_xyz_mm=xyz.astype(float).tolist(),
                    latest_corners=latest_corners,
                    latest_markers=latest_markers,
                    progress_percent=int(round(100.0 * probe_index / max(1, max_probes))),
                )
                self.get_logger().warning(
                    'probe rejected: no stable board at %s; latest markers=%s corners=%s; bad_zones=%d'
                    % (
                        np.round(xyz, 1).tolist(),
                        latest_markers,
                        latest_corners,
                        len(self.bad_zones),
                    )
                )
                continue

            observations.append((xyz.astype(float).copy(), int(result['corner_count'])))
            self.add_corner_observation(
                xyz,
                int(result['corner_count']),
                markers=int(self.latest_detect_debug.get('markers', 0)),
                accepted=True,
            )
            waypoint = {
                'name': 'auto_pt_%02d' % (len(discovered) + 1),
                'xyz_mm': xyz.astype(float).copy(),
            }
            discovered.append(waypoint)
            z_key = str(int(round(float(xyz[2]))))
            accepted_by_z[z_key] = int(accepted_by_z.get(z_key, 0)) + 1
            self.write_discovery_progress(
                state='accepted',
                total_candidates=total_candidates,
                max_probes=max_probes,
                probed=probe_index,
                accepted=len(discovered),
                rejected=rejected,
                target_waypoints=target_count,
                remaining_candidates=len(candidates),
                bad_zones=len(self.bad_zones),
                accepted_by_z=dict(accepted_by_z),
                current_xyz_mm=xyz.astype(float).tolist(),
                latest_corners=int(result['corner_count']),
                progress_percent=int(round(100.0 * probe_index / max(1, max_probes))),
            )
            self.get_logger().info(
                'probe accepted as %s: delta=%s corners=%d camera=%s'
                % (
                    waypoint['name'],
                    np.round(xyz, 1).tolist(),
                    result['corner_count'],
                    np.round(np.array(result['camera_xyz_m'], dtype=float), 4).tolist(),
                )
            )
            if save_samples:
                self.save_sample(detection=result.get('detection'))
                if len(discovered) < target_count:
                    self.home_and_wait_if_enabled('discovery sample %d' % len(self.samples))
                self.write_discovery_progress(
                    state='accepted_sample_saved',
                    total_candidates=total_candidates,
                    max_probes=max_probes,
                    probed=probe_index,
                    accepted=len(discovered),
                    rejected=rejected,
                    target_waypoints=target_count,
                    remaining_candidates=len(candidates),
                    bad_zones=len(self.bad_zones),
                    accepted_by_z=dict(accepted_by_z),
                    current_xyz_mm=xyz.astype(float).tolist(),
                    latest_corners=int(result['corner_count']),
                    progress_percent=int(round(100.0 * probe_index / max(1, max_probes))),
                )

        if not discovered:
            self.get_logger().error('discovery failed: no visible waypoints found')
            return False

        self.waypoints = discovered
        self.current_waypoint_index = 0
        self.save_waypoints()
        self.log_box(
            'DISCOVERY FINISHED',
            [
                'saved waypoints: %d/%d' % (len(discovered), target_count),
                'waypoint file: %s' % self.waypoint_path(),
                'samples saved during discovery: %d' % (len(self.samples) - saved_before),
                'next run can use command "auto" to sample these fixed points',
            ],
        )
        self.write_discovery_progress(
            state='stopped' if self.cancel_long_task else 'rotation_ready',
            total_candidates=total_candidates,
            max_probes=max_probes,
            probed=probe_index,
            accepted=len(discovered),
            rejected=rejected,
            target_waypoints=target_count,
            remaining_candidates=len(candidates),
            bad_zones=len(self.bad_zones),
            accepted_by_z=dict(accepted_by_z),
            progress_percent=100 if len(discovered) >= target_count else int(round(100.0 * probe_index / max(1, max_probes))),
            waypoint_path=str(self.waypoint_path()),
            bad_zone_path=str(self.bad_zone_path()),
            corner_observation_path=str(self.corner_observation_path()),
            next_action=(
                'Automatic translation calibration sampling is finished. Rotation-axis sampling is optional '
                'and independent; use the Board Rotation buttons only when you intentionally want that data.'
            ) if not self.cancel_long_task and len(discovered) >= target_count else '',
        )

        if compute_after and len(self.samples) >= 4:
            self.compute_and_save()
        elif compute_after:
            self.get_logger().warning(
                'not enough samples to compute calibration after discovery: %d saved, need >=4'
                % len(self.samples)
            )

        if self.hold_after_auto_run():
            self.hold_current_pose()
        return True

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

    def publish_home(self):
        self.delta_home_pub.publish(Empty())
        self.get_logger().info('sent home command G28')

    def home_and_wait(self, reason='requested'):
        self.get_logger().warning(
            'sending G28 before %s; waiting %.1f sec'
            % (reason, self.home_settle_sec())
        )
        self.publish_home()
        self.wait_with_spin(self.home_settle_sec())
        self.delta_xyz_mm = np.array(
            [
                float(self.get_parameter('home_x_mm').value),
                float(self.get_parameter('home_y_mm').value),
                float(self.get_parameter('home_z_mm').value),
            ],
            dtype=float,
        )
        self.get_logger().info('internal delta pose reset to home %s mm' % np.round(self.delta_xyz_mm, 1).tolist())

    def home_and_wait_if_enabled(self, reason='sample saved'):
        if not self.home_between_samples():
            return
        self.home_and_wait(reason)

    def publish_motor_enable(self, enabled):
        msg = Bool()
        msg.data = bool(enabled)
        self.motor_enable_pub.publish(msg)
        self.get_logger().info('sent motor %s' % ('ENABLE' if enabled else 'DISABLE'))

    def hold_current_pose(self):
        self.publish_motor_enable(True)
        self.publish_move(self.delta_xyz_mm)
        self.get_logger().info(
            'holding current delta pose at %s mm; rotate the board by hand and press v for each pose'
            % np.round(self.delta_xyz_mm, 1).tolist()
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

    def publish_move_unchecked(self, xyz_mm):
        if self.use_staged_motion():
            self.publish_staged_move(xyz_mm)
        else:
            self.publish_direct_move(xyz_mm)

    def validate_target(self, xyz_mm):
        target = np.array(xyz_mm, dtype=float)
        if not self.motion_safety_enabled():
            return True

        bounds = self.workspace_bounds()
        if not (bounds['x_min'] <= target[0] <= bounds['x_max']):
            self.get_logger().error(
                'blocked move: X %.1f mm outside [%.1f, %.1f]'
                % (target[0], bounds['x_min'], bounds['x_max'])
            )
            return False
        if not (bounds['y_min'] <= target[1] <= bounds['y_max']):
            self.get_logger().error(
                'blocked move: Y %.1f mm outside [%.1f, %.1f]'
                % (target[1], bounds['y_min'], bounds['y_max'])
            )
            return False
        if not (bounds['z_min'] <= target[2] <= bounds['z_max']):
            self.get_logger().error(
                'blocked move: Z %.1f mm outside [%.1f, %.1f]'
                % (target[2], bounds['z_min'], bounds['z_max'])
            )
            return False
        return True

    def needs_xy_guard_descent(self, target):
        if not self.motion_safety_enabled():
            return False
        current = np.array(self.delta_xyz_mm, dtype=float)
        xy_moves = np.linalg.norm(target[:2] - current[:2]) > 1e-6
        return xy_moves and current[2] > self.safe_xy_z_mm()

    def publish_move(self, xyz_mm):
        target = np.array(xyz_mm, dtype=float)
        if not self.validate_target(target):
            return False

        if self.needs_xy_guard_descent(target):
            guard = np.array([self.delta_xyz_mm[0], self.delta_xyz_mm[1], self.safe_xy_z_mm()], dtype=float)
            if not self.validate_target(guard):
                return False
            self.get_logger().warning(
                'XY move requested while Z=%.1f is above safe_xy_z=%.1f; descending Z first'
                % (self.delta_xyz_mm[2], self.safe_xy_z_mm())
            )
            self.publish_move_unchecked(guard)
            self.wait_with_spin(self.move_settle_sec())

        self.publish_move_unchecked(target)
        return True

    def jog(self, dx_mm=0.0, dy_mm=0.0, dz_mm=0.0):
        target = np.array(self.delta_xyz_mm, dtype=float) + np.array([dx_mm, dy_mm, dz_mm], dtype=float)
        if self.publish_move(target):
            self.get_logger().info('jog target %s mm' % np.round(target, 1).tolist())

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

    def move_to_manual_rotation_waypoint(self):
        index = self.manual_rotation_waypoint_index() - 1
        if not self.move_to_waypoint(index):
            return
        self.wait_with_spin(self.move_settle_sec())
        self.hold_current_pose()

    def wait_with_spin(self, duration_sec):
        end_time = time.time() + max(0.0, duration_sec)
        while rclpy.ok() and time.time() < end_time:
            if threading.get_ident() == self.spin_thread_id:
                rclpy.spin_once(self, timeout_sec=0.05)
            else:
                time.sleep(0.05)

    def wait_for_stable_detection(self, min_image_seq):
        timeout_sec = self.post_move_detect_timeout_sec()
        stable_needed = self.stable_detection_frames()
        tolerance_m = self.stable_detection_tolerance_mm() / 1000.0
        end_time = time.time() + timeout_sec
        last_detection_seq = -1
        last_tvec = None
        stable_count = 0

        while rclpy.ok() and time.time() < end_time:
            if threading.get_ident() == self.spin_thread_id:
                rclpy.spin_once(self, timeout_sec=0.05)
            else:
                time.sleep(0.05)
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
            if index < end_index - 1:
                self.home_and_wait_if_enabled('auto waypoint %d' % (index + 1))
        self.get_logger().info(
            'auto-run finished: saved %d new samples'
            % (len(self.samples) - saved_before)
        )
        if self.hold_after_auto_run():
            self.hold_current_pose()

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
        self.publish_debug_image(msg.header)
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
        min_corners = int(self.get_parameter('min_charuco_corners').value)
        lines = [
                        f'corners: {corner_count}  (>={min_corners} ok, more is better)',
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
        self.latest_raw_detection = {
            'charuco_corners': charuco_corners,
            'charuco_ids': charuco_ids,
            'marker_corners': marker_corners,
            'marker_ids': marker_ids,
            'corner_count': charuco_count,
            'frame_id': frame_id,
        }
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

        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        reprojection_errors = np.linalg.norm(
            projected.reshape(-1, 2) - image_points.reshape(-1, 2),
            axis=1,
        )
        mean_reprojection_error = float(np.mean(reprojection_errors))
        max_reprojection_error = float(np.max(reprojection_errors))
        max_allowed_error = float(self.get_parameter('max_reprojection_error_px').value)
        if mean_reprojection_error > max_allowed_error:
            self.latest_detect_debug = {
                'markers': marker_count,
                'charuco_corners': charuco_count,
                'reason': 'reprojection error %.2f px > %.2f px' % (
                    mean_reprojection_error,
                    max_allowed_error,
                ),
            }
            return None

        return {
            'rvec': rvec.reshape(3, 1),
            'tvec': tvec.reshape(3, 1),
            'corner_count': len(ids),
            'mean_reprojection_error_px': mean_reprojection_error,
            'max_reprojection_error_px': max_reprojection_error,
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

    def publish_debug_image(self, header):
        if not bool(self.get_parameter('publish_debug_image').value):
            return
        if self.latest_image is None:
            return

        debug = self.latest_image.copy()
        detection = self.latest_detection if self.latest_detection is not None else self.latest_raw_detection
        if detection is not None:
            self.draw_detection_overlay(debug, detection)
        if self.latest_detection is not None:
            cv2.drawFrameAxes(
                debug,
                self.camera_matrix,
                self.dist_coeffs,
                self.latest_detection['rvec'],
                self.latest_detection['tvec'],
                0.05,
            )
            text = 'ChArUco OK corners=%d' % int(self.latest_detection['corner_count'])
            color = (0, 220, 0)
        else:
            text = 'NO BOARD markers=%d corners=%d' % (
                int(self.latest_detect_debug.get('markers', 0)),
                int(self.latest_detect_debug.get('charuco_corners', 0)),
            )
            color = (0, 0, 255)

        cv2.putText(
            debug,
            text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        try:
            msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            msg.header = header
            self.debug_image_pub.publish(msg)
        except Exception as exc:
            self.get_logger().warning(f'failed to publish debug image: {exc}')

    def draw_detection_overlay(self, image, detection):
        marker_corners = detection.get('marker_corners')
        marker_ids = detection.get('marker_ids')
        charuco_corners = detection.get('charuco_corners')
        charuco_ids = detection.get('charuco_ids')

        if marker_corners is not None and marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(image, marker_corners, marker_ids)
            ids = marker_ids.reshape(-1).astype(int)
            for corners, marker_id in zip(marker_corners, ids):
                pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
                pts_i = pts.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(image, [pts_i], True, (0, 255, 255), 3, cv2.LINE_AA)
                center = np.mean(pts, axis=0).astype(int)
                cv2.circle(image, tuple(center), 5, (0, 255, 255), -1)
                cv2.circle(image, tuple(center), 7, (0, 0, 0), 2)
                cv2.putText(
                    image,
                    f'M{marker_id}',
                    (int(center[0]) + 4, int(center[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    f'M{marker_id}',
                    (int(center[0]) + 4, int(center[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 0:
            cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)
            ids = charuco_ids.reshape(-1).astype(int)
            pts = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
            for point, corner_id in zip(pts, ids):
                x_px, y_px = int(point[0]), int(point[1])
                cv2.circle(image, (x_px, y_px), 6, (255, 0, 255), -1)
                cv2.circle(image, (x_px, y_px), 8, (255, 255, 255), 2)
                cv2.putText(
                    image,
                    str(corner_id),
                    (x_px + 6, y_px + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    str(corner_id),
                    (x_px + 6, y_px + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

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

    def save_sample(self, detection=None):
        detection = self.latest_detection if detection is None else detection
        if detection is None:
            self.get_logger().warning('cannot save sample: ChArUco board is not detected')
            return False
        min_corners = int(self.get_parameter('min_charuco_corners').value)
        if int(detection.get('corner_count', 0)) < min_corners:
            self.get_logger().warning(
                'cannot save sample: only %d ChArUco corners; need at least %d'
                % (int(detection.get('corner_count', 0)), min_corners)
            )
            return False
        camera_xyz_m = detection['tvec'].reshape(3).astype(float)
        delta_xyz_m = self.delta_xyz_mm.astype(float) / 1000.0
        sample = {
            'camera_xyz_m': camera_xyz_m.tolist(),
            'delta_xyz_mm': self.delta_xyz_mm.astype(float).tolist(),
            'delta_xyz_m': delta_xyz_m.tolist(),
            'corner_count': int(detection['corner_count']),
            'mean_reprojection_error_px': float(detection.get('mean_reprojection_error_px', 0.0)),
            'max_reprojection_error_px': float(detection.get('max_reprojection_error_px', 0.0)),
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
        return True

    def save_manual_rotation_sample(self):
        if self.latest_detection is None:
            self.get_logger().warning('cannot save manual rotation sample: ChArUco board is not detected')
            return False

        rvec = self.latest_detection['rvec'].reshape(3).astype(float)
        tvec = self.latest_detection['tvec'].reshape(3).astype(float)
        r_mat, _ = cv2.Rodrigues(rvec)
        quat = self.rotation_matrix_to_quaternion(r_mat)
        charuco_ids = self.latest_detection.get('charuco_ids')
        marker_ids = self.latest_detection.get('marker_ids')

        sample = {
            'index': len(self.manual_rotation_samples) + 1,
            'time_sec': float(time.time()),
            'delta_xyz_mm': self.delta_xyz_mm.astype(float).tolist(),
            'board_tvec_camera_m': tvec.tolist(),
            'board_rvec_camera': rvec.tolist(),
            'board_rotation_camera': r_mat.tolist(),
            'board_quaternion_camera_xyzw': quat.tolist(),
            'corner_count': int(self.latest_detection['corner_count']),
            'charuco_ids': [] if charuco_ids is None else charuco_ids.reshape(-1).astype(int).tolist(),
            'marker_ids': [] if marker_ids is None else marker_ids.reshape(-1).astype(int).tolist(),
        }
        self.manual_rotation_samples.append(sample)
        self.write_manual_rotation_samples()
        self.log_box(
            f'SAVED MANUAL ROTATION SAMPLE {sample["index"]}',
            [
                'delta held at mm: %s' % np.round(self.delta_xyz_mm, 1).tolist(),
                'board_tvec_camera_m: %s' % np.round(tvec, 4).tolist(),
                'board_rvec_camera: %s' % np.round(rvec, 4).tolist(),
                f'corners: {sample["corner_count"]}',
                'rotate the board again and press v to record another pose',
            ],
        )
        return True

    def write_manual_rotation_samples(self):
        path = self.manual_rotation_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'description': (
                'Manual ChArUco board rotation captures at a fixed Delta pose. '
                'Use these full board poses for later pivot/TCP-offset calibration.'
            ),
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
            'sample_count': len(self.manual_rotation_samples),
            'samples': self.manual_rotation_samples,
        }
        with path.open('w', encoding='utf-8') as file:
            yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
        self.get_logger().info(f'wrote {len(self.manual_rotation_samples)} manual rotation samples to {path}')

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

        kept_indices = list(range(len(self.samples)))
        excluded_samples = []
        outlier_threshold_m = float(self.get_parameter('calibration_outlier_threshold_mm').value) / 1000.0

        for _iteration in range(3):
            if len(kept_indices) < 4:
                break
            camera_points_iter = np.array([self.samples[i]['camera_xyz_m'] for i in kept_indices], dtype=float)
            delta_points_iter = np.array([self.samples[i]['delta_xyz_m'] for i in kept_indices], dtype=float)
            r_iter, t_iter = fit_rigid_transform(camera_points_iter, delta_points_iter)
            predicted_iter = (r_iter @ camera_points_iter.T).T + t_iter
            errors_iter = np.linalg.norm(predicted_iter - delta_points_iter, axis=1)
            worst_local = int(np.argmax(errors_iter))
            worst_error = float(errors_iter[worst_local])
            if worst_error <= outlier_threshold_m:
                break
            sample_index = kept_indices.pop(worst_local)
            excluded_samples.append({
                'index': int(sample_index + 1),
                'reason': 'fit residual %.2f mm > %.2f mm' % (
                    worst_error * 1000.0,
                    outlier_threshold_m * 1000.0,
                ),
                'sample': self.samples[sample_index],
            })

        camera_points = np.array([self.samples[i]['camera_xyz_m'] for i in kept_indices], dtype=float)
        delta_points = np.array([self.samples[i]['delta_xyz_m'] for i in kept_indices], dtype=float)
        r_mat, t_vec = fit_rigid_transform(camera_points, delta_points)
        predicted = (r_mat @ camera_points.T).T + t_vec
        errors_m = np.linalg.norm(predicted - delta_points, axis=1)
        affine_xyz_coeffs, _affine_xyz_pred, affine_xyz_errors_m = fit_affine(
            camera_points,
            delta_points[:, :2],
        )
        affine_xy_coeffs, _affine_xy_pred, affine_xy_errors_m = fit_affine(
            camera_points[:, :2],
            delta_points[:, :2],
        )

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
            'planar_affine_camera_xyz_to_delta_xy': {
                'description': 'For single-Z Delta work plane use: delta_xy_m = [camera_x, camera_y, camera_z, 1] @ coeffs',
                'coeffs': affine_xyz_coeffs.tolist(),
                'delta_z_m': float(np.median(delta_points[:, 2])),
                'error': {
                    'count': int(len(affine_xyz_errors_m)),
                    'rmse_m': float(np.sqrt(np.mean(affine_xyz_errors_m ** 2))),
                    'mean_m': float(np.mean(affine_xyz_errors_m)),
                    'median_m': float(np.median(affine_xyz_errors_m)),
                    'max_m': float(np.max(affine_xyz_errors_m)),
                    'per_sample_m': affine_xyz_errors_m.tolist(),
                },
            },
            'planar_affine_camera_xy_to_delta_xy': {
                'description': 'For fixed-depth plane use: delta_xy_m = [camera_x, camera_y, 1] @ coeffs',
                'coeffs': affine_xy_coeffs.tolist(),
                'delta_z_m': float(np.median(delta_points[:, 2])),
                'error': {
                    'count': int(len(affine_xy_errors_m)),
                    'rmse_m': float(np.sqrt(np.mean(affine_xy_errors_m ** 2))),
                    'mean_m': float(np.mean(affine_xy_errors_m)),
                    'median_m': float(np.median(affine_xy_errors_m)),
                    'max_m': float(np.max(affine_xy_errors_m)),
                    'per_sample_m': affine_xy_errors_m.tolist(),
                },
            },
            'samples': [self.samples[i] for i in kept_indices],
            'excluded_samples': excluded_samples,
            'error': {
                'count': int(len(errors_m)),
                'rmse_m': float(np.sqrt(np.mean(errors_m ** 2))),
                'mean_m': float(np.mean(errors_m)),
                'median_m': float(np.median(errors_m)),
                'max_m': float(np.max(errors_m)),
                'per_sample_m': errors_m.tolist(),
                'source_sample_indices': [int(i + 1) for i in kept_indices],
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
        if result['error']['rmse_m'] > 0.005:
            self.get_logger().warning(
                'calibration RMSE %.2f mm is above 5.00 mm; inspect mechanics, pose diversity, board visibility, and outliers'
                % (result['error']['rmse_m'] * 1000.0)
            )
        return True

    def write_debug_image(self):
        if self.latest_image is None:
            self.get_logger().warning('no image received yet')
            return
        debug = self.latest_image.copy()
        detection = self.latest_detection if self.latest_detection is not None else self.latest_raw_detection
        if detection is not None:
            self.draw_detection_overlay(debug, detection)
        if self.latest_detection is not None:
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
        if not sys.stdin.isatty():
            return None
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
        self.spin_thread_id = threading.get_ident()
        if not sys.stdin.isatty():
            self.get_logger().warning('stdin is not a TTY; running without keyboard controls')
            rclpy.spin(self)
            return
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
            elif key == 'R':
                self.move_to_manual_rotation_waypoint()
            elif key == '0':
                self.publish_home()
            elif key == 'I':
                self.jog(dy_mm=self.jog_step_xy_mm())
            elif key == 'K':
                self.jog(dy_mm=-self.jog_step_xy_mm())
            elif key == 'J':
                self.jog(dx_mm=-self.jog_step_xy_mm())
            elif key == 'L':
                self.jog(dx_mm=self.jog_step_xy_mm())
            elif key == 'U':
                self.jog(dz_mm=self.jog_step_z_mm())
            elif key == 'O':
                self.jog(dz_mm=-self.jog_step_z_mm())
            elif key == 'm':
                self.clear_waypoints()
            elif key == 'h':
                self.hold_current_pose()
            elif key == 'e':
                self.publish_motor_enable(True)
            elif key == 'd':
                self.publish_motor_enable(False)
            elif key == 'v':
                self.save_manual_rotation_sample()
            elif key == 'x':
                self.write_manual_rotation_samples()
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
