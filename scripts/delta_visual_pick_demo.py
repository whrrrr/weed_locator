#!/usr/bin/env python3
"""Visual Delta pick demo using a colored target on a fixed work plane."""

import math
import select
import sys
import termios
import tty
from collections import deque

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Empty, String

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def load_transform(path):
    with open(path, 'r', encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    transform = np.array(data['T_delta_camera'], dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f'T_delta_camera in {path} is not 4x4')
    return transform


def parse_float_list(text):
    values = []
    for item in str(text).replace(';', ',').split(','):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def polynomial_terms(points_norm, degree):
    x = points_norm[:, 0]
    y = points_norm[:, 1]
    z = points_norm[:, 2]
    columns = [
        np.ones(points_norm.shape[0]),
        x,
        y,
        z,
    ]
    if degree >= 2:
        columns.extend([x * x, y * y, z * z, x * y, x * z, y * z])
    if degree >= 3:
        columns.extend([
            x * x * x,
            y * y * y,
            z * z * z,
            x * x * y,
            x * x * z,
            y * y * x,
            y * y * z,
            z * z * x,
            z * z * y,
            x * y * z,
        ])
    return np.column_stack(columns)


def load_empirical_model(path):
    with open(path, 'r', encoding='utf-8') as stream:
        data = yaml.safe_load(stream) or {}
    if data.get('model_type') != 'polynomial':
        raise ValueError(f'empirical model in {path} is not polynomial')
    model = {
        'path': path,
        'degree': int(data['degree']),
        'camera_mean_mm': np.array(data['camera_mean_mm'], dtype=float),
        'camera_scale_mm': np.array(data['camera_scale_mm'], dtype=float),
        'coefficients': np.array(data['coefficients'], dtype=float),
        'train_error': data.get('train_error', {}),
        'loocv_error': data.get('loocv_error', {}),
    }
    if model['coefficients'].ndim != 2 or model['coefficients'].shape[1] != 3:
        raise ValueError(f'empirical model coefficients in {path} must be Nx3')
    return model


class DeltaVisualPickDemo(Node):
    def __init__(self):
        super().__init__('delta_visual_pick_demo')

        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('depth_camera_info_topic', '/camera/depth/camera_info')
        self.declare_parameter('delta_move_topic', '/delta_arm/move_to')
        self.declare_parameter('trigger_topic', '/delta_visual_pick_demo/trigger')
        self.declare_parameter(
            'calibration_path',
            '/home/whr/cc_ws/tros_ws/calibration_targets/delta_hand_eye_filtered.yaml',
        )
        self.declare_parameter('use_layered_calibration', False)
        self.declare_parameter('layered_calibration_dir', '/home/whr/cc_ws/tros_ws/calibration_targets')
        self.declare_parameter('layered_calibration_prefix', 'delta_hand_eye_center_z')
        self.declare_parameter('layered_calibration_zs_mm', '-180,-190,-200,-210,-220,-230')
        self.declare_parameter('calibration_select_z_mm', float('nan'))
        self.declare_parameter('use_empirical_model', False)
        self.declare_parameter(
            'empirical_model_path',
            '/home/whr/cc_ws/tros_ws/calibration_targets/delta_hand_eye_refined_poly2_model.yaml',
        )
        self.declare_parameter('work_z_mm', -170.0)
        self.declare_parameter('use_depth', False)
        self.declare_parameter('depth_aligned_to_color', True)
        self.declare_parameter('depth_roi_px', 9)
        self.declare_parameter('depth_sample_mode', 'bbox_near')
        self.declare_parameter('depth_bbox_shrink', 0.25)
        self.declare_parameter('depth_u_offset_px', 0.0)
        self.declare_parameter('depth_v_offset_px', 0.0)
        self.declare_parameter('depth_percentile', 20.0)
        self.declare_parameter('depth_temporal_window', 5)
        self.declare_parameter('depth_temporal_max_pixel_jump', 20.0)
        self.declare_parameter('depth_min_m', 0.12)
        self.declare_parameter('depth_max_m', 1.20)
        self.declare_parameter('use_depth_for_z', False)
        self.declare_parameter('use_depth_for_layer', False)
        self.declare_parameter('depth_z_mapping', 'transform')
        self.declare_parameter('depth_near_m', 0.25)
        self.declare_parameter('depth_near_z_mm', -170.0)
        self.declare_parameter('depth_far_m', 0.35)
        self.declare_parameter('depth_far_z_mm', -230.0)
        self.declare_parameter('depth_z_offset_mm', 0.0)
        self.declare_parameter('approach_z_mm', -155.0)
        self.declare_parameter('move_to_work_z', False)
        self.declare_parameter('use_staged_motion', True)
        self.declare_parameter('home_clear_x_mm', 0.0)
        self.declare_parameter('home_clear_y_mm', 0.0)
        self.declare_parameter('first_drop_z_mm', -170.0)
        self.declare_parameter('xy_travel_z_mm', -170.0)
        self.declare_parameter('feedrate', 80.0)
        self.declare_parameter('offset_x_mm', 10.0)
        self.declare_parameter('offset_y_mm', -5.0)
        self.declare_parameter('offset_z_mm', 3.0)
        self.declare_parameter('enforce_workspace_limits', True)
        self.declare_parameter('x_min_mm', -100.0)
        self.declare_parameter('x_max_mm', 100.0)
        self.declare_parameter('y_min_mm', -100.0)
        self.declare_parameter('y_max_mm', 100.0)
        self.declare_parameter('z_min_mm', -220.0)
        self.declare_parameter('z_max_mm', -140.0)
        self.declare_parameter('min_area_px', 250.0)
        self.declare_parameter('max_area_px', 200000.0)
        self.declare_parameter('blur_kernel', 5)
        self.declare_parameter('show_window', True)
        self.declare_parameter('window_name', 'delta visual pick demo')
        self.declare_parameter('detector', 'color')
        self.declare_parameter(
            'yolo_model_path',
            '/home/whr/文档/xwechat_files/wxid_mc7cj27h4kzg22_bc6c/msg/file/2026-05/best.pt',
        )
        self.declare_parameter('yolo_conf', 0.35)
        self.declare_parameter('yolo_imgsz', 640)
        self.declare_parameter('yolo_class_id', -1)
        self.declare_parameter('yolo_class_name', '')

        # Yellow defaults in OpenCV HSV. Tune with params if lighting shifts.
        self.declare_parameter('h_min', 18)
        self.declare_parameter('h_max', 42)
        self.declare_parameter('s_min', 70)
        self.declare_parameter('v_min', 80)

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.depth_camera_info_topic = str(self.get_parameter('depth_camera_info_topic').value)
        self.delta_move_topic = str(self.get_parameter('delta_move_topic').value)
        self.trigger_topic = str(self.get_parameter('trigger_topic').value)
        self.calibration_path = str(self.get_parameter('calibration_path').value)
        self.use_layered_calibration = bool(self.get_parameter('use_layered_calibration').value)
        self.layered_calibration_dir = str(self.get_parameter('layered_calibration_dir').value)
        self.layered_calibration_prefix = str(self.get_parameter('layered_calibration_prefix').value)
        self.layered_calibration_zs_mm = parse_float_list(self.get_parameter('layered_calibration_zs_mm').value)
        self.calibration_select_z_mm = float(self.get_parameter('calibration_select_z_mm').value)
        self.use_empirical_model = bool(self.get_parameter('use_empirical_model').value)
        self.empirical_model_path = str(self.get_parameter('empirical_model_path').value)
        self.work_z_mm = float(self.get_parameter('work_z_mm').value)
        self.use_depth = bool(self.get_parameter('use_depth').value)
        self.depth_aligned_to_color = bool(self.get_parameter('depth_aligned_to_color').value)
        self.depth_roi_px = int(self.get_parameter('depth_roi_px').value)
        self.depth_sample_mode = str(self.get_parameter('depth_sample_mode').value).lower().strip()
        self.depth_bbox_shrink = float(self.get_parameter('depth_bbox_shrink').value)
        self.depth_u_offset_px = float(self.get_parameter('depth_u_offset_px').value)
        self.depth_v_offset_px = float(self.get_parameter('depth_v_offset_px').value)
        self.depth_percentile = float(self.get_parameter('depth_percentile').value)
        self.depth_temporal_window = int(self.get_parameter('depth_temporal_window').value)
        self.depth_temporal_max_pixel_jump = float(self.get_parameter('depth_temporal_max_pixel_jump').value)
        self.depth_min_m = float(self.get_parameter('depth_min_m').value)
        self.depth_max_m = float(self.get_parameter('depth_max_m').value)
        self.use_depth_for_z = bool(self.get_parameter('use_depth_for_z').value)
        self.use_depth_for_layer = bool(self.get_parameter('use_depth_for_layer').value)
        self.depth_z_mapping = str(self.get_parameter('depth_z_mapping').value).lower().strip()
        self.depth_near_m = float(self.get_parameter('depth_near_m').value)
        self.depth_near_z_mm = float(self.get_parameter('depth_near_z_mm').value)
        self.depth_far_m = float(self.get_parameter('depth_far_m').value)
        self.depth_far_z_mm = float(self.get_parameter('depth_far_z_mm').value)
        self.depth_z_offset_mm = float(self.get_parameter('depth_z_offset_mm').value)
        self.approach_z_mm = float(self.get_parameter('approach_z_mm').value)
        self.move_to_work_z = bool(self.get_parameter('move_to_work_z').value)
        self.use_staged_motion = bool(self.get_parameter('use_staged_motion').value)
        self.home_clear_x_mm = float(self.get_parameter('home_clear_x_mm').value)
        self.home_clear_y_mm = float(self.get_parameter('home_clear_y_mm').value)
        self.first_drop_z_mm = float(self.get_parameter('first_drop_z_mm').value)
        self.xy_travel_z_mm = float(self.get_parameter('xy_travel_z_mm').value)
        self.feedrate = float(self.get_parameter('feedrate').value)
        self.offset_mm = np.array(
            [
                float(self.get_parameter('offset_x_mm').value),
                float(self.get_parameter('offset_y_mm').value),
                float(self.get_parameter('offset_z_mm').value),
            ],
            dtype=float,
        )
        self.enforce_workspace_limits = bool(self.get_parameter('enforce_workspace_limits').value)
        self.workspace_limits = {
            'x': (
                float(self.get_parameter('x_min_mm').value),
                float(self.get_parameter('x_max_mm').value),
            ),
            'y': (
                float(self.get_parameter('y_min_mm').value),
                float(self.get_parameter('y_max_mm').value),
            ),
            'z': (
                float(self.get_parameter('z_min_mm').value),
                float(self.get_parameter('z_max_mm').value),
            ),
        }
        self.min_area_px = float(self.get_parameter('min_area_px').value)
        self.max_area_px = float(self.get_parameter('max_area_px').value)
        self.blur_kernel = int(self.get_parameter('blur_kernel').value)
        self.show_window = bool(self.get_parameter('show_window').value)
        self.window_name = str(self.get_parameter('window_name').value)
        self.detector = str(self.get_parameter('detector').value).lower().strip()
        self.yolo_model_path = str(self.get_parameter('yolo_model_path').value)
        self.yolo_conf = float(self.get_parameter('yolo_conf').value)
        self.yolo_imgsz = int(self.get_parameter('yolo_imgsz').value)
        self.yolo_class_id = int(self.get_parameter('yolo_class_id').value)
        self.yolo_class_name = str(self.get_parameter('yolo_class_name').value).strip()

        self.camera_matrix = None
        self.depth_camera_matrix = None
        self.latest_depth = None
        self.latest_target = None
        self.latest_frame = None
        self.bridge = CvBridge()
        self.yolo_model = None
        self.depth_history = deque(maxlen=max(1, self.depth_temporal_window))
        self.empirical_model = None

        self.layered_transforms = {}
        if self.use_layered_calibration:
            self.load_layered_calibrations()
        if self.use_empirical_model:
            self.empirical_model = load_empirical_model(self.empirical_model_path)

        self.t_delta_camera = load_transform(self.calibration_path)
        self.r_delta_camera = self.t_delta_camera[:3, :3]
        self.t_delta_camera_vec = self.t_delta_camera[:3, 3]

        if self.detector == 'yolo':
            self.load_yolo_model()

        self.move_pub = self.create_publisher(Point, self.delta_move_topic, 10)
        self.raw_gcode_pub = self.create_publisher(String, '/delta_arm/gcode_raw', 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.on_camera_info, 10)
        self.create_subscription(CameraInfo, self.depth_camera_info_topic, self.on_depth_camera_info, 10)
        self.create_subscription(Image, self.image_topic, self.on_image, 10)
        self.create_subscription(Image, self.depth_topic, self.on_depth_image, 10)
        self.create_subscription(Empty, self.trigger_topic, self.on_trigger, 10)
        self.create_timer(0.03, self.on_timer)
        self.create_timer(1.0, self.on_status_timer)

        self.get_logger().info('Delta visual pick demo ready')
        self.get_logger().info(f'image: {self.image_topic}')
        self.get_logger().info(f'depth: {self.depth_topic}, use_depth={self.use_depth}, aligned_to_color={self.depth_aligned_to_color}')
        self.get_logger().info(f'trigger: {self.trigger_topic}')
        self.get_logger().info(f'detector: {self.detector}')
        self.get_logger().info(f'calibration: {self.calibration_path}')
        if self.empirical_model:
            self.get_logger().info(
                'empirical model: %s, degree=%d, train=%s, loocv=%s'
                % (
                    self.empirical_model_path,
                    self.empirical_model['degree'],
                    self.empirical_model.get('train_error', {}),
                    self.empirical_model.get('loocv_error', {}),
                )
            )
        if self.use_layered_calibration:
            self.get_logger().info(
                'layered calibration: enabled, layers=%s, select_z=%.1f mm'
                % (sorted(self.layered_transforms.keys()), self.calibration_z_for_target())
            )
        self.get_logger().info(
                'fixed plane/depth fallback: work_z=%.1f mm, approach_z=%.1f mm, offset=%s mm, use_depth_for_z=%s, use_depth_for_layer=%s, depth_z_offset=%.1f mm'
            % (
                self.work_z_mm,
                self.approach_z_mm,
                np.round(self.offset_mm, 2).tolist(),
                self.use_depth_for_z,
                self.use_depth_for_layer,
                self.depth_z_offset_mm,
            )
        )
        self.get_logger().info(
            'depth sampling: mode=%s, roi=%d px, bbox_shrink=%.2f, offset=(%.1f, %.1f) depth_px, percentile=%.1f, temporal_window=%d'
            % (
                self.depth_sample_mode,
                self.depth_roi_px,
                self.depth_bbox_shrink,
                self.depth_u_offset_px,
                self.depth_v_offset_px,
                self.depth_percentile,
                self.depth_temporal_window,
            )
        )
        self.get_logger().info(
            'depth z mapping: %s, near %.3fm -> %.1fmm, far %.3fm -> %.1fmm'
            % (
                self.depth_z_mapping,
                self.depth_near_m,
                self.depth_near_z_mm,
                self.depth_far_m,
                self.depth_far_z_mm,
            )
        )
        self.get_logger().info(
            'motion: staged=%s, first_drop=(%.1f, %.1f, %.1f), xy_travel_z=%.1f, feedrate=%.1f'
            % (
                self.use_staged_motion,
                self.home_clear_x_mm,
                self.home_clear_y_mm,
                self.first_drop_z_mm,
                self.xy_travel_z_mm,
                self.feedrate,
            )
        )
        self.get_logger().info(
            'workspace limits: x=%s y=%s z=%s, enforce=%s'
            % (
                self.workspace_limits['x'],
                self.workspace_limits['y'],
                self.workspace_limits['z'],
                self.enforce_workspace_limits,
            )
        )
        self.get_logger().info('keys: SPACE/m/ENTER send move, w write debug image, q quit')

    def load_layered_calibrations(self):
        if not self.layered_calibration_zs_mm:
            raise RuntimeError('use_layered_calibration is true, but layered_calibration_zs_mm is empty')

        loaded = {}
        for z_mm in self.layered_calibration_zs_mm:
            z_key = int(round(abs(z_mm)))
            path = f'{self.layered_calibration_dir.rstrip("/")}/{self.layered_calibration_prefix}{z_key}.yaml'
            transform = load_transform(path)
            loaded[float(z_mm)] = transform

        self.layered_transforms = loaded

    def calibration_z_for_target(self):
        if math.isfinite(self.calibration_select_z_mm):
            return self.calibration_select_z_mm
        return self.work_z_mm

    def transform_for_target(self):
        if not self.use_layered_calibration or not self.layered_transforms:
            return self.r_delta_camera, self.t_delta_camera_vec, None

        select_z = self.calibration_z_for_target()
        layer_z = min(self.layered_transforms.keys(), key=lambda z: abs(z - select_z))
        transform = self.layered_transforms[layer_z]
        return transform[:3, :3], transform[:3, 3], layer_z

    def raw_depth_to_delta_z_mm(self, depth_m):
        near_m = self.depth_near_m
        far_m = self.depth_far_m
        if abs(far_m - near_m) < 1e-9:
            return self.depth_near_z_mm + self.depth_z_offset_mm

        ratio = (float(depth_m) - near_m) / (far_m - near_m)
        ratio = clamp(ratio, 0.0, 1.0)
        z_mm = self.depth_near_z_mm + ratio * (self.depth_far_z_mm - self.depth_near_z_mm)
        return z_mm + self.depth_z_offset_mm

    def predict_empirical_delta_mm(self, camera_xyz_m):
        if self.empirical_model is None:
            return None
        camera_xyz_mm = np.array(camera_xyz_m, dtype=float).reshape(1, 3) * 1000.0
        normalized = (camera_xyz_mm - self.empirical_model['camera_mean_mm']) / self.empirical_model['camera_scale_mm']
        phi = polynomial_terms(normalized, self.empirical_model['degree'])
        return (phi @ self.empirical_model['coefficients'])[0]

    def nearest_layer_for_z(self, z_mm):
        if not self.layered_transforms:
            return None
        return min(self.layered_transforms.keys(), key=lambda layer_z: abs(layer_z - float(z_mm)))

    def transform_for_depth_point(self, camera_xyz_m, depth_m=None):
        if not self.use_layered_calibration or not self.layered_transforms or not self.use_depth_for_layer:
            return self.transform_for_target()

        if self.depth_z_mapping == 'raw_linear' and depth_m is not None:
            target_z = self.raw_depth_to_delta_z_mm(depth_m)
            layer_z = min(self.layered_transforms.keys(), key=lambda z: abs(z - target_z))
            transform = self.layered_transforms[layer_z]
            return transform[:3, :3], transform[:3, 3], layer_z

        best = None
        for layer_z, transform in self.layered_transforms.items():
            delta_xyz_m = transform[:3, :3] @ camera_xyz_m + transform[:3, 3]
            delta_z_mm = delta_xyz_m[2] * 1000.0 + self.offset_mm[2] + self.depth_z_offset_mm
            error = abs(delta_z_mm - layer_z)
            if best is None or error < best[0]:
                best = (error, layer_z, transform)

        _, layer_z, transform = best
        return transform[:3, :3], transform[:3, 3], layer_z

    def on_trigger(self, _msg):
        self.send_latest_move()

    def load_yolo_model(self):
        if YOLO is None:
            raise RuntimeError('ultralytics is not installed; cannot use detector:=yolo')
        self.yolo_model = YOLO(self.yolo_model_path)
        names = getattr(self.yolo_model, 'names', {})
        self.get_logger().info(f'yolo model: {self.yolo_model_path}')
        self.get_logger().info(f'yolo classes: {names}')

    def on_camera_info(self, msg):
        self.camera_matrix = np.array(msg.k, dtype=float).reshape(3, 3)

    def on_depth_camera_info(self, msg):
        self.depth_camera_matrix = np.array(msg.k, dtype=float).reshape(3, 3)

    def on_depth_image(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().warning(f'depth conversion failed: {exc}')

    def yellow_mask(self, bgr):
        kernel = self.blur_kernel if self.blur_kernel % 2 == 1 else self.blur_kernel + 1
        kernel = int(clamp(kernel, 1, 31))
        if kernel > 1:
            bgr = cv2.GaussianBlur(bgr, (kernel, kernel), 0)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h_min = int(self.get_parameter('h_min').value)
        h_max = int(self.get_parameter('h_max').value)
        s_min = int(self.get_parameter('s_min').value)
        v_min = int(self.get_parameter('v_min').value)
        lower = np.array([clamp(h_min, 0, 179), clamp(s_min, 0, 255), clamp(v_min, 0, 255)])
        upper = np.array([clamp(h_max, 0, 179), 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        morph = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph)
        return mask

    def pixel_to_delta_on_plane(self, u, v):
        if self.camera_matrix is None:
            return None

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        if fx == 0.0 or fy == 0.0:
            return None

        ray_camera = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=float)
        r_delta_camera, t_delta_camera_vec, layer_z = self.transform_for_target()
        denominator = float(r_delta_camera[2, :] @ ray_camera)
        if abs(denominator) < 1e-6:
            return None

        work_z_m = self.work_z_mm / 1000.0
        scale = (work_z_m - t_delta_camera_vec[2]) / denominator
        if not math.isfinite(scale) or scale <= 0.0:
            return None

        camera_xyz_m = ray_camera * scale
        delta_xyz_m = r_delta_camera @ camera_xyz_m + t_delta_camera_vec
        delta_xyz_mm = delta_xyz_m * 1000.0 + self.offset_mm
        return camera_xyz_m, delta_xyz_mm, layer_z

    def depth_box_from_target(self, u, v, color_shape, bbox=None):
        depth_h, depth_w = self.latest_depth.shape[:2]
        color_h, color_w = color_shape[:2]

        du = int(round(u * depth_w / color_w + self.depth_u_offset_px))
        dv = int(round(v * depth_h / color_h + self.depth_v_offset_px))
        du = clamp(du, 0, depth_w - 1)
        dv = clamp(dv, 0, depth_h - 1)

        mode = self.depth_sample_mode
        if bbox is not None and mode.startswith('bbox'):
            x, y, w, h = bbox
            shrink = clamp(self.depth_bbox_shrink, 0.0, 0.8)
            x0_color = x + w * shrink * 0.5
            x1_color = x + w * (1.0 - shrink * 0.5)
            y0_color = y + h * shrink * 0.5
            y1_color = y + h * (1.0 - shrink * 0.5)

            x0 = int(math.floor(x0_color * depth_w / color_w + self.depth_u_offset_px))
            x1 = int(math.ceil(x1_color * depth_w / color_w + self.depth_u_offset_px))
            y0 = int(math.floor(y0_color * depth_h / color_h + self.depth_v_offset_px))
            y1 = int(math.ceil(y1_color * depth_h / color_h + self.depth_v_offset_px))
            x0 = clamp(x0, 0, depth_w - 1)
            x1 = clamp(x1, x0 + 1, depth_w)
            y0 = clamp(y0, 0, depth_h - 1)
            y1 = clamp(y1, y0 + 1, depth_h)
            return int(du), int(dv), int(x0), int(x1), int(y0), int(y1)

        roi = max(1, int(self.depth_roi_px))
        if roi % 2 == 0:
            roi += 1
        half = roi // 2
        x0 = max(0, du - half)
        x1 = min(depth_w, du + half + 1)
        y0 = max(0, dv - half)
        y1 = min(depth_h, dv + half + 1)
        return int(du), int(dv), int(x0), int(x1), int(y0), int(y1)

    def choose_depth_value(self, valid):
        mode = self.depth_sample_mode
        if mode in ('mean', 'bbox_mean'):
            return float(np.mean(valid))
        if mode in ('bbox_near_mean', 'near_mean'):
            percentile = clamp(self.depth_percentile, 0.0, 100.0)
            threshold = float(np.percentile(valid, percentile))
            near = valid[valid <= threshold]
            if near.size >= 3:
                return float(np.mean(near))
            return threshold
        if mode in ('bbox_near', 'near', 'percentile'):
            percentile = clamp(self.depth_percentile, 0.0, 100.0)
            return float(np.percentile(valid, percentile))
        return float(np.median(valid))

    def smooth_depth_value(self, du, dv, z):
        if self.depth_temporal_window <= 1:
            self.depth_history.clear()
            return z, z, 1

        if self.depth_history:
            last_u, last_v, _last_z = self.depth_history[-1]
            jump = math.hypot(float(du) - last_u, float(dv) - last_v)
            if jump > self.depth_temporal_max_pixel_jump:
                self.depth_history.clear()

        self.depth_history.append((float(du), float(dv), float(z)))
        values = np.array([item[2] for item in self.depth_history], dtype=float)
        return float(np.mean(values)), z, int(values.size)

    def pixel_to_delta_with_depth(self, u, v, color_shape, bbox=None):
        if self.latest_depth is None:
            return None

        depth = self.latest_depth
        if depth.ndim != 2:
            return None

        depth_h, depth_w = depth.shape[:2]
        color_h, color_w = color_shape[:2]
        if depth_w <= 0 or depth_h <= 0 or color_w <= 0 or color_h <= 0:
            return None

        k = self.depth_camera_matrix if self.depth_camera_matrix is not None else self.camera_matrix
        if not self.depth_aligned_to_color:
            k = self.depth_camera_matrix

        if k is None:
            return None

        du, dv, x0, x1, y0, y1 = self.depth_box_from_target(u, v, color_shape, bbox)
        patch = depth[y0:y1, x0:x1].astype(np.float32)
        if patch.size == 0:
            return None

        if depth.dtype == np.uint16 or np.nanmax(patch) > 20.0:
            patch_m = patch / 1000.0
        else:
            patch_m = patch

        valid = patch_m[np.isfinite(patch_m)]
        valid = valid[(valid >= self.depth_min_m) & (valid <= self.depth_max_m)]
        min_valid = max(3, min(int(self.depth_roi_px), 30))
        if valid.size < min_valid:
            return None

        raw_z = self.choose_depth_value(valid)
        z, raw_z, temporal_count = self.smooth_depth_value(du, dv, raw_z)
        fx = k[0, 0]
        fy = k[1, 1]
        cx = k[0, 2]
        cy = k[1, 2]
        if fx == 0.0 or fy == 0.0:
            return None

        camera_xyz_m = np.array(
            [
                (du - cx) * z / fx,
                (dv - cy) * z / fy,
                z,
            ],
            dtype=float,
        )
        empirical_delta_mm = self.predict_empirical_delta_mm(camera_xyz_m)
        if empirical_delta_mm is not None:
            delta_xyz_mm = empirical_delta_mm + self.offset_mm
            layer_z = self.nearest_layer_for_z(delta_xyz_mm[2])
            stats = {
                'count': int(valid.size),
                'temporal_count': int(temporal_count),
                'raw_m': float(raw_z),
                'smooth_m': float(z),
                'min_m': float(np.min(valid)),
                'median_m': float(np.median(valid)),
                'mean_m': float(np.mean(valid)),
                'max_m': float(np.max(valid)),
                'box': (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
                'depth_pixel': (int(du), int(dv)),
                'sample_mode': self.depth_sample_mode,
                'mapping': 'empirical_model',
            }
            return camera_xyz_m, delta_xyz_mm, z, stats, layer_z

        r_delta_camera, t_delta_camera_vec, layer_z = self.transform_for_depth_point(camera_xyz_m, z)
        delta_xyz_m = r_delta_camera @ camera_xyz_m + t_delta_camera_vec
        delta_xyz_mm = delta_xyz_m * 1000.0 + self.offset_mm
        stats = {
            'count': int(valid.size),
            'temporal_count': int(temporal_count),
            'raw_m': float(raw_z),
            'smooth_m': float(z),
            'min_m': float(np.min(valid)),
            'median_m': float(np.median(valid)),
            'mean_m': float(np.mean(valid)),
            'max_m': float(np.max(valid)),
            'box': (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
            'depth_pixel': (int(du), int(dv)),
            'sample_mode': self.depth_sample_mode,
        }
        return camera_xyz_m, delta_xyz_mm, z, stats, layer_z

    def detect_target(self, bgr):
        if self.detector == 'yolo':
            return self.detect_yolo_target(bgr)

        mask = self.yellow_mask(bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area_px or area > self.max_area_px:
                continue
            moments = cv2.moments(contour)
            if abs(moments['m00']) < 1e-9:
                continue
            u = float(moments['m10'] / moments['m00'])
            v = float(moments['m01'] / moments['m00'])
            x, y, w, h = cv2.boundingRect(contour)
            result = self.pixel_to_delta_for_target(u, v, bgr.shape, (x, y, w, h))
            if result is None:
                continue
            camera_xyz_m, delta_xyz_mm, depth_info = result
            return {
                'u': u,
                'v': v,
                'area': area,
                'bbox': (x, y, w, h),
                'camera_xyz_m': camera_xyz_m,
                'delta_xyz_mm': delta_xyz_mm,
                'depth_info': depth_info,
            }, mask
        return None, mask

    def pixel_to_delta_for_target(self, u, v, image_shape, bbox=None):
        if self.use_depth:
            depth_result = self.pixel_to_delta_with_depth(u, v, image_shape, bbox)
            if depth_result is not None:
                camera_xyz_m, delta_xyz_mm, depth_m, depth_stats, layer_z = depth_result
                return camera_xyz_m, delta_xyz_mm, {
                    'source': 'depth',
                    'depth_m': depth_m,
                    'count': depth_stats['count'],
                    'stats': depth_stats,
                    'layer_z_mm': layer_z,
                }
        plane_result = self.pixel_to_delta_on_plane(u, v)
        if plane_result is None:
            return None
        camera_xyz_m, delta_xyz_mm, layer_z = plane_result
        return camera_xyz_m, delta_xyz_mm, {
            'source': 'plane',
            'depth_m': None,
            'count': 0,
            'layer_z_mm': layer_z,
        }

    def yolo_class_allowed(self, cls_id):
        if self.yolo_class_id >= 0 and cls_id != self.yolo_class_id:
            return False
        if self.yolo_class_name:
            names = getattr(self.yolo_model, 'names', {})
            if str(names.get(cls_id, '')).lower() != self.yolo_class_name.lower():
                return False
        return True

    def detect_yolo_target(self, bgr):
        if self.yolo_model is None:
            return None, np.zeros(bgr.shape[:2], dtype=np.uint8)

        results = self.yolo_model.predict(
            source=bgr,
            conf=self.yolo_conf,
            imgsz=self.yolo_imgsz,
            verbose=False,
        )
        if not results:
            return None, np.zeros(bgr.shape[:2], dtype=np.uint8)

        best = None
        for box in results[0].boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
            conf = float(box.conf[0].detach().cpu().numpy())
            cls_id = int(box.cls[0].detach().cpu().numpy())
            if not self.yolo_class_allowed(cls_id):
                continue

            x1, y1, x2, y2 = xyxy
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area < self.min_area_px or area > self.max_area_px:
                continue
            if best is None or conf > best['conf']:
                best = {
                    'xyxy': xyxy,
                    'area': area,
                    'conf': conf,
                    'cls_id': cls_id,
                }

        mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        if best is None:
            return None, mask

        x1, y1, x2, y2 = best['xyxy']
        u = float((x1 + x2) * 0.5)
        v = float((y1 + y2) * 0.5)
        bbox = (
            int(round(x1)),
            int(round(y1)),
            int(round(x2 - x1)),
            int(round(y2 - y1)),
        )
        result = self.pixel_to_delta_for_target(u, v, bgr.shape, bbox)
        if result is None:
            return None, mask
        camera_xyz_m, delta_xyz_mm, depth_info = result
        names = getattr(self.yolo_model, 'names', {})
        label = str(names.get(best['cls_id'], best['cls_id']))
        return {
            'u': u,
            'v': v,
            'area': best['area'],
            'bbox': bbox,
            'camera_xyz_m': camera_xyz_m,
            'delta_xyz_mm': delta_xyz_mm,
            'label': label,
            'confidence': best['conf'],
            'depth_info': depth_info,
        }, mask

    def on_image(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warning(f'image conversion failed: {exc}')
            return

        target, mask = self.detect_target(frame)
        self.latest_target = target
        self.latest_frame = self.draw_overlay(frame, target, mask)

    def draw_overlay(self, frame, target, mask):
        canvas = frame.copy()
        if target is None:
            target_name = 'YOLO TARGET' if self.detector == 'yolo' else 'YELLOW TARGET'
            cv2.putText(
                canvas,
                f'NO {target_name}',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            u = int(round(target['u']))
            v = int(round(target['v']))
            x, y, w, h = target['bbox']
            delta = target['delta_xyz_mm']
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.drawMarker(canvas, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            depth_info = target.get('depth_info') or {}
            stats = depth_info.get('stats') or {}
            box = stats.get('box')
            if depth_info.get('source') == 'depth' and box and self.latest_depth is not None:
                depth_h, depth_w = self.latest_depth.shape[:2]
                color_h, color_w = frame.shape[:2]
                dx, dy, dw, dh = box
                cx0 = int(round(dx * color_w / max(depth_w, 1)))
                cy0 = int(round(dy * color_h / max(depth_h, 1)))
                cx1 = int(round((dx + dw) * color_w / max(depth_w, 1)))
                cy1 = int(round((dy + dh) * color_h / max(depth_h, 1)))
                cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), (0, 255, 0), 2)
                depth_pixel = stats.get('depth_pixel')
                if depth_pixel:
                    du, dv = depth_pixel
                    cu = int(round(du * color_w / max(depth_w, 1)))
                    cv = int(round(dv * color_h / max(depth_h, 1)))
                    cv2.drawMarker(canvas, (cu, cv), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
            self.draw_text_panel(canvas, self.overlay_lines(target), (12, 12))
            if not self.point_in_workspace(delta[0], delta[1], self.command_z_mm(delta[2], target)):
                cv2.putText(
                    canvas,
                    'OUT OF WORKSPACE - move rejected',
                    (20, 215),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        small_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        small_mask = cv2.resize(small_mask, (canvas.shape[1] // 4, canvas.shape[0] // 4))
        y0 = canvas.shape[0] - small_mask.shape[0] - 10
        x0 = canvas.shape[1] - small_mask.shape[1] - 10
        canvas[y0:y0 + small_mask.shape[0], x0:x0 + small_mask.shape[1]] = small_mask
        return canvas

    def draw_text_panel(self, canvas, lines, origin):
        x, y = origin
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.58
        thickness = 1
        line_h = 23
        widths = []
        for line in lines:
            (text_w, _text_h), _baseline = cv2.getTextSize(line, font, scale, thickness)
            widths.append(text_w)
        panel_w = min(canvas.shape[1] - x - 8, max(widths, default=0) + 22)
        panel_h = line_h * len(lines) + 15

        overlay = canvas.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + panel_w, y + panel_h),
            (20, 20, 20),
            -1,
        )
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
        cv2.rectangle(canvas, (x, y), (x + panel_w, y + panel_h), (80, 80, 80), 1)

        for idx, line in enumerate(lines):
            color = (0, 220, 0) if idx == 0 else (255, 255, 255)
            cv2.putText(
                canvas,
                line,
                (x + 10, y + 24 + idx * line_h),
                font,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

    def overlay_lines(self, target):
        delta = target['delta_xyz_mm']
        command_z = self.command_z_mm(delta[2], target)
        depth_info = target.get('depth_info') or {}
        stats = depth_info.get('stats') or {}
        label = target.get('label', 'target')
        confidence = target.get('confidence', 0.0)
        layer_z = depth_info.get('layer_z_mm')
        source = depth_info.get('source', 'none')
        mapping = stats.get('mapping', self.depth_z_mapping)

        lines = [
            'READY  %s %.2f  pixel=(%.0f, %.0f) area=%.0f'
            % (label, confidence, target['u'], target['v'], target['area']),
            'cmd xyz=(%.1f, %.1f, %.1f) mm'
            % (delta[0], delta[1], command_z),
            'transform_z=%.1f mm  cmd_z=%.1f mm'
            % (delta[2], command_z),
            'source=%s  calibZ=%s  mode=%s  map=%s'
            % (
                source,
                'none' if layer_z is None else '%.0f' % layer_z,
                self.depth_sample_mode,
                mapping,
            ),
        ]

        if source == 'depth':
            lines.append(
                'depth smooth=%.3f m  raw=%.3f m  temporal=%d'
                % (
                    depth_info.get('depth_m', 0.0),
                    stats.get('raw_m', 0.0),
                    stats.get('temporal_count', 0),
                )
            )
            lines.append(
                'patch min/mean/med/max=%.3f/%.3f/%.3f/%.3f m  n=%d'
                % (
                    stats.get('min_m', 0.0),
                    stats.get('mean_m', 0.0),
                    stats.get('median_m', 0.0),
                    stats.get('max_m', 0.0),
                    stats.get('count', 0),
                )
            )
            lines.append(
                'depth px=%s  box=%s  offset_px=(%.1f, %.1f)'
                % (
                    stats.get('depth_pixel', 'none'),
                    stats.get('box', 'none'),
                    self.depth_u_offset_px,
                    self.depth_v_offset_px,
                )
            )
        else:
            lines.append('depth unavailable, using fixed plane/work_z')

        lines.append(
            'offset=(%.1f, %.1f, %.1f)  rawZ %.3fm->%.0f %.3fm->%.0f'
            % (
                self.offset_mm[0],
                self.offset_mm[1],
                self.offset_mm[2],
                self.depth_near_m,
                self.depth_near_z_mm + self.depth_z_offset_mm,
                self.depth_far_m,
                self.depth_far_z_mm + self.depth_z_offset_mm,
            )
        )
        return lines

    def target_summary_text(self, target):
        text = 'pixel=(%.0f, %.0f) area=%.0f' % (
            target['u'],
            target['v'],
            target['area'],
        )
        if 'label' in target:
            text += ' %s %.2f' % (target['label'], target.get('confidence', 0.0))
        depth_info = target.get('depth_info') or {}
        if depth_info.get('source') == 'depth':
            text += ' depth=%.3fm' % depth_info.get('depth_m', 0.0)
            stats = depth_info.get('stats') or {}
            if stats:
                text += ' raw=%.3f d=[%.3f,%.3f,%.3f] n=%d/%d' % (
                    stats.get('raw_m', 0.0),
                    stats.get('min_m', 0.0),
                    stats.get('median_m', 0.0),
                    stats.get('max_m', 0.0),
                    stats.get('count', 0),
                    stats.get('temporal_count', 0),
                )
        elif depth_info.get('source') == 'plane':
            text += ' planeZ'
        if depth_info.get('layer_z_mm') is not None:
            text += ' calibZ=%.0f' % depth_info.get('layer_z_mm')
        return text

    def command_z_mm(self, transformed_z_mm, target=None):
        depth_info = (target or {}).get('depth_info') or {}
        if self.use_depth_for_z and depth_info.get('source') == 'depth':
            stats = depth_info.get('stats') or {}
            if stats.get('mapping') == 'empirical_model':
                return float(transformed_z_mm) + self.depth_z_offset_mm
            if self.depth_z_mapping == 'raw_linear':
                return self.raw_depth_to_delta_z_mm(depth_info.get('depth_m', 0.0))
            return float(transformed_z_mm) + self.depth_z_offset_mm
        if self.move_to_work_z:
            return transformed_z_mm
        return self.approach_z_mm

    def send_latest_move(self):
        if self.latest_target is None:
            self.get_logger().warning('no target to move to')
            return
        delta = self.latest_target['delta_xyz_mm']
        command_z = self.command_z_mm(delta[2], self.latest_target)
        if not self.point_in_workspace(delta[0], delta[1], command_z):
            self.get_logger().error(
                'move rejected: target outside workspace x=%.1f y=%.1f z=%.1f mm, limits x=%s y=%s z=%s'
                % (
                    delta[0],
                    delta[1],
                    command_z,
                    self.workspace_limits['x'],
                    self.workspace_limits['y'],
                    self.workspace_limits['z'],
                )
            )
            return

        if self.use_staged_motion:
            self.send_staged_move(float(delta[0]), float(delta[1]), float(command_z))
            return

        msg = Point()
        msg.x = float(delta[0])
        msg.y = float(delta[1])
        msg.z = float(command_z)
        self.move_pub.publish(msg)
        self.get_logger().info(
            'sent demo move: x=%.1f y=%.1f z=%.1f mm'
            % (msg.x, msg.y, msg.z)
        )

    def send_staged_move(self, x_mm, y_mm, z_mm):
        staged_points = [
            (self.home_clear_x_mm, self.home_clear_y_mm, self.first_drop_z_mm),
            (x_mm, y_mm, self.xy_travel_z_mm),
            (x_mm, y_mm, z_mm),
        ]
        for px, py, pz in staged_points:
            if not self.point_in_workspace(px, py, pz):
                self.get_logger().error(
                    'staged move rejected: waypoint outside workspace x=%.1f y=%.1f z=%.1f mm'
                    % (px, py, pz)
                )
                return

        lines = ['G90']
        for px, py, pz in staged_points:
            lines.append('G1 X%.2f Y%.2f Z%.2f F%.2f' % (px, py, pz, self.feedrate))

        msg = String()
        msg.data = '\n'.join(lines)
        self.raw_gcode_pub.publish(msg)
        self.get_logger().info(
            'sent staged move: drop=(%.1f, %.1f, %.1f) -> xy=(%.1f, %.1f, %.1f) -> final=(%.1f, %.1f, %.1f)'
            % (
                self.home_clear_x_mm,
                self.home_clear_y_mm,
                self.first_drop_z_mm,
                x_mm,
                y_mm,
                self.xy_travel_z_mm,
                x_mm,
                y_mm,
                z_mm,
            )
        )

    def point_in_workspace(self, x_mm, y_mm, z_mm):
        if not self.enforce_workspace_limits:
            return True
        x_low, x_high = self.workspace_limits['x']
        y_low, y_high = self.workspace_limits['y']
        z_low, z_high = self.workspace_limits['z']
        return (
            x_low <= float(x_mm) <= x_high
            and y_low <= float(y_mm) <= y_high
            and z_low <= float(z_mm) <= z_high
        )

    def on_status_timer(self):
        if self.camera_matrix is None:
            self.get_logger().info('waiting for camera info...')
            return
        if self.latest_target is None:
            if self.detector == 'yolo':
                self.get_logger().info('NO TARGET: show a model-detectable object in the camera view')
            else:
                self.get_logger().info('NO TARGET: show a yellow object in the camera view')
            return
        delta = self.latest_target['delta_xyz_mm']
        self.get_logger().info(
            'READY target %s, delta=(%.1f, %.1f, %.1f) mm'
            % (
                self.target_summary_text(self.latest_target),
                delta[0],
                delta[1],
                self.command_z_mm(delta[2], self.latest_target),
            )
        )

    def get_key(self):
        if not sys.stdin.isatty():
            return None
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            if select.select([sys.stdin], [], [], 0.0)[0]:
                return sys.stdin.read(1)
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def on_timer(self):
        key = self.get_key()
        if key in (' ', 'm', '\n', '\r'):
            self.send_latest_move()
        elif key == 'w':
            if self.latest_frame is not None:
                path = '/tmp/delta_visual_pick_demo.png'
                cv2.imwrite(path, self.latest_frame)
                self.get_logger().info(f'wrote {path}')
        elif key == 'q' or key == '\x03':
            raise KeyboardInterrupt

        if self.show_window and self.latest_frame is not None:
            cv2.imshow(self.window_name, self.latest_frame)
            window_key = cv2.waitKey(1) & 0xFF
            if window_key == ord(' '):
                self.send_latest_move()
            elif window_key == ord('w'):
                path = '/tmp/delta_visual_pick_demo.png'
                cv2.imwrite(path, self.latest_frame)
                self.get_logger().info(f'wrote {path}')
            elif window_key == ord('q'):
                raise KeyboardInterrupt

    def close(self):
        if self.show_window:
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = DeltaVisualPickDemo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
