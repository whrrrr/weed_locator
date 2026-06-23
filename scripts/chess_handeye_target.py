#!/usr/bin/env python3
import json
import math
from pathlib import Path

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


BOUNDARY_POINTS_XY = [
    (-40.0, 70.0),
    (-50.0, 50.0),
    (-60.0, 20.0),
    (-70.0, 10.0),
    (-80.0, 0.0),
    (-90.0, -10.0),
    (-110.0, -40.0),
    (0.0, -60.0),
    (110.0, -40.0),
    (90.0, -10.0),
    (80.0, 0.0),
    (70.0, 10.0),
    (60.0, 20.0),
    (50.0, 50.0),
    (40.0, 70.0),
    (0.0, 120.0),
]


def sort_boundary(points):
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def shrink_polygon(points, margin):
    if margin <= 0.0:
        return list(points)
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    shrunk = []
    for x, y in points:
        dx = x - cx
        dy = y - cy
        dist = math.hypot(dx, dy)
        if dist <= margin:
            shrunk.append((x, y))
        else:
            scale = (dist - margin) / dist
            shrunk.append((cx + dx * scale, cy + dy * scale))
    return shrunk


def point_in_polygon(x, y, polygon):
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def fit_affine_uv(pixel_points, delta_xy_points):
    pixels = np.asarray(pixel_points, dtype=float)
    delta_xy = np.asarray(delta_xy_points, dtype=float)
    design = np.column_stack([pixels[:, 0], pixels[:, 1], np.ones(len(pixels), dtype=float)])
    coeffs, *_ = np.linalg.lstsq(design, delta_xy, rcond=None)
    prediction = design @ coeffs
    errors = np.linalg.norm(prediction - delta_xy, axis=1)
    return coeffs, prediction, errors


class ChessHandeyeTarget(Node):
    def __init__(self):
        super().__init__('chess_handeye_target')

        self.declare_parameter('handeye_path', '/home/wyy/gpt_dev_ws/calibration_targets/delta_hand_eye.yaml')
        self.declare_parameter('dual_model_source', 'depth')
        self.declare_parameter('selected_pixel_topic', '/chess/selected_pixel_center')
        self.declare_parameter('detections_json_topic', '/chess/detections_json')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('depth_camera_info_topic', '/camera/depth/camera_info')
        self.declare_parameter('command_topic', '/chess/handeye_command')
        self.declare_parameter('camera_point_topic', '/chess/camera_point')
        self.declare_parameter('delta_target_topic', '/chess/delta_target')
        self.declare_parameter('move_status_topic', '/chess/move_status')
        self.declare_parameter('move_topic', '/delta_arm/move_to')
        self.declare_parameter('raw_gcode_topic', '/delta_arm/gcode_raw')
        self.declare_parameter('depth_window_px', 5)
        # Registered depth can contain a small hole at a dark/reflective chess
        # center.  Only search within the chess piece's immediate neighborhood;
        # a large window can silently select the background and is unsafe for motion.
        self.declare_parameter('depth_search_window_px', 11)
        self.declare_parameter('min_depth_m', 0.05)
        self.declare_parameter('max_depth_m', 2.0)
        self.declare_parameter('target_z_override_mm', float('nan'))
        self.declare_parameter('target_x_offset_mm', 0.0)
        self.declare_parameter('target_y_offset_mm', 0.0)
        self.declare_parameter('target_z_offset_mm', 0.0)
        self.declare_parameter('publish_move_on_go', True)
        self.declare_parameter('staged_move_on_go', True)
        self.declare_parameter('safe_xy_z_mm', -210.0)
        self.declare_parameter('approach_feedrate', 80.0)
        self.declare_parameter('workspace_check_enabled', True)
        self.declare_parameter('min_x_mm', -90.0)
        self.declare_parameter('max_x_mm', 90.0)
        self.declare_parameter('min_y_mm', -60.0)
        self.declare_parameter('max_y_mm', 100.0)
        self.declare_parameter('min_z_mm', -320.0)
        self.declare_parameter('max_z_mm', 0.0)
        self.declare_parameter('use_polygon_workspace', True)
        self.declare_parameter('polygon_margin_mm', 8.0)
        self.declare_parameter('publish_live_target', True)
        self.declare_parameter('live_target_rate_hz', 15.0)
        self.declare_parameter('require_fresh_detection_for_go', True)
        self.declare_parameter('max_detection_age_sec', 0.35)
        self.declare_parameter('single_layer_pixel_mode', True)

        self.bridge = CvBridge()
        self.latest_pixel = None
        self.latest_selected_bbox = None
        self.latest_selected_is_held = False
        self.latest_selected_stamp_sec = 0.0
        self.latest_depth_msg = None
        self.camera_matrix = None
        self.empirical_affine_model = None
        self.planar_model = None
        self.pixel_planar_model = None
        self.model_name = 'unknown'
        self.sample_camera_min_m = None
        self.sample_camera_max_m = None
        self.dual_samples = []
        self.t_delta_camera = self.load_handeye_transform()
        self.last_delta_target_mm = None
        self.last_camera_xyz_m = None
        self.last_live_compute_ok = False
        self.last_live_status = ''
        self.safe_polygon = shrink_polygon(
            sort_boundary(BOUNDARY_POINTS_XY),
            float(self.get_parameter('polygon_margin_mm').value),
        )

        self.camera_point_pub = self.create_publisher(
            PointStamped,
            str(self.get_parameter('camera_point_topic').value),
            10,
        )
        self.delta_target_pub = self.create_publisher(
            PointStamped,
            str(self.get_parameter('delta_target_topic').value),
            10,
        )
        self.move_status_pub = self.create_publisher(
            String,
            str(self.get_parameter('move_status_topic').value),
            10,
        )
        self.move_pub = self.create_publisher(
            Point,
            str(self.get_parameter('move_topic').value),
            10,
        )
        self.raw_gcode_pub = self.create_publisher(
            String,
            str(self.get_parameter('raw_gcode_topic').value),
            10,
        )

        self.create_subscription(
            PointStamped,
            str(self.get_parameter('selected_pixel_topic').value),
            self.on_selected_pixel,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter('detections_json_topic').value),
            self.on_detections_json,
            10,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter('depth_topic').value),
            self.on_depth,
            10,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter('depth_camera_info_topic').value),
            self.on_camera_info,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter('command_topic').value),
            self.on_command,
            10,
        )

        if bool(self.get_parameter('publish_live_target').value):
            live_rate_hz = max(1.0, float(self.get_parameter('live_target_rate_hz').value))
            self.create_timer(1.0 / live_rate_hz, self.on_live_timer)

        self.get_logger().info('chess_handeye_target ready')
        self.get_logger().info('command: ros2 topic pub --once /chess/handeye_command std_msgs/msg/String "{data: capture}"')
        self.get_logger().info('command: ros2 topic pub --once /chess/handeye_command std_msgs/msg/String "{data: go}"')

    def load_handeye_transform(self):
        path = Path(str(self.get_parameter('handeye_path').value)).expanduser()
        if not path.exists():
            raise RuntimeError(f'hand-eye file not found: {path}')
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        if 'pnp_model_camera_xyz_to_delta_xy' in data and 'depth_model_camera_xyz_to_delta_xy' in data:
            source = str(self.get_parameter('dual_model_source').value).strip().lower()
            if source not in ('pnp', 'depth'):
                raise RuntimeError('dual_model_source must be "pnp" or "depth"')
            model_key = '%s_model_camera_xyz_to_delta_xy' % source
            source_model = data[model_key]
            self.dual_samples = list(data.get('samples') or [])
            self.empirical_affine_model = None
            self.planar_model = {
                'coeffs': source_model['coeffs'],
                'delta_z_m': float(data['delta_z_m']),
                'error': source_model.get('self_fit_error', {}),
            }
            self.pixel_planar_model = None
            self.model_name = 'dual-%s' % source
            self.load_dual_sample_coverage(data, source)
            self.get_logger().info('loaded dual hand-eye %s model: %s' % (source, path))
            return np.eye(4, dtype=float)
        self.empirical_affine_model = data.get('empirical_affine_camera_xyz_to_delta_xyz')
        self.planar_model = data.get('planar_affine_camera_xyz_to_delta_xy')
        self.pixel_planar_model = data.get('pixel_planar_uv_to_delta_xy')
        self.model_name = 'standard'
        self.load_sample_coverage(data)
        transform = np.array(data['T_delta_camera'], dtype=float)
        if transform.shape != (4, 4):
            raise RuntimeError(f'T_delta_camera in {path} is not 4x4')
        self.get_logger().info(f'loaded hand-eye transform: {path}')
        if self.empirical_affine_model:
            err = (self.empirical_affine_model.get('error') or {}).get('rmse_m')
            if err is not None:
                self.get_logger().info('using empirical affine XYZ model, rmse=%.2f mm' % (float(err) * 1000.0))
        elif self.planar_model:
            err = (self.planar_model.get('error') or {}).get('rmse_m')
            if err is not None:
                self.get_logger().info('using planar affine XY model, rmse=%.2f mm' % (float(err) * 1000.0))
        return transform

    def load_dual_sample_coverage(self, data, source):
        key = '%s_camera_xyz_mm' % source
        points = [item.get(key) for item in data.get('samples') or []]
        points = [point for point in points if point is not None and len(point) >= 3]
        if not points:
            self.sample_camera_min_m = None
            self.sample_camera_max_m = None
            return
        arr = np.asarray(points, dtype=float) / 1000.0
        self.sample_camera_min_m = arr.min(axis=0)
        self.sample_camera_max_m = arr.max(axis=0)
        self.get_logger().info(
            'dual-%s sample coverage mm: min=%s max=%s'
            % (source, np.round(self.sample_camera_min_m * 1000.0, 1).tolist(),
               np.round(self.sample_camera_max_m * 1000.0, 1).tolist())
        )

    def load_sample_coverage(self, data):
        points = []
        for sample in data.get('samples') or []:
            camera = (
                sample.get('camera_position_m')
                or sample.get('camera_xyz_m')
                or sample.get('camera_point_m')
                or sample.get('camera')
            )
            if camera is None:
                for key, value in sample.items():
                    if 'camera' in str(key) and isinstance(value, list) and len(value) >= 3:
                        camera = value
                        break
            if camera is not None and len(camera) >= 3:
                points.append([float(camera[0]), float(camera[1]), float(camera[2])])

        if not points:
            self.sample_camera_min_m = None
            self.sample_camera_max_m = None
            self.get_logger().warning('hand-eye file has no sample camera coverage data')
            return

        arr = np.array(points, dtype=float)
        self.sample_camera_min_m = arr.min(axis=0)
        self.sample_camera_max_m = arr.max(axis=0)
        self.get_logger().info(
            'sample camera coverage mm: min=%s max=%s'
            % (
                np.round(self.sample_camera_min_m * 1000.0, 1).tolist(),
                np.round(self.sample_camera_max_m * 1000.0, 1).tolist(),
            )
        )

    def on_selected_pixel(self, msg):
        self.latest_pixel = msg

    def on_detections_json(self, msg):
        """Keep the currently selected YOLO box for depth recovery inside the piece."""
        try:
            payload = json.loads(msg.data)
            selected = int(payload.get('selected_index', -1))
            detections = payload.get('detections', [])
            if selected < 0 or selected >= len(detections):
                self.latest_selected_bbox = None
                self.latest_selected_is_held = False
                return
            bbox = detections[selected].get('bbox_xyxy')
            if not isinstance(bbox, list) or len(bbox) != 4:
                self.latest_selected_bbox = None
                self.latest_selected_is_held = False
                return
            x1, y1, x2, y2 = [float(value) for value in bbox]
            if x2 <= x1 or y2 <= y1:
                self.latest_selected_bbox = None
                self.latest_selected_is_held = False
                return
            self.latest_selected_bbox = (x1, y1, x2, y2)
            self.latest_selected_is_held = bool(payload.get('selected_held_from_previous_frame', False))
            stamp = payload.get('stamp') or {}
            self.latest_selected_stamp_sec = float(stamp.get('sec', 0)) + float(stamp.get('nanosec', 0)) * 1e-9
        except (ValueError, TypeError, json.JSONDecodeError):
            self.latest_selected_bbox = None
            self.latest_selected_is_held = False

    def on_depth(self, msg):
        self.latest_depth_msg = msg

    def on_camera_info(self, msg):
        self.camera_matrix = np.array(msg.k, dtype=float).reshape(3, 3)
        self.try_build_pixel_planar_model()

    def try_build_pixel_planar_model(self):
        if self.pixel_planar_model is not None:
            return
        if not bool(self.get_parameter('single_layer_pixel_mode').value):
            return
        if self.camera_matrix is None or not self.dual_samples:
            return

        source = str(self.get_parameter('dual_model_source').value).strip().lower()
        sample_key = '%s_camera_xyz_mm' % source
        pixel_points = []
        delta_xy_points = []
        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])
        cx = float(self.camera_matrix[0, 2])
        cy = float(self.camera_matrix[1, 2])
        if fx == 0.0 or fy == 0.0:
            return

        for sample in self.dual_samples:
            camera_xyz_mm = sample.get(sample_key)
            delta_xyz_mm = sample.get('delta_xyz_mm')
            if camera_xyz_mm is None or delta_xyz_mm is None:
                continue
            x_m, y_m, z_m = [float(v) / 1000.0 for v in camera_xyz_mm[:3]]
            if abs(z_m) < 1e-9:
                continue
            u = fx * x_m / z_m + cx
            v = fy * y_m / z_m + cy
            pixel_points.append([u, v])
            delta_xy_points.append([float(delta_xyz_mm[0]) / 1000.0, float(delta_xyz_mm[1]) / 1000.0])

        if len(pixel_points) < 3:
            self.get_logger().warning('pixel planar model unavailable: need 3 projected samples, got %d' % len(pixel_points))
            return

        coeffs, prediction, errors = fit_affine_uv(pixel_points, delta_xy_points)
        self.pixel_planar_model = {
            'coeffs': coeffs.tolist(),
            'delta_z_m': float(self.planar_model.get('delta_z_m', -0.23)) if self.planar_model else -0.23,
            'error': {
                'count': int(len(errors)),
                'rmse_m': float(np.sqrt(np.mean(errors ** 2))),
                'mean_m': float(np.mean(errors)),
                'median_m': float(np.median(errors)),
                'max_m': float(np.max(errors)),
            },
        }
        self.get_logger().info(
            'built single-layer pixel model from %d samples, rmse=%.2f mm'
            % (len(errors), float(np.sqrt(np.mean(errors ** 2))) * 1000.0)
        )

    def on_command(self, msg):
        command = msg.data.strip().lower()
        if command in ('capture', 'calc', 'compute'):
            self.compute_target(publish_move=False, log_result=True)
        elif command in ('go', 'move'):
            publish_move = bool(self.get_parameter('publish_move_on_go').value)
            self.compute_target(publish_move=publish_move, log_result=True)
        elif command == 'reload':
            self.t_delta_camera = self.load_handeye_transform()
        else:
            self.get_logger().warning('unknown command "%s"; use capture, go, or reload' % command)

    def on_live_timer(self):
        ok = self.compute_target(publish_move=False, log_result=False)
        self.last_live_compute_ok = bool(ok)

    def depth_window_px(self):
        return max(1, int(self.get_parameter('depth_window_px').value))

    def depth_search_window_px(self):
        requested = int(self.get_parameter('depth_search_window_px').value)
        return max(self.depth_window_px(), min(11, requested))

    def depth_image_to_meters(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if image.ndim != 2:
            raise RuntimeError(f'depth image must be single-channel, got shape {image.shape}')
        if msg.encoding in ('16UC1', 'mono16'):
            return image.astype(np.float32) / 1000.0
        if msg.encoding == '32FC1':
            return image.astype(np.float32)
        # Some drivers leave encoding nonstandard while still publishing depth-like numeric arrays.
        if image.dtype == np.uint16:
            return image.astype(np.float32) / 1000.0
        return image.astype(np.float32)

    def depth_at_pixel(self, depth_m, u, v):
        z_m = self.depth_at_pixel_window(depth_m, u, v, self.depth_window_px())
        if z_m is not None:
            return z_m

        search_window = self.depth_search_window_px()
        if search_window <= self.depth_window_px():
            return None

        z_m = self.depth_at_pixel_window(depth_m, u, v, search_window, nearest_count=25)
        if z_m is not None:
            self.get_logger().warning(
                'depth center patch invalid; using only nearby depth in %dx%d window at u=%.1f v=%.1f: %.1fmm'
                % (search_window, search_window, u, v, z_m * 1000.0)
            )
            return z_m

        z_m = self.depth_at_selected_bbox(depth_m)
        if z_m is not None:
            self.get_logger().warning(
                'depth center patch invalid; using median valid depth inside selected YOLO box: %.1fmm'
                % (z_m * 1000.0)
            )
        return z_m

    def depth_at_selected_bbox(self, depth_m):
        if self.latest_selected_bbox is None:
            return None
        height, width = depth_m.shape[:2]
        x1, y1, x2, y2 = self.latest_selected_bbox

        # Ignore the outer 15% of the detection box so the lookup remains on the
        # chess piece rather than the board/background at the box boundary.
        margin_x = (x2 - x1) * 0.15
        margin_y = (y2 - y1) * 0.15
        u0 = max(0, int(math.ceil(x1 + margin_x)))
        u1 = min(width, int(math.floor(x2 - margin_x)) + 1)
        v0 = max(0, int(math.ceil(y1 + margin_y)))
        v1 = min(height, int(math.floor(y2 - margin_y)) + 1)
        if u1 <= u0 or v1 <= v0:
            return None
        patch = depth_m[v0:v1, u0:u1]
        min_depth = float(self.get_parameter('min_depth_m').value)
        max_depth = float(self.get_parameter('max_depth_m').value)
        valid = patch[np.isfinite(patch) & (patch >= min_depth) & (patch <= max_depth)]
        # A few isolated depth pixels are not sufficient evidence for a move.
        if valid.size < 12:
            return None
        return float(np.median(valid))

    def depth_at_pixel_window(self, depth_m, u, v, window_px, nearest_count=None):
        height, width = depth_m.shape[:2]
        radius = max(0, int(window_px) // 2)
        u0 = max(0, int(round(u)) - radius)
        u1 = min(width, int(round(u)) + radius + 1)
        v0 = max(0, int(round(v)) - radius)
        v1 = min(height, int(round(v)) + radius + 1)
        patch = depth_m[v0:v1, u0:u1]
        min_depth = float(self.get_parameter('min_depth_m').value)
        max_depth = float(self.get_parameter('max_depth_m').value)
        valid_mask = np.isfinite(patch) & (patch >= min_depth) & (patch <= max_depth)
        valid = patch[valid_mask]
        if valid.size == 0:
            return None
        if nearest_count is not None and valid.size > nearest_count:
            yy, xx = np.nonzero(valid_mask)
            uu = xx + u0
            vv = yy + v0
            dist2 = (uu.astype(float) - float(u)) ** 2 + (vv.astype(float) - float(v)) ** 2
            keep = np.argsort(dist2)[:nearest_count]
            valid = patch[valid_mask][keep]
        return float(np.median(valid))

    def camera_xyz_from_pixel(self, u, v, depth_m):
        if self.camera_matrix is None:
            raise RuntimeError('no camera info received yet')
        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])
        cx = float(self.camera_matrix[0, 2])
        cy = float(self.camera_matrix[1, 2])
        if fx == 0.0 or fy == 0.0:
            raise RuntimeError('invalid camera intrinsics fx/fy=0')
        x = (float(u) - cx) * depth_m / fx
        y = (float(v) - cy) * depth_m / fy
        z = depth_m
        return np.array([x, y, z], dtype=float)

    def compute_target(self, publish_move=False, log_result=True):
        if self.latest_pixel is None:
            if log_result:
                self.get_logger().warning('no selected chess pixel yet')
            return False
        if self.latest_depth_msg is None:
            if log_result:
                self.get_logger().warning('no depth image yet')
            return False

        u = float(self.latest_pixel.point.x)
        v = float(self.latest_pixel.point.y)
        if publish_move and bool(self.get_parameter('require_fresh_detection_for_go').value):
            if self.latest_selected_is_held:
                text = 'blocked: 当前目标是保持帧，不是新检测帧'
                if log_result:
                    self.get_logger().warning(text)
                self.publish_move_status(text)
                return False
            max_age = max(0.0, float(self.get_parameter('max_detection_age_sec').value))
            if max_age > 0.0 and self.latest_selected_stamp_sec > 0.0 and self.latest_depth_msg is not None:
                depth_stamp = float(self.latest_depth_msg.header.stamp.sec) + float(self.latest_depth_msg.header.stamp.nanosec) * 1e-9
                if depth_stamp - self.latest_selected_stamp_sec > max_age:
                    text = 'blocked: 当前目标检测过旧，已超过 %.2f 秒' % max_age
                    if log_result:
                        self.get_logger().warning(text)
                    self.publish_move_status(text)
                    return False
        use_pixel_mode = bool(self.get_parameter('single_layer_pixel_mode').value)
        self.try_build_pixel_planar_model()
        z_m = None
        if use_pixel_mode and self.pixel_planar_model is not None:
            coeffs = np.array(self.pixel_planar_model['coeffs'], dtype=float)
            design = np.array([u, v, 1.0], dtype=float)
            delta_xy_m = design @ coeffs
            delta_z_m = float(self.pixel_planar_model.get('delta_z_m', -0.23))
            delta_xyz_m = np.array([delta_xy_m[0], delta_xy_m[1], delta_z_m], dtype=float)
            camera_xyz_m = np.array([u, v, float('nan')], dtype=float)
            self.last_camera_xyz_m = None
        else:
            depth_m = self.depth_image_to_meters(self.latest_depth_msg)
            z_m = self.depth_at_pixel(depth_m, u, v)
            if z_m is None:
                if log_result:
                    self.get_logger().warning('no valid depth around selected pixel u=%.1f v=%.1f' % (u, v))
                return False

            camera_xyz_m = self.camera_xyz_from_pixel(u, v, z_m)
            self.last_camera_xyz_m = camera_xyz_m.copy()
            if self.empirical_affine_model:
                coeffs = np.array(self.empirical_affine_model['coeffs'], dtype=float)
                design = np.array([camera_xyz_m[0], camera_xyz_m[1], camera_xyz_m[2], 1.0], dtype=float)
                delta_xyz_m = design @ coeffs
            elif self.planar_model:
                coeffs = np.array(self.planar_model['coeffs'], dtype=float)
                design = np.array([camera_xyz_m[0], camera_xyz_m[1], camera_xyz_m[2], 1.0], dtype=float)
                delta_xy_m = design @ coeffs
                delta_z_m = float(self.planar_model.get('delta_z_m', -0.23))
                delta_xyz_m = np.array([delta_xy_m[0], delta_xy_m[1], delta_z_m], dtype=float)
            else:
                camera_h = np.array([camera_xyz_m[0], camera_xyz_m[1], camera_xyz_m[2], 1.0], dtype=float)
                delta_xyz_m = (self.t_delta_camera @ camera_h)[:3]
        delta_xyz_mm = delta_xyz_m * 1000.0

        z_override = float(self.get_parameter('target_z_override_mm').value)
        if math.isfinite(z_override):
            delta_xyz_mm[2] = z_override
        delta_xyz_mm[0] += float(self.get_parameter('target_x_offset_mm').value)
        delta_xyz_mm[1] += float(self.get_parameter('target_y_offset_mm').value)
        delta_xyz_mm[2] += float(self.get_parameter('target_z_offset_mm').value)
        self.last_delta_target_mm = delta_xyz_mm

        camera_msg = PointStamped()
        camera_msg.header = self.latest_depth_msg.header
        camera_msg.point.x = float(camera_xyz_m[0])
        camera_msg.point.y = float(camera_xyz_m[1])
        camera_msg.point.z = float(camera_xyz_m[2])
        self.camera_point_pub.publish(camera_msg)

        delta_msg = PointStamped()
        delta_msg.header = self.latest_depth_msg.header
        delta_msg.header.frame_id = 'delta_base'
        delta_msg.point.x = float(delta_xyz_mm[0])
        delta_msg.point.y = float(delta_xyz_mm[1])
        delta_msg.point.z = float(delta_xyz_mm[2])
        self.delta_target_pub.publish(delta_msg)

        coverage_status = self.calibration_coverage_status(camera_xyz_m)
        if not publish_move:
            self.publish_live_status(coverage_status)

        if log_result:
            if use_pixel_mode and self.pixel_planar_model is not None:
                self.get_logger().info(
                    'pixel-mode chess pixel=(%.1f, %.1f) -> delta_mm=%s'
                    % (
                        u,
                        v,
                        np.round(delta_xyz_mm, 1).tolist(),
                    )
                )
            else:
                self.get_logger().info(
                    'chess pixel=(%.1f, %.1f) depth=%.1fmm camera_mm=%s -> delta_mm=%s'
                    % (
                        u,
                        v,
                        z_m * 1000.0,
                        np.round(camera_xyz_m * 1000.0, 1).tolist(),
                        np.round(delta_xyz_mm, 1).tolist(),
                    )
                )

        if publish_move:
            ok, reason = self.validate_delta_target(delta_xyz_mm)
            if not ok:
                self.get_logger().error(
                    'BLOCKED unsafe chess move: %s; target=%s mm'
                    % (reason, np.round(delta_xyz_mm, 1).tolist())
                )
                self.publish_move_status('blocked: %s; target_mm=%s' % (reason, np.round(delta_xyz_mm, 1).tolist()))
                return False
            if bool(self.get_parameter('staged_move_on_go').value):
                self.publish_staged_move(delta_xyz_mm)
            else:
                move = Point()
                move.x = float(delta_xyz_mm[0])
                move.y = float(delta_xyz_mm[1])
                move.z = float(delta_xyz_mm[2])
                self.move_pub.publish(move)
                self.get_logger().warning('published /delta_arm/move_to: %s mm' % np.round(delta_xyz_mm, 1).tolist())
                self.publish_move_status('moved direct: target_mm=%s' % np.round(delta_xyz_mm, 1).tolist())
        return True

    def calibration_coverage_status(self, camera_xyz_m):
        if self.sample_camera_min_m is None or self.sample_camera_max_m is None:
            return '标定区域: 无样本范围数据'

        labels = ('X', 'Y', 'Z')
        outside = []
        for index, label in enumerate(labels):
            value = float(camera_xyz_m[index])
            low = float(self.sample_camera_min_m[index])
            high = float(self.sample_camera_max_m[index])
            if value < low or value > high:
                outside.append(
                    '%s=%.1fmm 不在 [%.1f, %.1f]mm'
                    % (label, value * 1000.0, low * 1000.0, high * 1000.0)
                )

        if outside:
            return '警告: 当前象棋超出标定样本范围; ' + '; '.join(outside)
        return '标定区域: 当前象棋在样本范围内'

    def publish_live_status(self, text):
        if text == self.last_live_status:
            return
        self.last_live_status = text
        self.publish_move_status(text)

    def publish_move_status(self, text):
        msg = String()
        msg.data = str(text)
        self.move_status_pub.publish(msg)

    def validate_delta_target(self, delta_xyz_mm):
        if not bool(self.get_parameter('workspace_check_enabled').value):
            return True, ''
        x, y, z = [float(v) for v in delta_xyz_mm]
        min_x = float(self.get_parameter('min_x_mm').value)
        max_x = float(self.get_parameter('max_x_mm').value)
        min_y = float(self.get_parameter('min_y_mm').value)
        max_y = float(self.get_parameter('max_y_mm').value)
        min_z = float(self.get_parameter('min_z_mm').value)
        max_z = float(self.get_parameter('max_z_mm').value)
        safe_z = float(self.get_parameter('safe_xy_z_mm').value)

        if not (min_x <= x <= max_x):
            return False, 'X %.1f outside [%.1f, %.1f]' % (x, min_x, max_x)
        if not (min_y <= y <= max_y):
            return False, 'Y %.1f outside [%.1f, %.1f]' % (y, min_y, max_y)
        if not (min_z <= z <= max_z):
            return False, 'Z %.1f outside [%.1f, %.1f]' % (z, min_z, max_z)
        if not (min_z <= safe_z <= max_z):
            return False, 'safe Z %.1f outside [%.1f, %.1f]' % (safe_z, min_z, max_z)
        if bool(self.get_parameter('use_polygon_workspace').value):
            margin = float(self.get_parameter('polygon_margin_mm').value)
            self.safe_polygon = shrink_polygon(sort_boundary(BOUNDARY_POINTS_XY), margin)
            if not point_in_polygon(x, y, self.safe_polygon):
                return False, 'XY (%.1f, %.1f) outside measured safe polygon' % (x, y)
        return True, ''

    def publish_staged_move(self, delta_xyz_mm):
        x, y, z = [float(v) for v in delta_xyz_mm]
        safe_z = float(self.get_parameter('safe_xy_z_mm').value)
        feedrate = max(1.0, float(self.get_parameter('approach_feedrate').value))

        lines = [
            'G90',
            'G1 Z%.2f F%.2f' % (safe_z, feedrate),
            'G1 X%.2f Y%.2f Z%.2f F%.2f' % (x, y, safe_z, feedrate),
            'G1 X%.2f Y%.2f Z%.2f F%.2f' % (x, y, z, feedrate),
        ]
        msg = String()
        msg.data = '\n'.join(lines)
        self.raw_gcode_pub.publish(msg)
        self.get_logger().warning(
            'published staged chess move: safe_z=%.1f target=%s mm'
            % (safe_z, np.round(delta_xyz_mm, 1).tolist())
        )
        self.publish_move_status(
            'moved staged: safe_z=%.1f target_mm=%s'
            % (safe_z, np.round(delta_xyz_mm, 1).tolist())
        )


def main(args=None):
    rclpy.init(args=args)
    node = ChessHandeyeTarget()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
