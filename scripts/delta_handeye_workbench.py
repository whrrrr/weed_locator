#!/usr/bin/env python3
"""Independent GUI workbench for Delta hand-eye data collection and validation.

This program intentionally owns its data files and sampling logic.  It does not
call the legacy calibration or chess scripts.  ROS is used only as transport for
the camera, YOLO image, and raw Delta G-code topics.
"""

import copy
import json
import math
import queue
import shutil
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from PIL import Image as PilImage
from PIL import ImageTk
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Empty, String


ROOT = Path('/home/wyy/gpt_dev_ws/手眼标定_YAML文件')
CONFIG_DIR = ROOT / '配置文件'
OUTPUT_ROOT = ROOT / '运行输出'
RESET_ARCHIVE_DIR = OUTPUT_ROOT / '已归零历史'
DEFAULT_RAW = CONFIG_DIR / '手眼标定采点原始.yaml'
DEFAULT_RANGE = CONFIG_DIR / '标定运动范围.yaml'
DEFAULT_SETTINGS = CONFIG_DIR / '采点设置文件.yaml'
DEFAULT_KNOWN_BAD = CONFIG_DIR / '已知不符合要求点.yaml'
DEFAULT_EXEC = CONFIG_DIR / '执行设置.yaml'
FILTER_DIR = OUTPUT_ROOT / '范围筛选输出'
SAMPLE_DIR = OUTPUT_ROOT / '采点输出'
RESULT_DIR = OUTPUT_ROOT / '手眼标定结果'
SCAN_DIR = OUTPUT_ROOT / '象棋巡点输出'


def now_stamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def load_yaml(path, fallback=None):
    try:
        with Path(path).expanduser().open('r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle)
        return fallback if data is None else data
    except FileNotFoundError:
        return copy.deepcopy(fallback)


def write_yaml(path, data):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False, default_flow_style=False)
    tmp.replace(path)


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def point_in_polygon(x, y, polygon):
    """Boundary counts as valid; polygon is a list of [x, y]."""
    if not polygon or len(polygon) < 3:
        return True
    inside = False
    px, py = polygon[-1]
    for qx, qy in polygon:
        if min(px, qx) <= x <= max(px, qx) and min(py, qy) <= y <= max(py, qy):
            cross = (qx - px) * (y - py) - (qy - py) * (x - px)
            if abs(cross) < 1e-8:
                return True
        if (qy > y) != (py > y):
            x_cross = (px - qx) * (y - qy) / (py - qy) + qx
            if x <= x_cross:
                inside = not inside
        px, py = qx, qy
    return inside


def distance_xy(a, b):
    return math.hypot(float(a['x_mm']) - float(b['x_mm']), float(a['y_mm']) - float(b['y_mm']))


def fit_affine(camera_mm, delta_mm):
    source = np.asarray(camera_mm, dtype=float)
    target = np.asarray(delta_mm, dtype=float)
    design = np.column_stack([source, np.ones(len(source))])
    coeff, _residuals, rank, _singular = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ coeff
    errors = predicted - target
    return coeff, predicted, errors, int(rank)


def fit_plane(points_mm):
    points = np.asarray(points_mm, dtype=float)
    if len(points) < 3:
        return None
    center = points.mean(axis=0)
    _u, _s, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    signed = (points - center) @ normal
    return {
        'center_mm': center.tolist(),
        'normal_unit': normal.tolist(),
        'residual_mm': signed.tolist(),
        'rmse_mm': float(np.sqrt(np.mean(signed ** 2))),
        'max_abs_mm': float(np.max(np.abs(signed))),
    }


def xyz_abs_error_summary(errors):
    errors = np.abs(np.asarray(errors, dtype=float))
    names = ('X', 'Y', 'Z')
    return {
        name: {
            '最小绝对误差_mm': float(np.min(errors[:, index])),
            '中位数绝对误差_mm': float(np.median(errors[:, index])),
            '平均绝对误差_mm': float(np.mean(errors[:, index])),
            '最大绝对误差_mm': float(np.max(errors[:, index])),
        }
        for index, name in enumerate(names)
    }


def default_layers():
    # 54 candidates per layer: 9 x 6, leaving substantial replacement room
    # when the requested valid sample count is 15.
    points = [[x, y] for y in (-40, -16, 8, 32, 56, 80)
              for x in (-60, -45, -30, -15, 0, 15, 30, 45, 60)]
    return [{'z_mm': z, 'points': [{'x_mm': x, 'y_mm': y} for x, y in points]}
            for z in (-230, -240, -250, -260, -270, -290)]


def ensure_templates():
    ROOT.mkdir(parents=True, exist_ok=True)
    for directory in (CONFIG_DIR, OUTPUT_ROOT, RESET_ARCHIVE_DIR, FILTER_DIR, SAMPLE_DIR, RESULT_DIR, SCAN_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_RAW.exists():
        write_yaml(DEFAULT_RAW, {
            'format': 'delta_handeye_candidate_points/v1',
            'units': 'mm',
            'layers': default_layers(),
            'note': '编辑每层 z_mm 和 points。采集前会先被范围文件过滤。',
        })
    if not DEFAULT_RANGE.exists():
        write_yaml(DEFAULT_RANGE, {
            'format': 'delta_handeye_motion_range/v1',
            'units': 'mm',
            'default_polygon_xy_mm': [[-75, -55], [75, -55], [75, 95], [-75, 95]],
            'layers': {},
            'note': '默认范围和各层范围都必须是 Delta 坐标系中 Z 固定的 XY 平面多边形。',
        })
    if not DEFAULT_SETTINGS.exists():
        write_yaml(DEFAULT_SETTINGS, {
            'format': 'delta_handeye_sampling_settings/v1',
            'min_charuco_corners': 14,
            'stable_frames': 4,
            'stable_origin_tolerance_mm': 1.5,
            'max_wait_per_point_sec': 8.0,
            'initial_motion_wait_sec': 2.0,
            'samples_per_layer': 8,
            'start_layer_z_mm': -230.0,
            'layer_count': 0,
            'home_before_start': True,
            'home_after_each_point': False,
            'home_after_each_layer': True,
            'feedrate_mm_per_min': 80.0,
            'travel_z_mm': -210.0,
            'minimum_depth_corners': 4,
            'max_pnp_reprojection_error_px': 1.5,
        })
    if not DEFAULT_KNOWN_BAD.exists():
        write_yaml(DEFAULT_KNOWN_BAD, {'format': 'delta_handeye_known_bad/v1', 'radius_mm': 4.0, 'points': []})
    if not DEFAULT_EXEC.exists():
        write_yaml(DEFAULT_EXEC, {
            'format': 'delta_handeye_execution_settings/v1',
            'model_source': 'pnp',
            'offset_mm': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        })


class WorkbenchNode(Node):
    def __init__(self):
        super().__init__('delta_handeye_workbench')
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.color_image = None
        self.color_header = None
        self.color_sequence = 0
        self.color_camera_matrix = None
        self.color_distortion = None
        self.depth_image_m = None
        self.depth_header = None
        self.depth_camera_matrix = None
        self.charuco_image = None
        self.yolo_image = None
        self.yolo_detections = []
        self.yolo_selected_index = -1
        self.yolo_camera_xyz_m = None
        self.yolo_camera_time = 0.0
        self.yolo_pixel_uv = None
        self.yolo_pixel_time = 0.0
        self.yolo_pixel_history = deque(maxlen=40)
        self.gcode_pub = self.create_publisher(String, '/delta_arm/gcode_raw', 20)
        self.motor_enable_pub = self.create_publisher(Bool, '/delta_arm/motor_enable', 10)
        self.home_pub = self.create_publisher(Empty, '/delta_arm/home', 10)
        self.create_subscription(Image, '/camera/color/image_raw', self.on_color, 10)
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.on_color_info, 10)
        self.create_subscription(Image, '/camera/depth_registered/image_raw', self.on_depth, 10)
        self.create_subscription(CameraInfo, '/camera/depth_registered/camera_info', self.on_depth_info, 10)
        self.create_subscription(Image, '/chess/detection_image', self.on_yolo_image, 10)
        self.create_subscription(String, '/chess/detections_json', self.on_yolo_detections, 10)
        self.create_subscription(PointStamped, '/chess/camera_point', self.on_yolo_camera, 10)
        self.create_subscription(PointStamped, '/chess/selected_pixel_center', self.on_yolo_pixel, 10)

    def on_color(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.color_image, self.color_header = image, msg.header
                self.color_sequence += 1
        except Exception as exc:
            self.get_logger().warning('color image decode failed: %s' % exc)

    def on_color_info(self, msg):
        with self.lock:
            self.color_camera_matrix = np.asarray(msg.k, dtype=float).reshape(3, 3)
            self.color_distortion = np.asarray(msg.d, dtype=float).reshape(-1, 1)

    def on_depth(self, msg):
        try:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if raw.ndim != 2:
                return
            if msg.encoding in ('16UC1', 'mono16'):
                depth_m = raw.astype(np.float32) / 1000.0
            elif msg.encoding in ('32FC1', '32FC'):
                depth_m = raw.astype(np.float32)
            else:
                return
            with self.lock:
                self.depth_image_m, self.depth_header = depth_m, msg.header
        except Exception as exc:
            self.get_logger().warning('depth decode failed: %s' % exc)

    def on_depth_info(self, msg):
        with self.lock:
            self.depth_camera_matrix = np.asarray(msg.k, dtype=float).reshape(3, 3)

    def on_yolo_image(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.yolo_image = image
        except Exception:
            pass

    def on_yolo_detections(self, msg):
        try:
            payload = json.loads(msg.data)
            with self.lock:
                self.yolo_detections = list(payload.get('detections', []))
                self.yolo_selected_index = int(payload.get('selected_index', -1))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    def on_yolo_camera(self, msg):
        with self.lock:
            self.yolo_camera_xyz_m = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=float)
            self.yolo_camera_time = time.monotonic()

    def on_yolo_pixel(self, msg):
        with self.lock:
            self.yolo_pixel_uv = np.array([msg.point.x, msg.point.y], dtype=float)
            self.yolo_pixel_time = time.monotonic()
            self.yolo_pixel_history.append((self.yolo_pixel_time, self.yolo_pixel_uv.copy()))

    def send_gcode(self, line):
        msg = String()
        msg.data = str(line)
        self.gcode_pub.publish(msg)

    def set_motor_enabled(self, enabled):
        msg = Bool()
        msg.data = bool(enabled)
        self.motor_enable_pub.publish(msg)

    def home(self):
        self.home_pub.publish(Empty())

    def snapshot(self):
        with self.lock:
            return (
                None if self.color_image is None else self.color_image.copy(),
                None if self.color_camera_matrix is None else self.color_camera_matrix.copy(),
                None if self.color_distortion is None else self.color_distortion.copy(),
                None if self.depth_image_m is None else self.depth_image_m.copy(),
                None if self.depth_camera_matrix is None else self.depth_camera_matrix.copy(),
                self.color_sequence,
            )

    def pixel_to_camera_xyz_mm(self, pixel, depth, camera):
        if pixel is None or depth is None or camera is None:
            return None
        u, v = int(round(pixel[0])), int(round(pixel[1]))
        x0, x1 = max(0, u - 5), min(depth.shape[1], u + 6)
        y0, y1 = max(0, v - 5), min(depth.shape[0], v + 6)
        values = depth[y0:y1, x0:x1].reshape(-1)
        values = values[np.isfinite(values) & (values > 0.05) & (values < 2.0)]
        if not len(values):
            return None
        z = float(np.median(values))
        fx, fy, cx, cy = camera[0, 0], camera[1, 1], camera[0, 2], camera[1, 2]
        if fx == 0.0 or fy == 0.0:
            return None
        return np.array([(pixel[0] - cx) * z / fx, (pixel[1] - cy) * z / fy, z], dtype=float) * 1000.0

    def yolo_camera_xyz_mm(self, max_pixel_age_sec=2.0):
        """Reconstruct the currently selected YOLO pixel from registered depth.

        This intentionally lives in the workbench instead of depending on the
        legacy chess hand-eye node.  The depth topic is already registered into
        the color camera image plane by depth_to_color_registration.
        """
        with self.lock:
            pixel = None if self.yolo_pixel_uv is None else self.yolo_pixel_uv.copy()
            depth = None if self.depth_image_m is None else self.depth_image_m.copy()
            camera = None if self.depth_camera_matrix is None else self.depth_camera_matrix.copy()
            age = time.monotonic() - self.yolo_pixel_time
        if pixel is None or depth is None or camera is None or age > max_pixel_age_sec:
            return None
        return self.pixel_to_camera_xyz_mm(pixel, depth, camera)

    def recent_yolo_camera_xyz_mm(self, window_sec=0.5):
        """Return the newest RGB-D-valid chess detection from a short history."""
        with self.lock:
            depth = None if self.depth_image_m is None else self.depth_image_m.copy()
            camera = None if self.depth_camera_matrix is None else self.depth_camera_matrix.copy()
            history = list(self.yolo_pixel_history)
        now = time.monotonic()
        for stamp, pixel in reversed(history):
            if now - stamp > window_sec:
                break
            point = self.pixel_to_camera_xyz_mm(pixel, depth, camera)
            if point is not None and np.all(np.isfinite(point)):
                return point
        return None


class HandeyeWorkbench:
    def __init__(self, root, node):
        ensure_templates()
        self.root, self.node = root, node
        self.stop_requested = False
        self.busy = False
        self.current_status = '空闲'
        self.progress = {'mode': '空闲', 'current': 0, 'total': 0, 'layer': '-', 'detail': '-'}
        self.latest_charuco = None
        self.last_display_charuco = None
        self.last_display_charuco_time = 0.0
        self.latest_scan_offset = None
        self.last_live_charuco_time = 0.0
        self.charuco_preview_busy = False
        self.charuco_preview_lock = threading.Lock()
        self.charuco_detector_lock = threading.Lock()
        self.charuco_photo = None
        self.yolo_photo = None
        self.charuco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.charuco_board = cv2.aruco.CharucoBoard((8, 5), 0.020, 0.014, self.charuco_dictionary)
        self.charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board)
        self.settings_entries = {}
        self.raw_path = tk.StringVar(value=str(DEFAULT_RAW))
        self.range_path = tk.StringVar(value=str(DEFAULT_RANGE))
        self.settings_path = tk.StringVar(value=str(DEFAULT_SETTINGS))
        self.bad_path = tk.StringVar(value=str(DEFAULT_KNOWN_BAD))
        self.output_path = tk.StringVar(value='')
        self.result_path = tk.StringVar(value='')
        self.scan_source_path = tk.StringVar(value='')
        self.scan_output_path = tk.StringVar(value='')
        self.exec_result_path = tk.StringVar(value='')
        self.exec_settings_path = tk.StringVar(value=str(DEFAULT_EXEC))
        self.exec_model = tk.StringVar(value='pnp')
        self.exec_x = tk.StringVar(value='0.0')
        self.exec_y = tk.StringVar(value='0.0')
        self.exec_z = tk.StringVar(value='0.0')
        self.exec_speed = tk.StringVar(value='80.0')
        self.manual_x = tk.StringVar(value='0.0')
        self.manual_y = tk.StringVar(value='0.0')
        self.manual_z = tk.StringVar(value='-230.0')
        self.new_layer_z = tk.StringVar(value='-290.0')
        self.new_layer_count = tk.StringVar(value='1')
        self.new_layer_step = tk.StringVar(value='10.0')
        self.status_var = tk.StringVar(value='就绪')
        self.progress_var = tk.StringVar(value='进度：空闲')
        self.charuco_info_var = tk.StringVar(value='ChArUco：等待彩色相机与内参')
        self.yolo_info_var = tk.StringVar(value='象棋：等待 YOLO')
        self.yolo_delta_var = tk.StringVar(value='手眼转换 Delta 坐标：等待模型结果文件和象棋坐标')
        self.preview_model_cache_path = None
        self.preview_model_cache_mtime = None
        self.preview_model_cache = None
        existing_results = sorted(RESULT_DIR.glob('*_手眼标定结果文件.yaml'))
        if existing_results:
            self.result_path.set(str(existing_results[-1]))
            self.exec_result_path.set(str(existing_results[-1]))
        self.root.title('Delta 手眼标定工作台')
        self.root.geometry('1500x980')
        self.root.minsize(1180, 800)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.build_ui()
        self.load_settings_to_ui()
        self.load_exec_to_ui()
        self.root.after(10, self.spin_ros)
        self.root.after(16, self.refresh_ui)

    def build_ui(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill='both', expand=True)
        left = ttk.Frame(outer)
        left.pack(side='left', fill='both', expand=True)
        right = ttk.Frame(outer, width=500)
        right.pack(side='right', fill='both', padx=(8, 0))
        right.pack_propagate(False)
        views = ttk.PanedWindow(left, orient='vertical')
        views.pack(fill='both', expand=True)
        charuco_frame = ttk.LabelFrame(views, text='OpenCV ChArUco 标定画面')
        yolo_frame = ttk.LabelFrame(views, text='YOLO 象棋画面')
        self.charuco_label = ttk.Label(charuco_frame, anchor='center')
        self.charuco_label.pack(fill='both', expand=True)
        self.yolo_label = ttk.Label(yolo_frame, anchor='center')
        self.yolo_label.pack(fill='both', expand=True)
        views.add(charuco_frame, weight=1)
        views.add(yolo_frame, weight=1)
        ttk.Label(left, textvariable=self.charuco_info_var).pack(anchor='w', pady=(5, 0))
        ttk.Label(left, textvariable=self.yolo_info_var).pack(anchor='w')
        ttk.Label(left, textvariable=self.yolo_delta_var, justify='left', wraplength=900).pack(anchor='w')
        ttk.Label(left, textvariable=self.progress_var, font=('Sans', 11, 'bold')).pack(anchor='w', pady=(3, 0))
        ttk.Label(left, textvariable=self.status_var).pack(anchor='w')
        arm = ttk.Frame(left)
        arm.pack(fill='x', pady=(4, 0))
        ttk.Label(arm, text='机械臂：').pack(side='left')
        ttk.Button(arm, text='使能', command=self.enable_motors).pack(side='left', padx=(3, 0))
        ttk.Button(arm, text='失能', command=self.disable_motors).pack(side='left', padx=(3, 0))
        ttk.Button(arm, text='归零 G28', command=self.home_arm).pack(side='left', padx=(3, 0))
        ttk.Button(arm, text='恢复未标定状态（仅清空输出）', command=self.reset_runtime_outputs).pack(side='left', padx=(14, 0))

        tabs = ttk.Notebook(right)
        tabs.pack(fill='both', expand=True)
        collect = ttk.Frame(tabs, padding=8)
        model = ttk.Frame(tabs, padding=8)
        scan = ttk.Frame(tabs, padding=8)
        execute = ttk.Frame(tabs, padding=8)
        tabs.add(collect, text='采集')
        tabs.add(model, text='模型')
        tabs.add(scan, text='巡点')
        tabs.add(execute, text='执行')
        self.build_collect_tab(collect)
        self.build_model_tab(model)
        self.build_scan_tab(scan)
        self.build_execute_tab(execute)

    def path_row(self, parent, label, variable, browse=True):
        row = ttk.Frame(parent)
        row.pack(fill='x', pady=2)
        ttk.Label(row, text=label, width=14).pack(side='left')
        ttk.Entry(row, textvariable=variable).pack(side='left', fill='x', expand=True)
        if browse:
            ttk.Button(row, text='选择', command=lambda: self.choose_yaml(variable)).pack(side='left', padx=(4, 0))

    def build_collect_tab(self, parent):
        ttk.Label(parent, text='文件选择', font=('Sans', 11, 'bold')).pack(anchor='w')
        self.path_row(parent, '原始候选点', self.raw_path)
        self.path_row(parent, '运动范围', self.range_path)
        self.path_row(parent, '采点设置', self.settings_path)
        self.path_row(parent, '已知坏点', self.bad_path)
        self.path_row(parent, '本次输出', self.output_path, False)
        buttons = ttk.Frame(parent)
        buttons.pack(fill='x', pady=6)
        ttk.Button(buttons, text='范围过滤并预览', command=self.filter_preview).pack(side='left')
        ttk.Button(buttons, text='开始采集', command=self.start_collection).pack(side='left', padx=5)
        ttk.Button(buttons, text='停止当前任务', command=self.request_stop).pack(side='left')
        ttk.Button(buttons, text='把当前位置记为坏点', command=self.mark_current_bad).pack(side='left', padx=5)
        ttk.Separator(parent).pack(fill='x', pady=5)
        ttk.Label(parent, text='向原始 YAML 添加标定层', font=('Sans', 10, 'bold')).pack(anchor='w')
        add_layer = ttk.Frame(parent)
        add_layer.pack(fill='x', pady=3)
        for label, variable, width in [
            ('起始 Z', self.new_layer_z, 9), ('层数', self.new_layer_count, 5), ('向下间距', self.new_layer_step, 7),
        ]:
            ttk.Label(add_layer, text=label).pack(side='left', padx=(2, 0))
            ttk.Entry(add_layer, textvariable=variable, width=width).pack(side='left', padx=(2, 5))
        ttk.Button(add_layer, text='添加层到原始 YAML', command=self.add_layers_to_raw_yaml).pack(side='left')
        ttk.Separator(parent).pack(fill='x', pady=5)
        ttk.Label(parent, text='采点设置（直接改后点击保存）', font=('Sans', 11, 'bold')).pack(anchor='w')
        form = ttk.Frame(parent)
        form.pack(fill='x', pady=4)
        fields = [
            ('min_charuco_corners', '最少 corner'), ('stable_frames', '稳定帧数'),
            ('stable_origin_tolerance_mm', '稳定阈值 mm'), ('max_wait_per_point_sec', '单点最长等待 s'),
            ('initial_motion_wait_sec', '运动后初等 s'), ('samples_per_layer', '每层采点数'),
            ('start_layer_z_mm', '开始层 Z mm'), ('layer_count', '采集层数(0=全部)'),
            ('home_before_start', '开始前回零'), ('home_after_each_point', '每点后回零'),
            ('home_after_each_layer', '每层后回零'), ('feedrate_mm_per_min', '速度 mm/s（固件 F）'),
            ('travel_z_mm', '横移安全 Z'), ('minimum_depth_corners', '最少深度 corner'),
            ('max_pnp_reprojection_error_px', 'PnP 重投影 px'),
        ]
        for index, (key, text) in enumerate(fields):
            row, col = divmod(index, 2)
            box = ttk.Frame(form)
            box.grid(row=row, column=col, sticky='ew', padx=2, pady=2)
            ttk.Label(box, text=text, width=15).pack(side='left')
            var = tk.StringVar()
            ttk.Entry(box, textvariable=var, width=12).pack(side='left', fill='x', expand=True)
            self.settings_entries[key] = var
        form.columnconfigure(0, weight=1)
        form.columnconfigure(1, weight=1)
        ttk.Button(parent, text='保存采点设置', command=self.save_settings_from_ui).pack(anchor='w', pady=5)

    def build_model_tab(self, parent):
        ttk.Label(parent, text='从采点输出 YAML 生成两套模型', font=('Sans', 11, 'bold')).pack(anchor='w')
        self.path_row(parent, '采点输出', self.output_path)
        self.path_row(parent, '模型结果', self.result_path, False)
        ttk.Button(parent, text='生成 PnP / 深度模型与误差报告', command=self.start_model_fit).pack(anchor='w', pady=6)
        ttk.Label(parent, text='每种路线都会写入：公式系数、参与拟合的样本、斜平面残差、逐点反推误差，以及输入输出 YAML 路径。', wraplength=450).pack(anchor='w')

    def build_scan_tab(self, parent):
        ttk.Label(parent, text='巡点：将象棋放在末端，逐点记录象棋相机坐标并与 Delta 指令坐标对比。', wraplength=450).pack(anchor='w')
        self.path_row(parent, '巡点来源(结果/采点/自建)', self.scan_source_path)
        self.path_row(parent, '模型结果', self.result_path)
        self.path_row(parent, '巡点输出', self.scan_output_path, False)
        ttk.Button(parent, text='开始巡点', command=self.start_scan).pack(anchor='w', pady=6)
        self.scan_offset_var = tk.StringVar(value='当前成功巡检点 XYZ offset：等待巡检数据')
        ttk.Label(parent, textvariable=self.scan_offset_var, justify='left', wraplength=450).pack(anchor='w', pady=(2, 6))
        ttk.Label(parent, text='每个点先等待机械臂与相机稳定；稳定后最多等待 5 秒 YOLO 象棋坐标，超时会记录失败并继续下一点。', wraplength=450).pack(anchor='w')

    def build_execute_tab(self, parent):
        ttk.Label(parent, text='把当前 YOLO 象棋相机坐标转换为 Delta 目标并执行。', wraplength=450).pack(anchor='w')
        self.path_row(parent, '实时转换结果', self.exec_result_path)
        self.path_row(parent, '执行设置', self.exec_settings_path)
        ttk.Label(parent, text='点击“选择”可换用任何手眼标定结果 YAML；实时 PnP / 深度 Delta 坐标和执行命令都会使用这里选择的文件。', wraplength=450).pack(anchor='w', pady=(0, 5))
        row = ttk.Frame(parent)
        row.pack(fill='x', pady=5)
        ttk.Label(row, text='模型').pack(side='left')
        ttk.Combobox(row, textvariable=self.exec_model, values=['pnp', 'depth'], state='readonly', width=12).pack(side='left', padx=5)
        for label, var in [('X 偏移', self.exec_x), ('Y 偏移', self.exec_y), ('Z 偏移', self.exec_z)]:
            ttk.Label(row, text=label).pack(side='left', padx=(8, 0))
            ttk.Entry(row, textvariable=var, width=7).pack(side='left')
        speed_row = ttk.Frame(parent)
        speed_row.pack(fill='x', pady=(0, 5))
        ttk.Label(speed_row, text='机械臂速度 mm/s（固件 F）').pack(side='left')
        ttk.Entry(speed_row, textvariable=self.exec_speed, width=9).pack(side='left', padx=5)
        ttk.Button(parent, text='保存执行设置', command=self.save_exec_from_ui).pack(anchor='w')
        ttk.Button(parent, text='将当前象棋转换并移动', command=self.start_execute).pack(anchor='w', pady=8)
        ttk.Separator(parent).pack(fill='x', pady=8)
        ttk.Label(parent, text='直接移动到输入坐标（Delta mm）', font=('Sans', 10, 'bold')).pack(anchor='w')
        manual = ttk.Frame(parent)
        manual.pack(fill='x', pady=4)
        for label, variable in [('X', self.manual_x), ('Y', self.manual_y), ('Z', self.manual_z)]:
            ttk.Label(manual, text=label).pack(side='left', padx=(4, 0))
            ttk.Entry(manual, textvariable=variable, width=9).pack(side='left', padx=(2, 6))
        ttk.Button(parent, text='移动到输入 XYZ', command=self.start_manual_move).pack(anchor='w')
        ttk.Separator(parent).pack(fill='x', pady=8)
        ttk.Label(parent, text='吸嘴控制', font=('Sans', 10, 'bold')).pack(anchor='w')
        vacuum = ttk.Frame(parent)
        vacuum.pack(fill='x', pady=4)
        ttk.Button(vacuum, text='开启吸嘴', command=self.start_suction_on).pack(side='left')
        ttk.Button(vacuum, text='关闭吸嘴', command=self.start_suction_off).pack(side='left', padx=5)
        ttk.Button(vacuum, text='释放棋子', command=self.start_suction_release).pack(side='left')
        ttk.Label(parent, text='开启：关闭泄气阀后开泵。关闭：只停泵。释放：打开泄气阀约 0.4 秒后停泵并关闭阀门。', wraplength=450).pack(anchor='w')
        ttk.Label(parent, text='传送带控制', font=('Sans', 10, 'bold')).pack(anchor='w', pady=(8, 0))
        conveyor = ttk.Frame(parent)
        conveyor.pack(fill='x', pady=4)
        ttk.Button(conveyor, text='停止', command=self.start_conveyor_stop).pack(side='left')
        ttk.Button(conveyor, text='正转低速', command=lambda: self.start_conveyor_speed('正转', '低速', 'M201')).pack(side='left', padx=5)
        ttk.Button(conveyor, text='正转中速', command=lambda: self.start_conveyor_speed('正转', '中速', 'M202')).pack(side='left', padx=5)
        ttk.Button(conveyor, text='正转高速', command=lambda: self.start_conveyor_speed('正转', '高速', 'M203')).pack(side='left')
        conveyor_reverse = ttk.Frame(parent)
        conveyor_reverse.pack(fill='x', pady=(0, 4))
        ttk.Button(conveyor_reverse, text='反转低速', command=lambda: self.start_conveyor_speed('反转', '低速', 'M204')).pack(side='left')
        ttk.Button(conveyor_reverse, text='反转中速', command=lambda: self.start_conveyor_speed('反转', '中速', 'M205')).pack(side='left', padx=5)
        ttk.Button(conveyor_reverse, text='反转高速', command=lambda: self.start_conveyor_speed('反转', '高速', 'M206')).pack(side='left')
        ttk.Label(parent, text='传送带：停止 M200；正转低/中/高 M201/M202/M203；反转低/中/高 M204/M205/M206。', wraplength=450).pack(anchor='w')
        ttk.Label(parent, text='执行采用固定的三段运动：先到 Z=-210 mm，再横移 XY，最后到目标 Z。模型和偏移只从这里的执行设置读取。', wraplength=450).pack(anchor='w')

    def choose_yaml(self, variable):
        path = filedialog.askopenfilename(filetypes=[('YAML', '*.yaml *.yml'), ('全部文件', '*')])
        if path:
            variable.set(path)

    def reset_runtime_outputs(self):
        """Return the workbench to an uncalibrated active state without touching config YAMLs."""
        if self.busy:
            messagebox.showwarning('任务正在运行', '请先停止当前采集、拟合或巡点任务，再清空运行输出。')
            return
        confirmed = messagebox.askyesno(
            '恢复未标定状态',
            '这会清空当前可见的范围筛选、采点、手眼结果和象棋巡点输出。\n\n'
            '旧输出会移动到“运行输出/已归零历史/时间戳”而不是直接删除。\n'
            '“配置文件”目录中的原始候选点、运动范围、采点设置、坏点和执行设置完全不会改动。\n\n'
            '继续吗？',
        )
        if not confirmed:
            return
        archived = RESET_ARCHIVE_DIR / ('%s_归零前运行输出' % now_stamp())
        moved = []
        for output_dir in (FILTER_DIR, SAMPLE_DIR, RESULT_DIR, SCAN_DIR):
            if output_dir.exists() and any(output_dir.iterdir()):
                archived.mkdir(parents=True, exist_ok=True)
                shutil.move(str(output_dir), str(archived / output_dir.name))
                moved.append(output_dir.name)
            output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path.set('')
        self.result_path.set('')
        self.scan_source_path.set('')
        self.scan_output_path.set('')
        self.exec_result_path.set('')
        self.preview_model_cache_path = None
        self.preview_model_cache_mtime = None
        self.preview_model_cache = None
        self.latest_scan_offset = None
        self.current_status = '已恢复未标定状态：清空活动运行输出%s；配置文件未修改。' % (
            '，旧输出已归档到 %s' % archived if moved else '',
        )

    def add_layers_to_raw_yaml(self):
        try:
            start_z = float(self.new_layer_z.get())
            count = int(self.new_layer_count.get())
            step = abs(float(self.new_layer_step.get()))
        except ValueError:
            messagebox.showerror('层参数无效', '起始 Z、层数、向下间距必须是数值。')
            return
        if count <= 0 or step <= 0.0:
            messagebox.showerror('层参数无效', '层数和向下间距必须大于 0。')
            return
        data = load_yaml(self.raw_path.get(), {})
        layers = data.setdefault('layers', [])
        if not layers:
            messagebox.showerror('原始 YAML 无点', '原始 YAML 至少需要已有一层作为候选 XY 网格模板。')
            return
        template = copy.deepcopy(layers[0].get('points', []))
        existing = [as_float(layer.get('z_mm')) for layer in layers]
        added = []
        for index in range(count):
            z = start_z - index * step
            if any(abs(z - old_z) < 0.01 for old_z in existing):
                continue
            layers.append({'z_mm': z, 'points': copy.deepcopy(template)})
            existing.append(z)
            added.append(z)
        layers.sort(key=lambda layer: as_float(layer.get('z_mm')), reverse=True)
        write_yaml(self.raw_path.get(), data)
        if added:
            self.current_status = '已添加标定层 Z=%s；每层复制 %d 个候选 XY 点。' % (added, len(template))
        else:
            self.current_status = '没有添加：这些 Z 层已存在于原始 YAML。'

    def load_settings_to_ui(self):
        data = load_yaml(self.settings_path.get(), {})
        fallback = {'start_layer_z_mm': -230.0, 'layer_count': 0, 'initial_motion_wait_sec': 2.0}
        for key, var in self.settings_entries.items():
            var.set(str(data.get(key, fallback.get(key, ''))))

    def settings_from_ui(self):
        bool_keys = {'home_before_start', 'home_after_each_point', 'home_after_each_layer'}
        int_keys = {'min_charuco_corners', 'stable_frames', 'samples_per_layer', 'minimum_depth_corners', 'layer_count'}
        data = {}
        for key, var in self.settings_entries.items():
            raw = var.get().strip()
            if key in bool_keys:
                data[key] = raw.lower() in ('1', 'true', 'yes', 'on', '是')
            elif key in int_keys:
                data[key] = int(float(raw))
            else:
                data[key] = float(raw)
        data['format'] = 'delta_handeye_sampling_settings/v1'
        return data

    def save_settings_from_ui(self):
        try:
            write_yaml(self.settings_path.get(), self.settings_from_ui())
            self.status_var.set('已保存采点设置。')
        except Exception as exc:
            messagebox.showerror('保存失败', str(exc))

    def load_exec_to_ui(self):
        data = load_yaml(self.exec_settings_path.get(), {})
        self.exec_model.set(str(data.get('model_source', 'pnp')))
        offset = data.get('offset_mm', {})
        self.exec_x.set(str(offset.get('x', 0.0)))
        self.exec_y.set(str(offset.get('y', 0.0)))
        self.exec_z.set(str(offset.get('z', 0.0)))
        self.exec_speed.set(str(data.get('arm_feedrate_mm_per_sec', 80.0)))

    def save_exec_from_ui(self):
        speed = as_float(self.exec_speed.get(), -1.0)
        if speed <= 0.0:
            messagebox.showerror('执行设置无效', '机械臂速度必须大于 0 mm/s。')
            return False
        data = {'format': 'delta_handeye_execution_settings/v1', 'model_source': self.exec_model.get(),
                'offset_mm': {'x': as_float(self.exec_x.get()), 'y': as_float(self.exec_y.get()), 'z': as_float(self.exec_z.get())},
                'arm_feedrate_mm_per_sec': speed}
        write_yaml(self.exec_settings_path.get(), data)
        self.status_var.set('已保存执行设置。')
        return True

    def normalized_layers(self, source_path):
        data = load_yaml(source_path, {})
        layers = data.get('layers', []) if isinstance(data, dict) else []
        normalized = []
        for layer in layers:
            z = as_float(layer.get('z_mm'))
            points = []
            for point in layer.get('points', []):
                if isinstance(point, dict):
                    points.append({'x_mm': as_float(point.get('x_mm')), 'y_mm': as_float(point.get('y_mm'))})
            if points:
                normalized.append({'z_mm': z, 'points': points})
        if normalized:
            return normalized
        # Also accept a simple mapping: z230: [[x,y], ...]
        if isinstance(data, dict):
            for key, values in data.items():
                if not isinstance(values, list):
                    continue
                z_text = str(key).lower().replace('z', '').replace('_mm', '')
                try:
                    z = as_float(z_text)
                except Exception:
                    continue
                points = []
                for value in values:
                    if isinstance(value, dict):
                        points.append({'x_mm': as_float(value.get('x_mm')), 'y_mm': as_float(value.get('y_mm'))})
                    elif isinstance(value, (list, tuple)) and len(value) >= 2:
                        points.append({'x_mm': as_float(value[0]), 'y_mm': as_float(value[1])})
                if points:
                    normalized.append({'z_mm': z, 'points': points})
        return normalized

    def filtered_layers(self, source_path, range_path, bad_path):
        layers = self.normalized_layers(source_path)
        ranges = load_yaml(range_path, {}) if range_path else {}
        known_bad = load_yaml(bad_path, {'points': []})
        radius = as_float(known_bad.get('radius_mm', 4.0), 4.0)
        default_polygon = ranges.get('default_polygon_xy_mm') if isinstance(ranges, dict) else None
        per_layer = ranges.get('layers', {}) if isinstance(ranges, dict) else {}
        bad_points = known_bad.get('points', []) if isinstance(known_bad, dict) else []
        result = []
        report = []
        for layer in layers:
            z_key = str(int(layer['z_mm']))
            entry = per_layer.get(z_key, per_layer.get(str(layer['z_mm']), {})) if isinstance(per_layer, dict) else {}
            polygon = entry.get('polygon_xy_mm', default_polygon) if isinstance(entry, dict) else default_polygon
            kept = []
            rejected = []
            for point in layer['points']:
                item = dict(point)
                item['z_mm'] = layer['z_mm']
                reason = None
                if polygon and not point_in_polygon(point['x_mm'], point['y_mm'], polygon):
                    reason = 'outside_motion_range'
                if reason is None:
                    for bad in bad_points:
                        if abs(as_float(bad.get('z_mm')) - layer['z_mm']) < 0.1 and distance_xy(point, bad) <= radius:
                            reason = 'known_bad_neighborhood'
                            break
                if reason:
                    rejected.append({**item, 'reason': reason})
                else:
                    kept.append(item)
            result.append({'z_mm': layer['z_mm'], 'points': kept, 'polygon_xy_mm': polygon or []})
            report.append({'z_mm': layer['z_mm'], 'kept': len(kept), 'rejected': rejected})
        return result, report

    def filter_preview(self):
        try:
            layers, report = self.filtered_layers(self.raw_path.get(), self.range_path.get(), self.bad_path.get())
            range_stem = Path(self.range_path.get()).stem
            raw_stem = Path(self.raw_path.get()).stem
            path = FILTER_DIR / ('%s_%s_范围筛选后.yaml' % (range_stem, raw_stem))
            write_yaml(path, {'format': 'delta_handeye_filtered_candidates/v1', 'source_raw_yaml': self.raw_path.get(),
                              'range_yaml': self.range_path.get(), 'known_bad_yaml': self.bad_path.get(),
                              'created_at': datetime.now().isoformat(timespec='seconds'), 'layers': layers,
                              'filter_report': report})
            self.status_var.set('范围过滤完成：%s' % path)
            messagebox.showinfo('范围过滤', '已写入：\n%s' % path)
        except Exception as exc:
            messagebox.showerror('范围过滤失败', str(exc))

    def select_uniform_points(self, points, desired):
        if len(points) <= desired:
            return list(points)
        array = np.array([[p['x_mm'], p['y_mm']] for p in points], dtype=float)
        selected = [int(np.argmin(np.sum((array - array.mean(axis=0)) ** 2, axis=1)))]
        while len(selected) < desired:
            dist = np.min(np.linalg.norm(array[:, None, :] - array[selected][None, :, :], axis=2), axis=1)
            dist[selected] = -1.0
            selected.append(int(np.argmax(dist)))
        return [points[index] for index in selected]

    def start_worker(self, name, function, *args):
        if self.busy:
            self.status_var.set('已有任务在运行。')
            return
        self.busy, self.stop_requested = True, False
        def run():
            try:
                self.current_status = name
                function(*args)
            except Exception as exc:
                self.node.get_logger().error('%s failed: %s' % (name, exc))
                self.current_status = '%s 失败：%s' % (name, exc)
            finally:
                self.busy = False
                if self.current_status == name:
                    self.current_status = '%s 完成' % name
        threading.Thread(target=run, daemon=True).start()

    def request_stop(self):
        self.stop_requested = True
        self.status_var.set('已请求在当前点结束后停止。')

    def enable_motors(self):
        self.node.set_motor_enabled(True)
        self.current_status = '已发送电机使能 M17。'

    def disable_motors(self):
        self.node.set_motor_enabled(False)
        self.current_status = '已发送电机失能 M18。'

    def home_arm(self):
        if self.busy:
            self.current_status = '采集/巡点任务运行中，不能从 GUI 单独归零。先停止任务。'
            return
        self.node.home()
        self.current_status = '已发送归零 G28。'

    def start_suction_on(self):
        self.start_worker('开启吸嘴', self.suction_on_worker)

    def suction_on_worker(self):
        self.node.send_gcode('M2')
        time.sleep(0.10)
        self.node.send_gcode('M121')
        self.current_status = '吸嘴已开启：泄气阀关闭，气泵开启。'

    def start_suction_off(self):
        self.start_worker('关闭吸嘴', self.suction_off_worker)

    def suction_off_worker(self):
        self.node.send_gcode('M122')
        self.current_status = '吸嘴已关闭：气泵停止，未主动泄压。'

    def start_suction_release(self):
        self.start_worker('释放棋子', self.suction_release_worker)

    def suction_release_worker(self):
        self.node.send_gcode('M1')
        time.sleep(0.40)
        self.node.send_gcode('M122')
        time.sleep(0.10)
        self.node.send_gcode('M2')
        self.current_status = '已释放：泄气 0.4 秒，气泵停止，泄气阀复位关闭。'

    def start_conveyor_stop(self):
        self.start_worker('停止传送带', self.conveyor_stop_worker)

    def conveyor_stop_worker(self):
        self.node.send_gcode('M200')
        self.current_status = '已发送传送带停止 M200。'

    def start_conveyor_speed(self, direction_name, speed_name, gcode):
        self.start_worker(
            f'传送带{direction_name}{speed_name}',
            lambda: self.conveyor_speed_worker(direction_name, speed_name, gcode),
        )

    def conveyor_speed_worker(self, direction_name, speed_name, gcode):
        self.node.send_gcode(gcode)
        self.current_status = f'已发送传送带{direction_name}{speed_name}：{gcode}。'

    def start_collection(self):
        try:
            settings = self.settings_from_ui()
        except Exception as exc:
            messagebox.showerror('设置无效', str(exc))
            return
        self.save_settings_from_ui()
        layers, filter_report = self.filtered_layers(self.raw_path.get(), self.range_path.get(), self.bad_path.get())
        if not any(layer['points'] for layer in layers):
            messagebox.showerror('没有可采点', '范围过滤和已知坏点过滤后没有可用候选点。')
            return
        start_z = as_float(settings['start_layer_z_mm'])
        start_index = next((index for index, layer in enumerate(layers) if abs(layer['z_mm'] - start_z) < 0.01), None)
        if start_index is None:
            available = ', '.join(str(layer['z_mm']) for layer in layers)
            messagebox.showerror('开始层不存在', '开始层 Z=%.1f 不在范围筛选后的层中。可选：%s' % (start_z, available))
            return
        layer_count = int(settings['layer_count'])
        layers = layers[start_index:] if layer_count <= 0 else layers[start_index:start_index + layer_count]
        layers = [layer for layer in layers if layer['points']]
        if not layers:
            messagebox.showerror('没有可采层', '所选层没有通过范围筛选的候选点。')
            return
        selected_file = FILTER_DIR / ('%s_%s_范围筛选后.yaml' % (Path(self.range_path.get()).stem, Path(self.raw_path.get()).stem))
        write_yaml(selected_file, {'format': 'delta_handeye_filtered_candidates/v1', 'source_raw_yaml': self.raw_path.get(),
                                   'range_yaml': self.range_path.get(), 'known_bad_yaml': self.bad_path.get(), 'layers': layers,
                                   'filter_report': filter_report})
        output = SAMPLE_DIR / ('%s_手眼标定采点输出.yaml' % now_stamp())
        self.output_path.set(str(output))
        self.start_worker('采集', self.collect_worker, layers, settings, selected_file, output)

    def home(self, settings):
        self.node.home()
        time.sleep(max(1.0, as_float(settings['initial_motion_wait_sec']) + 3.0))

    def move_to(self, point, settings):
        z = as_float(point['z_mm'])
        x, y = as_float(point['x_mm']), as_float(point['y_mm'])
        f = as_float(settings['feedrate_mm_per_min'])
        travel_z = as_float(settings['travel_z_mm'])
        self.node.send_gcode('G90')
        self.node.send_gcode('G1 Z%.2f F%.2f' % (travel_z, f))
        self.node.send_gcode('G1 X%.2f Y%.2f Z%.2f F%.2f' % (x, y, travel_z, f))
        self.node.send_gcode('G1 X%.2f Y%.2f Z%.2f F%.2f' % (x, y, z, f))

    def charuco_observation(self, settings):
        image, camera, distortion, depth, depth_camera, image_seq = self.node.snapshot()
        if image is None or camera is None:
            return {'valid': False, 'reason': 'missing_color_image_or_camera_info', 'image_seq': image_seq}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        board = self.charuco_board
        with self.charuco_detector_lock:
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)
        marker_count = 0 if marker_ids is None else int(len(marker_ids))
        count = 0 if charuco_ids is None else int(len(charuco_ids))
        marker_pixel_corners = [] if marker_corners is None else [
            np.asarray(marker, dtype=float).reshape(-1, 2).tolist() for marker in marker_corners
        ]
        common = {'marker_count': marker_count, 'corner_count': count, 'corner_ids': [], 'corner_pixel_uv': [],
                  'marker_pixel_corners': marker_pixel_corners,
                  'image_seq': image_seq,
                  'pnp': {'valid': False}, 'depth': {'valid': False}}
        if charuco_corners is None or charuco_ids is None:
            common.update({'valid': False, 'reason': 'no_charuco_corners'})
            return common
        ids = charuco_ids.reshape(-1).astype(int)
        pixels = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
        all_board = np.asarray(board.getChessboardCorners(), dtype=np.float32)
        object_points = all_board[ids].reshape(-1, 1, 3)
        image_points = pixels.reshape(-1, 1, 2)
        common['corner_ids'] = [int(v) for v in ids]
        common['corner_pixel_uv'] = [[float(u), float(v)] for u, v in pixels]
        if count < int(settings['min_charuco_corners']):
            common.update({'valid': False, 'reason': 'too_few_corners'})
            return common
        ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            common.update({'valid': False, 'reason': 'solvepnp_failed'})
            return common
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera, distortion)
        reproj = np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)
        common['pnp'] = {'valid': bool(float(np.mean(reproj)) <= as_float(settings['max_pnp_reprojection_error_px'])),
                         'origin_camera_xyz_mm': (tvec.reshape(3) * 1000.0).tolist(),
                         'corner_camera_xyz_mm': (((cv2.Rodrigues(rvec)[0] @ object_points.reshape(-1, 3).T).T + tvec.reshape(1, 3)) * 1000.0).astype(float).tolist(),
                         'corner_reprojection_error_px': [float(value) for value in reproj],
                         'mean_reprojection_error_px': float(np.mean(reproj)), 'max_reprojection_error_px': float(np.max(reproj))}
        if depth is None or depth_camera is None or depth.shape[:2] != image.shape[:2]:
            common['depth'] = {'valid': False, 'reason': 'missing_or_unaligned_registered_depth'}
        else:
            fx, fy, cx, cy = depth_camera[0, 0], depth_camera[1, 1], depth_camera[0, 2], depth_camera[1, 2]
            depth_points, board_points, depth_records, valid_depth_records = [], [], [], []
            for cid, pixel, board_point in zip(ids, pixels, object_points.reshape(-1, 3)):
                u, v = int(round(pixel[0])), int(round(pixel[1]))
                x0, x1 = max(0, u - 2), min(depth.shape[1], u + 3)
                y0, y1 = max(0, v - 2), min(depth.shape[0], v + 3)
                values = depth[y0:y1, x0:x1].reshape(-1)
                values = values[np.isfinite(values) & (values > 0.05) & (values < 2.0)]
                if not len(values):
                    depth_records.append({'corner_id': int(cid), 'pixel_uv': [float(pixel[0]), float(pixel[1])],
                                          'camera_xyz_mm': None, 'depth_mm': None, 'valid': False,
                                          'reason': 'no_valid_depth'})
                    continue
                z = float(np.median(values))
                p = np.array([(pixel[0] - cx) * z / fx, (pixel[1] - cy) * z / fy, z])
                depth_points.append(p)
                board_points.append(board_point)
                record = {'corner_id': int(cid), 'pixel_uv': [float(pixel[0]), float(pixel[1])],
                          'camera_xyz_mm': (p * 1000.0).tolist(), 'depth_mm': z * 1000.0, 'valid': True}
                depth_records.append(record)
                valid_depth_records.append(record)
            if len(depth_points) < int(settings['minimum_depth_corners']):
                common['depth'] = {'valid': False, 'reason': 'too_few_valid_depth_corners', 'corner_camera_xyz_mm': depth_records}
            else:
                b = np.asarray(board_points, dtype=float)
                d = np.asarray(depth_points, dtype=float)
                bc, dc = b.mean(axis=0), d.mean(axis=0)
                u_mat, _s, vt = np.linalg.svd((b - bc).T @ (d - dc))
                rotation = vt.T @ u_mat.T
                if np.linalg.det(rotation) < 0:
                    vt[-1] *= -1
                    rotation = vt.T @ u_mat.T
                origin = dc - rotation @ bc
                residual = np.linalg.norm((rotation @ b.T).T + origin - d, axis=1) * 1000.0
                for record, error_mm in zip(valid_depth_records, residual):
                    record['board_fit_error_mm'] = float(error_mm)
                common['depth'] = {'valid': True, 'origin_camera_xyz_mm': (origin * 1000.0).tolist(),
                                   'corner_camera_xyz_mm': depth_records, 'fit_rmse_mm': float(np.sqrt(np.mean(residual ** 2))),
                                   'fit_max_mm': float(np.max(residual)), 'valid_corner_count': len(depth_records)}
        common['valid'] = bool(common['pnp']['valid'])
        common['reason'] = 'ok' if common['valid'] else 'pnp_reprojection_error'
        return common

    def stable_observation(self, settings, min_image_seq=-1):
        deadline = time.monotonic() + as_float(settings['max_wait_per_point_sec'])
        required = int(settings['stable_frames'])
        tolerance = as_float(settings['stable_origin_tolerance_mm'])
        stable, last = [], None
        last_image_seq = int(min_image_seq)
        # Do not use a pre-command frame. After the configured initial wait,
        # accept only a sequence of new camera frames whose board pose settles.
        time.sleep(max(0.0, as_float(settings['initial_motion_wait_sec'])))
        while time.monotonic() < deadline and not self.stop_requested:
            item = self.charuco_observation(settings)
            self.latest_charuco = item
            image_seq = int(item.get('image_seq', -1))
            if image_seq <= last_image_seq:
                time.sleep(0.02)
                continue
            last_image_seq = image_seq
            if not item.get('valid'):
                stable, last = [], None
            else:
                origin = np.asarray(item['pnp']['origin_camera_xyz_mm'], dtype=float)
                if last is None or np.linalg.norm(origin - last) <= tolerance:
                    stable.append(item)
                else:
                    stable = [item]
                last = origin
                if len(stable) >= required:
                    chosen = copy.deepcopy(stable[-1])
                    for route in ('pnp', 'depth'):
                        valid = [x[route] for x in stable if x.get(route, {}).get('valid')]
                        if valid and 'origin_camera_xyz_mm' in valid[-1]:
                            origins = np.asarray([x['origin_camera_xyz_mm'] for x in valid], dtype=float)
                            chosen[route]['origin_camera_xyz_mm'] = np.median(origins, axis=0).tolist()
                    chosen['stable_frames_observed'] = len(stable)
                    chosen['position_stability'] = {
                        'initial_wait_sec': as_float(settings['initial_motion_wait_sec']),
                        'new_frames_after_command': len(stable),
                        'tolerance_mm': float(tolerance),
                    }
                    return chosen
            time.sleep(0.05)
        return {
            'valid': False, 'reason': 'stable_position_timeout', 'stable_frames_observed': len(stable),
            'last_image_seq': last_image_seq,
            'position_stability': {
                'initial_wait_sec': as_float(settings['initial_motion_wait_sec']),
                'new_frames_after_command': len(stable), 'tolerance_mm': float(tolerance),
            },
        }

    def collect_worker(self, layers, settings, selected_path, output_path):
        records = {'format': 'delta_handeye_samples/v2', 'created_at': datetime.now().isoformat(timespec='seconds'),
                   'source_raw_yaml': self.raw_path.get(), 'range_yaml': self.range_path.get(),
                   'filtered_candidate_yaml': str(selected_path), 'settings_yaml': self.settings_path.get(),
                   'known_bad_yaml': self.bad_path.get(), 'layers': [], 'all_attempts_including_rejected': True}
        if settings['home_before_start']:
            self.home(settings)
        total = sum(len(layer['points']) for layer in layers)
        done = 0
        accepted_total = 0
        for layer in layers:
            if self.stop_requested:
                break
            desired = int(settings['samples_per_layer'])
            planned = self.select_uniform_points(layer['points'], desired)
            remaining = [p for p in layer['points'] if p not in planned]
            accepted, attempts = [], []
            queue_points = list(planned)
            while queue_points:
                candidate = queue_points.pop(0)
                if self.stop_requested or len(accepted) >= desired:
                    break
                point = {**candidate, 'z_mm': layer['z_mm']}
                done += 1
                self.progress = {
                    'mode': '采集', 'current': done, 'total': total, 'layer': layer['z_mm'], 'detail': '移动到 %.1f, %.1f' % (point['x_mm'], point['y_mm']),
                    'candidate_done': done, 'candidate_total': total, 'candidate_left': max(0, total - done),
                    'accepted_layer': len(accepted), 'requested_layer': desired, 'accepted_total': accepted_total,
                }
                before_move = self.charuco_observation(settings)
                image_seq_before_move = int(before_move.get('image_seq', -1))
                self.move_to(point, settings)
                observation = self.stable_observation(settings, min_image_seq=image_seq_before_move)
                accepted_flag = bool(observation.get('valid'))
                attempt = {'delta_target_mm': [point['x_mm'], point['y_mm'], point['z_mm']], 'accepted': accepted_flag,
                           'observed_at': datetime.now().isoformat(timespec='seconds'), **{k: v for k, v in observation.items() if k != 'debug_image'}}
                attempts.append(attempt)
                if accepted_flag:
                    accepted.append(attempt)
                    accepted_total += 1
                    self.progress['accepted_layer'] = len(accepted)
                    self.progress['accepted_total'] = accepted_total
                elif remaining:
                    # A failed seed is replaced by its nearest still-untried
                    # candidate, exactly so a local bad spot does not create a
                    # large hole in that layer's sampling coverage.
                    replacement = min(remaining, key=lambda item: distance_xy(item, candidate))
                    remaining.remove(replacement)
                    queue_points.insert(0, replacement)
                write_yaml(output_path, {**records, 'layers': records['layers'] + [{'z_mm': layer['z_mm'], 'accepted_count': len(accepted), 'requested_count': desired, 'accepted_samples': accepted, 'attempts': attempts}]})
                if settings['home_after_each_point'] and not self.stop_requested:
                    self.home(settings)
            records['layers'].append({'z_mm': layer['z_mm'], 'accepted_count': len(accepted), 'requested_count': desired,
                                      'accepted_samples': accepted, 'attempts': attempts})
            write_yaml(output_path, records)
            if settings['home_after_each_layer'] and not self.stop_requested:
                self.home(settings)
        self.output_path.set(str(output_path))
        self.current_status = '采集完成：%s' % output_path

    def mark_current_bad(self):
        if self.busy:
            self.status_var.set('任务运行中，不能改坏点文件。')
            return
        # The last commanded target is recorded by the progress object.
        detail = self.progress.get('detail', '')
        if '移动到' not in detail:
            messagebox.showwarning('无法记录', '还没有当前采样点。请在采集过程中停止后再标记。')
            return
        try:
            xy = detail.replace('移动到 ', '').split(',')
            point = {'x_mm': as_float(xy[0]), 'y_mm': as_float(xy[1]), 'z_mm': as_float(self.progress.get('layer')),
                     'reason': 'manual_mark', 'marked_at': datetime.now().isoformat(timespec='seconds')}
            data = load_yaml(self.bad_path.get(), {'format': 'delta_handeye_known_bad/v1', 'radius_mm': 4.0, 'points': []})
            data.setdefault('points', []).append(point)
            write_yaml(self.bad_path.get(), data)
            self.status_var.set('已写入已知不符合要求点：%s' % point)
        except Exception as exc:
            messagebox.showerror('记录坏点失败', str(exc))

    def start_model_fit(self):
        if not self.output_path.get() or not Path(self.output_path.get()).exists():
            messagebox.showerror('缺少数据', '先选择已有的手眼标定采点输出 YAML。')
            return
        out = RESULT_DIR / ('%s_手眼标定结果文件.yaml' % now_stamp())
        self.result_path.set(str(out))
        self.exec_result_path.set(str(out))
        self.start_worker('模型计算', self.model_worker, Path(self.output_path.get()), out)

    def route_model(self, samples, route):
        usable = [s for s in samples if s.get('accepted') and s.get(route, {}).get('valid') and s[route].get('origin_camera_xyz_mm')]
        if len(usable) < 4:
            return {'valid': False, 'reason': 'need_at_least_4_valid_samples', 'sample_count': len(usable)}
        camera = np.array([s[route]['origin_camera_xyz_mm'] for s in usable], dtype=float)
        delta = np.array([s['delta_target_mm'] for s in usable], dtype=float)
        coeff, predicted, errors, rank = fit_affine(camera, delta)
        norms = np.linalg.norm(errors, axis=1)
        return {'valid': True, 'formula': 'delta_xyz = [camera_x, camera_y, camera_z, 1] @ coefficients_4x3',
                'coefficients_4x3': coeff.tolist(), 'rank': rank, 'sample_count': len(usable),
                'camera_origin_plane': fit_plane(camera), 'delta_target_plane': fit_plane(delta),
                'back_predicted_delta_plane': fit_plane(predicted),
                'reverse_prediction_error_mm': {'rmse_xyz': np.sqrt(np.mean(errors ** 2, axis=0)).tolist(),
                                                'rmse_norm': float(np.sqrt(np.mean(norms ** 2))), 'mean_norm': float(np.mean(norms)),
                                                'median_norm': float(np.median(norms)), 'max_norm': float(np.max(norms)),
                                                'xyz_absolute_error_summary': xyz_abs_error_summary(errors),
                                                'per_sample': [{'delta_target_mm': s['delta_target_mm'], 'camera_origin_mm': c.tolist(),
                                                                'predicted_delta_mm': p.tolist(), 'error_mm': e.tolist(), 'error_norm_mm': float(n)}
                                                               for s, c, p, e, n in zip(usable, camera, predicted, errors, norms)]},
                'corner_error_summary': self.corner_error_summary(usable, route),
                'samples_used': usable}

    @staticmethod
    def basic_error_summary(values, unit, meaning):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return {'可用': False, '原因': '没有可统计的 corner 误差'}
        return {
            '可用': True,
            '这是什么': meaning,
            '单位': unit,
            'corner数量': int(values.size),
            '最小值': float(np.min(values)),
            '中位数': float(np.median(values)),
            '平均值': float(np.mean(values)),
            '最大值': float(np.max(values)),
        }

    def corner_error_summary(self, samples, route):
        if route == 'pnp':
            values = [error for sample in samples for error in sample.get('pnp', {}).get('corner_reprojection_error_px', [])]
            return self.basic_error_summary(
                values, 'pixel',
                'PnP 把已知标定板 corner 三维位置投影回彩色图后，与实际检测 corner 像素位置之间的距离。越小表示 PnP 对当前图像角点的解释越一致。',
            )
        values = [
            record['board_fit_error_mm']
            for sample in samples
            for record in sample.get('depth', {}).get('corner_camera_xyz_mm', [])
            if record.get('valid') and record.get('board_fit_error_mm') is not None
        ]
        return self.basic_error_summary(
            values, 'mm',
            '深度重建出的 corner 三维点与拟合标定板刚体后的距离。越小表示深度 corner 更接近同一块平面标定板。',
        )

    def pnp_depth_corner_difference_summary(self, samples):
        diffs = []
        for sample in samples:
            pnp = sample.get('pnp', {})
            depth = sample.get('depth', {})
            pnp_ids = pnp.get('corner_ids', sample.get('corner_ids', []))
            pnp_xyz = pnp.get('corner_camera_xyz_mm', [])
            pnp_by_id = {int(cid): np.asarray(xyz, dtype=float) for cid, xyz in zip(pnp_ids, pnp_xyz)}
            for record in depth.get('corner_camera_xyz_mm', []):
                if not record.get('valid') or record.get('camera_xyz_mm') is None:
                    continue
                cid = int(record.get('corner_id', -1))
                if cid in pnp_by_id:
                    diffs.append(np.asarray(record['camera_xyz_mm'], dtype=float) - pnp_by_id[cid])
        if not diffs:
            return {'可用': False, '原因': '没有同时具备 PnP 与深度三维坐标的同编号 corner'}
        diff_array = np.asarray(diffs, dtype=float)
        norm = np.linalg.norm(diff_array, axis=1)
        return {
            '可用': True,
            '这是什么': '同一编号 ChArUco corner 的 深度三维坐标 - PnP 三维坐标。用于直接衡量两条识别路线彼此差多少。',
            '同编号corner数量': len(diffs),
            'XYZ绝对差统计_mm': xyz_abs_error_summary(diff_array),
            '三维距离统计_mm': {
                '最小值': float(np.min(norm)), '中位数': float(np.median(norm)),
                '平均值': float(np.mean(norm)), '最大值': float(np.max(norm)),
            },
        }

    def evaluate_model_on_route(self, model, samples, source_route):
        """Apply one fitted model to either its own or the other route's points."""
        usable = [s for s in samples if s.get('accepted') and s.get(source_route, {}).get('valid')
                  and s[source_route].get('origin_camera_xyz_mm')]
        if not model.get('valid'):
            return {'valid': False, 'reason': 'model_invalid', 'sample_count': len(usable)}
        if not usable:
            return {'valid': False, 'reason': 'no_valid_source_samples', 'sample_count': 0}
        camera = np.asarray([s[source_route]['origin_camera_xyz_mm'] for s in usable], dtype=float)
        actual = np.asarray([s['delta_target_mm'] for s in usable], dtype=float)
        coeff = np.asarray(model['coefficients_4x3'], dtype=float)
        predicted = np.column_stack([camera, np.ones(len(camera))]) @ coeff
        errors = predicted - actual
        norms = np.linalg.norm(errors, axis=1)
        return {
            'valid': True,
            'input_route': source_route,
            'sample_count': len(usable),
            'input_camera_points_plane': fit_plane(camera),
            'predicted_delta_points_plane': fit_plane(predicted),
            'error_mm': {
                'rmse_xyz': np.sqrt(np.mean(errors ** 2, axis=0)).tolist(),
                'rmse_norm': float(np.sqrt(np.mean(norms ** 2))),
                'mean_norm': float(np.mean(norms)),
                'median_norm': float(np.median(norms)),
                'max_norm': float(np.max(norms)),
                'per_sample': [
                    {
                        'delta_target_mm': s['delta_target_mm'],
                        'camera_origin_mm': c.tolist(),
                        'predicted_delta_mm': p.tolist(),
                        'error_mm': e.tolist(),
                        'error_norm_mm': float(n),
                    }
                    for s, c, p, e, n in zip(usable, camera, predicted, errors, norms)
                ],
            },
        }

    @staticmethod
    def compact_evaluation(evaluation):
        if not evaluation.get('valid'):
            return {
                '这组检查能不能用': False,
                '为什么不能用': evaluation.get('reason'),
                '参与点数': evaluation.get('sample_count', 0),
            }
        return {
            '这组检查能不能用': True,
            '这是什么': '把某一套手眼公式代入指定来源的相机点，再与机械臂当时实际目标坐标比较。误差越小，说明这套公式在这批点上越可信。',
            '参与点数': evaluation['sample_count'],
            '坐标反推误差_mm': {key: evaluation['error_mm'][key] for key in ('rmse_xyz', 'rmse_norm', 'mean_norm', 'median_norm', 'max_norm')},
            '输入相机点拟合斜平面误差_mm': {
                'rmse': evaluation['input_camera_points_plane']['rmse_mm'],
                'max_abs': evaluation['input_camera_points_plane']['max_abs_mm'],
            },
            '公式反推Delta点平面检查_不代表XY精度': {
                '说明': '本层训练目标的 Delta Z 本来就是固定值，因此该平面误差天然接近 0；它只确认输出仍在同一 Z 层，不能证明 XY 转换准确。',
                'rmse': evaluation['predicted_delta_points_plane']['rmse_mm'],
                'max_abs': evaluation['predicted_delta_points_plane']['max_abs_mm'],
            },
        }

    def compact_model(self, model):
        if not model.get('valid'):
            return {
                '模型能不能生成': False,
                '为什么不能生成': model.get('reason'),
                '可用点数': model.get('sample_count', 0),
            }
        return {
            '模型能不能生成': True,
            '这是什么': '这是一套把相机坐标 camera_XYZ 换算成机械臂坐标 Delta_XYZ 的公式。',
            '公式': model['formula'],
            '公式系数_4x3': model['coefficients_4x3'],
            '参与拟合点数': model['sample_count'],
            '识别corner误差摘要': model['corner_error_summary'],
            '本模型输入相机点拟合斜平面误差_mm': {
                'rmse': model['camera_origin_plane']['rmse_mm'], 'max_abs': model['camera_origin_plane']['max_abs_mm'],
            },
            '本模型反推Delta点平面检查_不代表XY精度': {
                '说明': '这一层的机械臂目标 Z 固定，所以这里接近 0 是正常数学结果，不能把它当作手眼标定精度。真正看 XY 要看 本模型用自身数据反推误差 和 交叉验证误差。',
                'rmse': model['back_predicted_delta_plane']['rmse_mm'], 'max_abs': model['back_predicted_delta_plane']['max_abs_mm'],
            },
            '本模型用自身数据反推误差_mm': {
                key: model['reverse_prediction_error_mm'][key]
                for key in ('rmse_xyz', 'rmse_norm', 'mean_norm', 'median_norm', 'max_norm', 'xyz_absolute_error_summary')
            },
        }

    def model_worker(self, sample_path, result_path):
        data = load_yaml(sample_path, {})
        global_samples = [item for layer in data.get('layers', []) for item in layer.get('accepted_samples', [])]
        sampling_completeness = []
        for layer in data.get('layers', []):
            requested = int(layer.get('requested_count', 0))
            accepted = int(layer.get('accepted_count', len(layer.get('accepted_samples', []))))
            sampling_completeness.append({
                'z_mm': layer.get('z_mm'),
                '要求合格点数': requested,
                '实际合格点数': accepted,
                '尝试过的候选点数': len(layer.get('attempts', [])),
                '这一层是否采够': accepted >= requested,
                '说明': '未采够时仍会保留并计算临时模型，但该模型只能用于诊断，不应作为实际执行模型。',
            })
        global_pnp = self.route_model(global_samples, 'pnp')
        global_depth = self.route_model(global_samples, 'depth')
        global_checks = {
            'PNP模型_用PNP点反推': self.evaluate_model_on_route(global_pnp, global_samples, 'pnp'),
            'PNP模型_用深度点交叉验证': self.evaluate_model_on_route(global_pnp, global_samples, 'depth'),
            '深度模型_用深度点反推': self.evaluate_model_on_route(global_depth, global_samples, 'depth'),
            '深度模型_用PNP点交叉验证': self.evaluate_model_on_route(global_depth, global_samples, 'pnp'),
        }
        layers = []
        for layer in data.get('layers', []):
            samples = layer.get('accepted_samples', [])
            pnp_model = self.route_model(samples, 'pnp')
            depth_model = self.route_model(samples, 'depth')
            checks = {
                'PNP模型_用PNP点反推': self.evaluate_model_on_route(pnp_model, samples, 'pnp'),
                'PNP模型_用深度点交叉验证': self.evaluate_model_on_route(pnp_model, samples, 'depth'),
                '深度模型_用深度点反推': self.evaluate_model_on_route(depth_model, samples, 'depth'),
                '深度模型_用PNP点交叉验证': self.evaluate_model_on_route(depth_model, samples, 'pnp'),
            }
            layers.append({'z_mm': layer.get('z_mm'), 'pnp_model': pnp_model, 'depth_model': depth_model, 'cross_validation': checks})
        summary = {
            '这份文件是干嘛的': '它记录同一批标定点用 PnP 和深度两种方式得到的手眼标定公式，并验证两套公式各自和互相使用时的误差。单位全部是 mm。',
            '本次采样是否达到要求': {
                '是否全部采够': all(item['这一层是否采够'] for item in sampling_completeness),
                '每层采样完成情况': sampling_completeness,
                '结论': '只有每层实际合格点数达到要求点数，且交叉验证误差也足够小，该层模型才建议用于机械臂执行。',
            },
            '怎么看': [
                '先看 PNP模型 和 深度模型 的 本模型用自身数据反推误差_mm：这是公式拿原始采样点反算时的误差，越小越好。',
                '再看 交叉验证：PNP模型_用深度点交叉验证 和 深度模型_用PNP点交叉验证，表示两条路线的相机坐标能不能互相代入另一套公式。',
                '输入相机点拟合斜平面误差表示同一机械臂层在相机坐标系中是否仍像一个平面。rmse 和 max_abs 越小，说明该层点越稳定。',
                '反推Delta点平面检查在固定 Z 层必然接近 0，它不代表 XY 精度，不能拿来判断模型好坏。',
                '完整样本、所有 corner、逐点误差和每个公式的全部系数仍在 模型详情_完整保留，没有删除。',
            ],
            '全局_PNP模型': self.compact_model(global_pnp),
            '全局_深度模型': self.compact_model(global_depth),
            'PNP与深度_同编号corner三维差异摘要': self.pnp_depth_corner_difference_summary(global_samples),
            '全局_交叉验证': {key: self.compact_evaluation(value) for key, value in global_checks.items()},
            '每层模型摘要': [
                {
                    'z_mm': layer['z_mm'],
                    'PNP模型': self.compact_model(layer['pnp_model']),
                    '深度模型': self.compact_model(layer['depth_model']),
                    '交叉验证': {key: self.compact_evaluation(value) for key, value in layer['cross_validation'].items()},
                }
                for layer in layers
            ],
        }
        result = {
            'format': 'delta_handeye_result/v3',
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'source_sample_output_yaml': str(sample_path),
            '重要摘要_先看这里': summary,
            '模型详情_完整保留': {
                'global_pnp_model': global_pnp,
                'global_depth_model': global_depth,
                'global_cross_validation': global_checks,
                'layer_models': layers,
            },
        }
        write_yaml(result_path, result)
        self.current_status = '模型已写入：%s' % result_path

    def scan_points(self, source):
        data = load_yaml(source, {})
        if str(data.get('format', '')).startswith('delta_handeye_result'):
            details = data.get('模型详情_完整保留', data)
            grouped = {}
            seen = set()
            for layer in details.get('layer_models', []):
                z = as_float(layer.get('z_mm'))
                for route in ('pnp', 'depth'):
                    for sample in layer.get('%s_model' % route, {}).get('samples_used', []):
                        target = sample.get('delta_target_mm', [])
                        if len(target) != 3:
                            continue
                        key = tuple(round(as_float(value), 4) for value in target)
                        if key in seen:
                            continue
                        seen.add(key)
                        grouped.setdefault(z, []).append({'x_mm': as_float(target[0]), 'y_mm': as_float(target[1])})
            if not grouped:
                for route in ('pnp', 'depth'):
                    for sample in details.get('global_%s_model' % route, {}).get('samples_used', []):
                        target = sample.get('delta_target_mm', [])
                        if len(target) != 3:
                            continue
                        key = tuple(round(as_float(value), 4) for value in target)
                        if key in seen:
                            continue
                        seen.add(key)
                        z = as_float(target[2])
                        grouped.setdefault(z, []).append({'x_mm': as_float(target[0]), 'y_mm': as_float(target[1])})
            return [{'z_mm': z, 'points': points} for z, points in grouped.items()]
        if data.get('format', '').startswith('delta_handeye_samples'):
            return [{'z_mm': layer['z_mm'], 'points': [{'x_mm': item['delta_target_mm'][0], 'y_mm': item['delta_target_mm'][1]}
                                                        for item in layer.get('accepted_samples', [])]}
                    for layer in data.get('layers', [])]
        return self.normalized_layers(source)

    def start_scan(self):
        source = self.scan_source_path.get() or self.result_path.get() or self.output_path.get()
        if not source or not Path(source).exists() or not self.result_path.get() or not Path(self.result_path.get()).exists():
            messagebox.showerror('缺少文件', '巡点需要来源 YAML 和手眼标定结果文件。')
            return
        out = SCAN_DIR / ('%s_象棋巡点输出.yaml' % now_stamp())
        self.scan_output_path.set(str(out))
        settings = self.settings_from_ui()
        layers = self.scan_points(source)
        if not any(layer.get('points') for layer in layers):
            messagebox.showerror('没有巡检点', '来源文件中没有可回访的 Delta 点。')
            return
        self.start_worker('象棋巡点', self.scan_worker, layers, Path(self.result_path.get()), settings, out, source)

    def choose_layer_model(self, result, route, z):
        details = result.get('模型详情_完整保留', result)
        options = [(abs(as_float(layer.get('z_mm')) - z), layer.get('%s_model' % route, {})) for layer in details.get('layer_models', [])]
        valid = [item for item in options if item[1].get('valid')]
        if valid:
            return min(valid, key=lambda item: item[0])[1]
        return details.get('global_%s_model' % route, {})

    def choose_model_for_camera(self, result, route, camera_mm):
        details = result.get('模型详情_完整保留', result)
        candidates = []
        for layer in details.get('layer_models', []):
            model = layer.get('%s_model' % route, {})
            plane = model.get('camera_origin_plane') if model.get('valid') else None
            if plane is None:
                continue
            normal = np.asarray(plane['normal_unit'], dtype=float)
            center = np.asarray(plane['center_mm'], dtype=float)
            candidates.append((abs(float((np.asarray(camera_mm) - center) @ normal)), model, layer.get('z_mm')))
        if candidates:
            return min(candidates, key=lambda item: item[0])[1], min(candidates, key=lambda item: item[0])[2]
        return details.get('global_%s_model' % route, {}), None

    def active_handeye_result_path(self):
        """Use the execute selection when valid, otherwise a valid newly fitted result."""
        for candidate in (self.exec_result_path.get(), self.result_path.get()):
            if candidate and Path(candidate).exists():
                return Path(candidate)
        return None

    @staticmethod
    def normalize_handeye_result(result):
        """Accept the old one-layer dual model as a read-only selectable result.

        The current GUI writes 4x3 XYZ affine models. Earlier diagnostics wrote
        4x2 camera-XYZ -> Delta-XY coefficients with one fixed Delta Z layer.
        Normalize that old layout only for preview/execution; no source YAML is
        edited and no claim is made that an old model remains valid after moving
        the camera.
        """
        if not isinstance(result, dict) or 'pnp_model_camera_xyz_to_delta_xy' not in result:
            return result
        z_mm = as_float(result.get('delta_z_m')) * 1000.0

        def old_route(key):
            coeff_xy = np.asarray(result.get(key, {}).get('coeffs', []), dtype=float)
            if coeff_xy.shape != (4, 2):
                return {'valid': False, 'reason': 'legacy_coefficients_are_not_4x2'}
            coeff_xyz = np.zeros((4, 3), dtype=float)
            coeff_xyz[:, :2] = coeff_xy
            coeff_xyz[3, 2] = z_mm
            return {
                'valid': True,
                'formula': 'legacy: delta_xy = [camera_x, camera_y, camera_z, 1] @ coefficients_4x2; delta_z=fixed',
                'coefficients_4x3': coeff_xyz.tolist(),
                'legacy_fixed_delta_z_mm': z_mm,
            }

        return {
            'format': 'delta_handeye_legacy_dual_model_adapter/v1',
            'legacy_source_description': result.get('description', ''),
            '模型详情_完整保留': {
                'global_pnp_model': old_route('pnp_model_camera_xyz_to_delta_xy'),
                'global_depth_model': old_route('depth_model_camera_xyz_to_delta_xy'),
                'layer_models': [],
            },
        }

    def preview_handeye_result(self):
        path = self.active_handeye_result_path()
        if path is None:
            return None
        try:
            mtime = path.stat().st_mtime
            if path != self.preview_model_cache_path or mtime != self.preview_model_cache_mtime:
                self.preview_model_cache_path = path
                self.preview_model_cache_mtime = mtime
                self.preview_model_cache = self.normalize_handeye_result(load_yaml(path, {}))
            return self.preview_model_cache
        except OSError:
            return None

    def apply_model(self, model, camera_mm):
        if not model.get('valid'):
            return None
        coeff = np.asarray(model['coefficients_4x3'], dtype=float)
        return (np.append(np.asarray(camera_mm, dtype=float), 1.0) @ coeff).tolist()

    def stable_yolo(self, timeout_sec=5.0, min_pixel_received_time=0.0):
        deadline = time.monotonic() + timeout_sec
        samples = []
        while time.monotonic() < deadline and not self.stop_requested:
            with self.node.lock:
                pixel_time = self.node.yolo_pixel_time
            # A scan must observe the chess again after the robot has moved;
            # otherwise an old pre-move detection could be paired with a new pose.
            value = self.node.yolo_camera_xyz_mm() if pixel_time > min_pixel_received_time else None
            if value is not None and np.all(np.isfinite(value)):
                samples.append(value.copy())
                if len(samples) >= 3 and np.max(np.linalg.norm(np.asarray(samples[-3:]) - np.median(samples[-3:], axis=0), axis=1)) < 3.0:
                    return np.median(samples[-3:], axis=0).tolist()
            time.sleep(0.06)
        return None

    @staticmethod
    def scan_offset_summary(records, route):
        key = '%s_offset_to_delta_mm' % route
        usable = [record for record in records if record.get(key) is not None]
        if not usable:
            return {'可用': False, '原因': '没有成功获得象棋相机坐标或模型预测坐标'}
        raw = np.asarray([record[key] for record in usable], dtype=float)
        median = np.median(raw, axis=0)
        mean = np.mean(raw, axis=0)
        corrected = raw - median
        for record, before, after in zip(usable, raw, corrected):
            record['%s_选定中位数offset_mm' % route] = median.tolist()
            record['%s_手动补偿参考值_mm_不会自动执行' % route] = median.tolist()
            record['%s_补偿后剩余误差_mm' % route] = after.tolist()
            record['%s_补偿后剩余误差范数_mm' % route] = float(np.linalg.norm(after))
        before_norm = np.linalg.norm(raw, axis=1)
        after_norm = np.linalg.norm(corrected, axis=1)
        return {
            '可用': True,
            '这是什么': 'offset 定义为 模型预测的 Delta 坐标 - 该巡检点真实 Delta 坐标。中位数 offset 只用于诊断和人工决定是否填写执行设置；巡检程序不会自动修改执行 offset，也不会自动影响机械臂运动。',
            '参与巡检点数': len(usable),
            '平均XYZ_offset_mm': {'X': float(mean[0]), 'Y': float(mean[1]), 'Z': float(mean[2])},
            '中位数XYZ_offset_mm': {'X': float(median[0]), 'Y': float(median[1]), 'Z': float(median[2])},
            '每个成功巡检点的XYZ_offset_mm': [
                {
                    '巡检点序号': record['index'],
                    '真实Delta_XYZ_mm': record['delta_target_mm'],
                    'X': float(record[key][0]), 'Y': float(record[key][1]), 'Z': float(record[key][2]),
                }
                for record in usable
            ],
            '补偿前误差范数统计_mm': {
                '最小值': float(np.min(before_norm)), '中位数': float(np.median(before_norm)),
                '平均值': float(np.mean(before_norm)), '最大值': float(np.max(before_norm)),
            },
            '使用中位数offset补偿后误差范数统计_mm': {
                '最小值': float(np.min(after_norm)), '中位数': float(np.median(after_norm)),
                '平均值': float(np.mean(after_norm)), '最大值': float(np.max(after_norm)),
            },
            '每个点的补偿后误差已写入巡检记录': True,
        }

    def scan_worker(self, layers, result_path, settings, out_path, source_path):
        result = self.normalize_handeye_result(load_yaml(result_path, {}))
        report = {'format': 'delta_handeye_chess_scan/v1', 'created_at': datetime.now().isoformat(timespec='seconds'),
                  'scan_source_yaml': str(source_path), 'handeye_result_yaml': str(result_path), 'records': []}
        all_points = [{'x_mm': p['x_mm'], 'y_mm': p['y_mm'], 'z_mm': layer['z_mm']} for layer in layers for p in layer['points']]
        for index, point in enumerate(all_points, 1):
            if self.stop_requested:
                break
            self.progress = {'mode': '巡点', 'current': index, 'total': len(all_points), 'layer': point['z_mm'], 'detail': '移动到 %.1f, %.1f' % (point['x_mm'], point['y_mm'])}
            with self.node.lock:
                pixel_before_move = self.node.yolo_pixel_time
            self.move_to(point, settings)
            # A chess-on-tool scan must not require a ChArUco board.  The chess
            # center itself is the only visual object needed here: three stable
            # RGB-D observations mean the arm and target have settled enough to
            # record this verification point.
            time.sleep(max(0.0, as_float(settings['initial_motion_wait_sec'])))
            chess_camera_mm = self.stable_yolo(5.0, min_pixel_received_time=pixel_before_move)
            record = {'index': index, 'delta_target_mm': [point['x_mm'], point['y_mm'], point['z_mm']],
                      'motion_stable': chess_camera_mm is not None,
                      'motion_reason': 'yolo_chess_xyz_stable' if chess_camera_mm is not None else 'yolo_chess_xyz_timeout',
                      'chess_camera_xyz_mm': chess_camera_mm}
            for route in ('pnp', 'depth'):
                model = self.choose_layer_model(result, route, point['z_mm'])
                predicted = self.apply_model(model, chess_camera_mm) if chess_camera_mm else None
                record['%s_model_delta_xyz_mm' % route] = predicted
                record['%s_offset_to_delta_mm' % route] = None if predicted is None else (np.asarray(predicted) - np.array([point['x_mm'], point['y_mm'], point['z_mm']])).tolist()
            record['每点XYZ_offset_先看这里'] = {
                'PNP模型': None if record['pnp_offset_to_delta_mm'] is None else {
                    'X': record['pnp_offset_to_delta_mm'][0], 'Y': record['pnp_offset_to_delta_mm'][1], 'Z': record['pnp_offset_to_delta_mm'][2],
                },
                '深度模型': None if record['depth_offset_to_delta_mm'] is None else {
                    'X': record['depth_offset_to_delta_mm'][0], 'Y': record['depth_offset_to_delta_mm'][1], 'Z': record['depth_offset_to_delta_mm'][2],
                },
            }
            self.latest_scan_offset = record['每点XYZ_offset_先看这里']
            report['records'].append(record)
            write_yaml(out_path, report)
        report['offset_summary_先看这里'] = {
            'PNP模型': self.scan_offset_summary(report['records'], 'pnp'),
            '深度模型': self.scan_offset_summary(report['records'], 'depth'),
        }
        report['说明'] = '巡检点来自手眼标定结果文件时，程序会回访该结果文件中实际参与模型拟合的有效点。每条巡检记录保留补偿前 offset 和使用中位数 offset 后的剩余误差。'
        write_yaml(out_path, report)
        self.current_status = '巡点完成：%s' % out_path

    def start_execute(self):
        path = self.active_handeye_result_path()
        if path is None:
            messagebox.showerror('缺少模型', '选择手眼标定结果文件。')
            return
        if not self.save_exec_from_ui():
            return
        self.start_worker('转换并执行', self.execute_worker, path)

    def current_yolo_xyz_or_wait(self, timeout_sec=5.0):
        """Manual execute means snapshot the current visible chess immediately.

        Unlike a verification scan, this command is initiated by the user after
        they have positioned the chess.  It must not wait for three stable
        observations when a valid current RGB-D coordinate already exists.
        """
        current = self.node.recent_yolo_camera_xyz_mm(window_sec=0.5)
        if current is None:
            current = self.node.yolo_camera_xyz_mm(max_pixel_age_sec=2.0)
        if current is not None and np.all(np.isfinite(current)):
            return current.tolist()
        with self.node.lock:
            before = self.node.yolo_pixel_time
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            with self.node.lock:
                pixel_time = self.node.yolo_pixel_time
            if pixel_time > before:
                current = self.node.yolo_camera_xyz_mm(max_pixel_age_sec=2.0)
                if current is not None and np.all(np.isfinite(current)):
                    return current.tolist()
            time.sleep(0.05)
        return None

    def execute_worker(self, result_path):
        camera_mm = self.current_yolo_xyz_or_wait(5.0)
        if camera_mm is None:
            self.current_status = '执行取消：当前没有有效 YOLO 加深度象棋坐标，等待 5 秒后仍未获得新检测'
            return
        result = self.normalize_handeye_result(load_yaml(result_path, {}))
        route = self.exec_model.get()
        model, selected_z = self.choose_model_for_camera(result, route, camera_mm)
        target = self.apply_model(model, camera_mm)
        if target is None:
            self.current_status = '执行取消：所选模型无效'
            return
        target = np.asarray(target) + np.array([as_float(self.exec_x.get()), as_float(self.exec_y.get()), as_float(self.exec_z.get())])
        f, travel = as_float(self.exec_speed.get(), 80.0), -210.0
        self.node.send_gcode('G90')
        self.node.send_gcode('G1 Z%.2f F%.2f' % (travel, f))
        self.node.send_gcode('G1 X%.2f Y%.2f Z%.2f F%.2f' % (target[0], target[1], travel, f))
        self.node.send_gcode('G1 X%.2f Y%.2f Z%.2f F%.2f' % (target[0], target[1], target[2], f))
        suffix = '' if selected_z is None else '，选用 %.1f mm 层' % as_float(selected_z)
        self.current_status = '已执行 %s 模型%s：目标 %s mm' % (route, suffix, np.round(target, 2).tolist())

    def start_manual_move(self):
        if self.busy:
            self.current_status = '当前已有任务在运行。'
            return
        try:
            target = [float(self.manual_x.get()), float(self.manual_y.get()), float(self.manual_z.get())]
            speed = float(self.exec_speed.get())
        except ValueError:
            messagebox.showerror('输入无效', 'X、Y、Z 和机械臂速度都必须是数值。')
            return
        if speed <= 0.0:
            messagebox.showerror('速度无效', '机械臂速度必须大于 0 mm/s。')
            return
        if not self.save_exec_from_ui():
            return
        self.start_worker('手动移动', self.manual_move_worker, target, speed)

    def manual_move_worker(self, target, speed):
        x, y, z = target
        self.node.send_gcode('G90')
        self.node.send_gcode('G1 Z-210.00 F%.2f' % speed)
        self.node.send_gcode('G1 X%.2f Y%.2f Z-210.00 F%.2f' % (x, y, speed))
        self.node.send_gcode('G1 X%.2f Y%.2f Z%.2f F%.2f' % (x, y, z, speed))
        self.current_status = '已发送手动目标 Delta XYZ=%s mm，速度 %.1f mm/s' % (np.round(target, 2).tolist(), speed)

    def render_image(self, image, label, kind):
        if image is None:
            return
        overlay = image.copy()
        if kind == 'yolo':
            with self.node.lock:
                detections = list(self.node.yolo_detections)
                selected_index = self.node.yolo_selected_index
            for detection in detections:
                bbox = detection.get('bbox_xyxy', [])
                if len(bbox) != 4:
                    continue
                index = int(detection.get('index', -1))
                selected = index == selected_index
                color = (0, 255, 0) if selected else (0, 0, 255)
                x1, y1, x2, y2 = [int(round(value)) for value in bbox]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                center = detection.get('center_pixel', [])
                if len(center) == 2:
                    cv2.drawMarker(overlay, (int(round(center[0])), int(round(center[1]))), color, cv2.MARKER_CROSS, 16, 2)
                detection_label = 'CHESS %.2f%s' % (
                    float(detection.get('confidence', 0.0)),
                    ' SELECTED' if selected else '',
                )
                cv2.putText(overlay, detection_label, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            xyz = self.node.yolo_camera_xyz_mm()
            lines = []
            if xyz is None:
                lines.append('CAM XYZ: NO RGB-D')
            else:
                lines.append('CAM XYZ mm: [%.1f, %.1f, %.1f]' % tuple(xyz))
                result = self.preview_handeye_result()
                if result is not None:
                    pnp_model, _pnp_z = self.choose_model_for_camera(result, 'pnp', xyz)
                    depth_model, _depth_z = self.choose_model_for_camera(result, 'depth', xyz)
                    pnp_delta = self.apply_model(pnp_model, xyz)
                    depth_delta = self.apply_model(depth_model, xyz)
                    if pnp_delta is not None:
                        lines.append('PNP DELTA: [%.1f, %.1f, %.1f]' % tuple(pnp_delta))
                    if depth_delta is not None:
                        lines.append('DEPTH DELTA: [%.1f, %.1f, %.1f]' % tuple(depth_delta))
            panel_h = 12 + 22 * len(lines)
            cv2.rectangle(overlay, (8, 38), (440, 38 + panel_h), (15, 15, 15), -1)
            cv2.rectangle(overlay, (8, 38), (440, 38 + panel_h), (130, 130, 130), 1)
            for idx, line in enumerate(lines):
                cv2.putText(overlay, line, (16, 61 + 22 * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        elif kind == 'charuco':
            current = self.latest_charuco or {}
            if current.get('corner_count', 0) > 0:
                self.last_display_charuco = current
                self.last_display_charuco_time = time.monotonic()
            display = self.last_display_charuco if self.last_display_charuco is not None else current
            for marker in display.get('marker_pixel_corners', []):
                polygon = np.asarray(marker, dtype=np.int32).reshape(-1, 1, 2)
                if len(polygon) >= 4:
                    cv2.polylines(overlay, [polygon], True, (0, 190, 255), 2, cv2.LINE_AA)
            for index, pixel in enumerate(display.get('corner_pixel_uv', [])):
                if len(pixel) != 2:
                    continue
                point = (int(round(pixel[0])), int(round(pixel[1])))
                cv2.circle(overlay, point, 4, (0, 255, 0), -1, cv2.LINE_AA)
                if index < 30:
                    corner_ids = display.get('corner_ids', [])
                    corner_id = corner_ids[index] if index < len(corner_ids) else index
                    cv2.putText(overlay, str(corner_id), (point[0] + 5, point[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
            lines = [
                'MARKERS: %s  CORNERS: %s' % (display.get('marker_count', 0), display.get('corner_count', 0)),
            ]
            pnp = display.get('pnp', {})
            if pnp.get('mean_reprojection_error_px') is not None:
                lines.append('PNP ERR px: %.3f' % float(pnp['mean_reprojection_error_px']))
            else:
                lines.append('PNP ERR px: -')
            if self.progress.get('mode') == '采集':
                lines.append('CANDIDATES: %d/%d LEFT:%d' % (
                    int(self.progress.get('candidate_done', 0)), int(self.progress.get('candidate_total', 0)), int(self.progress.get('candidate_left', 0)),
                ))
                lines.append('VALID: %d/%d  TOTAL:%d' % (
                    int(self.progress.get('accepted_layer', 0)), int(self.progress.get('requested_layer', 0)), int(self.progress.get('accepted_total', 0)),
                ))
            panel_h = 12 + 21 * len(lines)
            cv2.rectangle(overlay, (8, 8), (390, 8 + panel_h), (15, 15, 15), -1)
            cv2.rectangle(overlay, (8, 8), (390, 8 + panel_h), (0, 255, 0), 1)
            for idx, line in enumerate(lines):
                cv2.putText(overlay, line, (16, 30 + 21 * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1, cv2.LINE_AA)
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        pil = PilImage.fromarray(rgb)
        max_w, max_h = (930, 420)
        scale = min(max_w / pil.width, max_h / pil.height, 1.0)
        if scale < 1.0:
            # Live video is refreshed at 60 Hz. Bilinear resizing is much
            # lighter than Lanczos here and avoids starving camera callbacks.
            resample = getattr(getattr(PilImage, 'Resampling', PilImage), 'BILINEAR')
            pil = pil.resize((int(pil.width * scale), int(pil.height * scale)), resample)
        photo_attr = 'charuco_photo' if kind == 'charuco' else 'yolo_photo'
        photo = getattr(self, photo_attr)
        if photo is None or photo.width() != pil.width or photo.height() != pil.height:
            photo = ImageTk.PhotoImage(pil)
            setattr(self, photo_attr, photo)
            label.configure(image=photo)
        else:
            # Reusing Tk's image storage avoids allocating 120 PhotoImage
            # objects per second for the two live previews.
            photo.paste(pil)

    def spin_ros(self):
        if rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.002)
            self.root.after(8, self.spin_ros)

    def update_charuco_preview_worker(self, settings):
        try:
            observation = self.charuco_observation(settings)
        except Exception as exc:
            observation = {'valid': False, 'reason': 'preview_error: %s' % exc}
        with self.charuco_preview_lock:
            self.latest_charuco = observation
            self.charuco_preview_busy = False

    def refresh_ui(self):
        # The displayed image itself uses the 60 Hz color stream. ChArUco runs
        # in a worker so its PnP/depth calculation never blocks Tk rendering.
        if (not self.charuco_preview_busy
                and time.monotonic() - self.last_live_charuco_time >= (1.0 / 15.0)):
            try:
                preview_settings = self.settings_from_ui()
            except Exception as exc:
                self.latest_charuco = {'valid': False, 'reason': 'preview_settings_error: %s' % exc}
            else:
                self.charuco_preview_busy = True
                self.last_live_charuco_time = time.monotonic()
                threading.Thread(
                    target=self.update_charuco_preview_worker,
                    args=(preview_settings,),
                    name='charuco-preview',
                    daemon=True,
                ).start()
        with self.charuco_preview_lock:
            charuco = self.latest_charuco
        with self.node.lock:
            # Both preview panels use the raw 60 Hz color image. Their OpenCV / YOLO
            # recognition results are drawn as the newest available overlay.
            color = None if self.node.color_image is None else self.node.color_image.copy()
        if charuco and color is not None:
            self.render_image(color, self.charuco_label, 'charuco')
            self.charuco_info_var.set('ChArUco：corner=%s marker=%s，%s' % (charuco.get('corner_count', 0), charuco.get('marker_count', 0), charuco.get('reason', '-')))
        elif charuco:
            self.charuco_info_var.set(
                'ChArUco：corner=%s marker=%s，%s'
                % (
                    charuco.get('corner_count', 0),
                    charuco.get('marker_count', 0),
                    charuco.get('reason', '-'),
                )
            )
        xyz = self.node.yolo_camera_xyz_mm()
        self.render_image(color, self.yolo_label, 'yolo')
        self.yolo_info_var.set('象棋相机坐标 mm：%s' % ('-' if xyz is None else np.round(xyz, 2).tolist()))
        preview_result = self.preview_handeye_result()
        if xyz is None or preview_result is None:
            self.yolo_delta_var.set('手眼转换 Delta 坐标：等待模型结果文件和象棋相机 XYZ。')
        else:
            pnp_model, pnp_z = self.choose_model_for_camera(preview_result, 'pnp', xyz)
            depth_model, depth_z = self.choose_model_for_camera(preview_result, 'depth', xyz)
            pnp_delta = self.apply_model(pnp_model, xyz)
            depth_delta = self.apply_model(depth_model, xyz)
            pnp_text = '-' if pnp_delta is None else '%s%s' % (np.round(pnp_delta, 2).tolist(), '' if pnp_z is None else ' (层 %.1f)' % pnp_z)
            depth_text = '-' if depth_delta is None else '%s%s' % (np.round(depth_delta, 2).tolist(), '' if depth_z is None else ' (层 %.1f)' % depth_z)
            self.yolo_delta_var.set('PNP 转换 Delta XYZ mm：%s\n深度转换 Delta XYZ mm：%s' % (pnp_text, depth_text))
        if self.latest_scan_offset is not None:
            self.scan_offset_var.set('当前成功巡检点 XYZ offset（预测 Delta - 真实 Delta，mm）：\nPNP: %s\n深度: %s' % (
                self.latest_scan_offset.get('PNP模型'), self.latest_scan_offset.get('深度模型')
            ))
        self.progress_var.set('进度：%s %d/%d | 层 %s | %s' % (self.progress['mode'], self.progress['current'], self.progress['total'], self.progress['layer'], self.progress['detail']))
        self.status_var.set(self.current_status)
        self.root.after(16, self.refresh_ui)

    def close(self):
        self.stop_requested = True
        self.root.destroy()


def main(args=None):
    rclpy.init(args=args)
    node = WorkbenchNode()
    root = tk.Tk()
    app = HandeyeWorkbench(root, node)
    try:
        root.mainloop()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
