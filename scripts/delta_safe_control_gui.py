#!/usr/bin/env python3
"""Tkinter GUI for safe Delta calibration control."""

import math
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import rclpy
import yaml
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, String


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


class DeltaGuiNode(Node):
    def __init__(self):
        super().__init__('delta_safe_control_gui')
        self.move_pub = self.create_publisher(Point, '/delta_arm/move_to', 10)
        self.raw_gcode_pub = self.create_publisher(String, '/delta_arm/gcode_raw', 10)
        self.motor_pub = self.create_publisher(Bool, '/delta_arm/motor_enable', 10)
        self.home_pub = self.create_publisher(Empty, '/delta_arm/home', 10)
        self.charuco_command_pub = self.create_publisher(String, '/delta_charuco/command', 10)


class DeltaSafeControlGui:
    def __init__(self, root, node):
        self.root = root
        self.node = node
        self.log_queue = queue.Queue()
        self.busy = False

        self.waypoint_path = Path('/home/wyy/gpt_dev_ws/calibration_targets/delta_safe_9_waypoints.yaml')
        self.discovery_dashboard_path = Path('/home/wyy/gpt_dev_ws/calibration_targets/delta_discovery_dashboard.txt')
        self.home_xyz = [0.0, 0.0, 0.0]
        self.current_xyz = list(self.home_xyz)
        self.safe_xy_z = -210.0
        self.min_z = -320.0
        self.max_z = 0.0
        self.min_x = -90.0
        self.max_x = 90.0
        self.min_y = -60.0
        self.max_y = 100.0
        self.polygon_margin = 8.0
        self.use_polygon = tk.BooleanVar(value=True)
        self.xy_step = tk.DoubleVar(value=5.0)
        self.z_step = tk.DoubleVar(value=5.0)
        self.hold_xy_step = tk.DoubleVar(value=5.0)
        self.hold_z_step = tk.DoubleVar(value=5.0)
        self.hold_feedrate = tk.DoubleVar(value=200.0)
        self.auto_delay = tk.DoubleVar(value=2.0)
        self.waypoints = []
        self.continuous_jog = None
        self.continuous_after_id = None
        self.hold_start_after_id = None
        self.hold_started = False
        self.pending_button_jog = None
        self.continuous_interval_ms = 450
        self.hold_start_delay_ms = 350
        self._pressed_keys = set()
        self.last_jog_block_log_time = 0.0
        self.safety_descend_active = False

        boundary = sort_boundary(BOUNDARY_POINTS_XY)
        self.safe_polygon = shrink_polygon(boundary, self.polygon_margin)

        self.root.title('Delta Safe Calibration Control')
        self.root.geometry('1180x760')
        self.root.minsize(1000, 680)
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)
        self.build_ui()
        self.bind_keys()
        self.load_waypoints(log_missing=False)
        self.refresh_all()
        self.root.after(50, self.spin_ros)
        self.root.after(100, self.flush_logs)
        self.root.after(1000, self.refresh_discovery_dashboard)
        self.log('GUI ready. Click this window before using keyboard shortcuts.')

    def build_ui(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill='both', expand=True)

        top = ttk.Frame(outer)
        top.pack(fill='x')

        pose = ttk.LabelFrame(top, text='Current Pose / 当前坐标', padding=8)
        pose.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self.pose_var = tk.StringVar()
        self.status_var = tk.StringVar(value='IDLE')
        ttk.Label(pose, textvariable=self.pose_var, font=('Sans', 14, 'bold')).pack(anchor='w')
        ttk.Label(pose, textvariable=self.status_var).pack(anchor='w', pady=(4, 0))

        motor = ttk.LabelFrame(top, text='Motor / 电机', padding=8)
        motor.pack(side='left', fill='y')
        ttk.Button(motor, text='Enable (E)', command=lambda: self.publish_motor(True)).grid(row=0, column=0, padx=3, pady=3)
        ttk.Button(motor, text='Disable (D)', command=lambda: self.publish_motor(False)).grid(row=0, column=1, padx=3, pady=3)
        ttk.Button(motor, text='G28 Home (0)', command=self.publish_home).grid(row=1, column=0, columnspan=2, sticky='ew', padx=3, pady=3)
        ttk.Button(motor, text='Set Current Zero', command=self.set_current_zero).grid(row=2, column=0, columnspan=2, sticky='ew', padx=3, pady=3)

        auto_top = ttk.LabelFrame(top, text='Auto Sampling / 自动踩点', padding=8)
        auto_top.pack(side='left', fill='y', padx=(8, 0))
        ttk.Button(
            auto_top,
            text='Start 25 Samples (Z=-230, C>=14)',
            command=lambda: self.publish_charuco_command('discover_calibrate'),
        ).grid(row=0, column=0, columnspan=2, sticky='ew', padx=3, pady=3)
        ttk.Button(
            auto_top,
            text='BAD + STOP',
            command=lambda: self.publish_charuco_command('mark_bad_stop'),
        ).grid(row=1, column=0, sticky='ew', padx=3, pady=3)
        ttk.Button(
            auto_top,
            text='Stop Auto',
            command=lambda: self.publish_charuco_command('stop'),
        ).grid(row=1, column=1, sticky='ew', padx=3, pady=3)
        auto_top.columnconfigure(0, weight=1)
        auto_top.columnconfigure(1, weight=1)

        rotation_top = ttk.LabelFrame(top, text='Board Rotation / 旋转轴采样', padding=8)
        rotation_top.pack(side='left', fill='y', padx=(8, 0))
        ttk.Button(
            rotation_top,
            text='Save Rotation Pose',
            command=lambda: self.publish_charuco_command('save_rotation'),
        ).grid(row=0, column=0, sticky='ew', padx=3, pady=3)
        ttk.Button(
            rotation_top,
            text='Write Rotation File',
            command=lambda: self.publish_charuco_command('write_rotation'),
        ).grid(row=1, column=0, sticky='ew', padx=3, pady=3)
        rotation_top.columnconfigure(0, weight=1)

        mid = ttk.Frame(outer)
        mid.pack(fill='both', expand=True, pady=8)

        controls = ttk.LabelFrame(mid, text='Jog Control / 点动控制', padding=8)
        controls.pack(side='left', fill='y', padx=(0, 8))

        self.make_jog_button(controls, 'Y+ / I', dx=0.0, dy=1.0, dz=0.0, row=0, column=1)
        self.make_jog_button(controls, 'X- / J', dx=-1.0, dy=0.0, dz=0.0, row=1, column=0)
        ttk.Button(controls, text='HOLD', width=12, command=self.hold).grid(row=1, column=1, padx=4, pady=4)
        self.make_jog_button(controls, 'X+ / L', dx=1.0, dy=0.0, dz=0.0, row=1, column=2)
        self.make_jog_button(controls, 'Y- / K', dx=0.0, dy=-1.0, dz=0.0, row=2, column=1)

        self.make_jog_button(controls, 'Z Up / U', dx=0.0, dy=0.0, dz=1.0, row=3, column=0, pady=(16, 4))
        self.make_jog_button(controls, 'Z Down / O', dx=0.0, dy=0.0, dz=-1.0, row=3, column=2, pady=(16, 4))

        step_frame = ttk.LabelFrame(controls, text='Step / 步长', padding=6)
        step_frame.grid(row=4, column=0, columnspan=3, sticky='ew', pady=(12, 4))
        ttk.Label(step_frame, text='XY mm').grid(row=0, column=0, sticky='w')
        ttk.Spinbox(step_frame, from_=1.0, to=30.0, increment=1.0, textvariable=self.xy_step, width=8).grid(row=0, column=1, padx=5)
        ttk.Label(step_frame, text='Z mm').grid(row=1, column=0, sticky='w')
        ttk.Spinbox(step_frame, from_=1.0, to=30.0, increment=1.0, textvariable=self.z_step, width=8).grid(row=1, column=1, padx=5)
        ttk.Label(step_frame, text='Hold XY mm').grid(row=2, column=0, sticky='w')
        ttk.Spinbox(step_frame, from_=1.0, to=20.0, increment=1.0, textvariable=self.hold_xy_step, width=8).grid(row=2, column=1, padx=5)
        ttk.Label(step_frame, text='Hold Z mm').grid(row=3, column=0, sticky='w')
        ttk.Spinbox(step_frame, from_=1.0, to=20.0, increment=1.0, textvariable=self.hold_z_step, width=8).grid(row=3, column=1, padx=5)
        ttk.Label(step_frame, text='Hold F mm/s').grid(row=4, column=0, sticky='w')
        ttk.Spinbox(step_frame, from_=80.0, to=1200.0, increment=20.0, textvariable=self.hold_feedrate, width=8).grid(row=4, column=1, padx=5)
        ttk.Label(step_frame, text='Auto delay s').grid(row=5, column=0, sticky='w')
        ttk.Spinbox(step_frame, from_=0.5, to=10.0, increment=0.5, textvariable=self.auto_delay, width=8).grid(row=5, column=1, padx=5)

        limits = ttk.LabelFrame(controls, text='Safety / 安全限制', padding=6)
        limits.grid(row=5, column=0, columnspan=3, sticky='ew', pady=4)
        self.limits_var = tk.StringVar()
        ttk.Label(limits, textvariable=self.limits_var, justify='left').pack(anchor='w')
        ttk.Checkbutton(limits, text='Use measured polygon', variable=self.use_polygon).pack(anchor='w')

        way_frame = ttk.LabelFrame(mid, text='Manual Waypoints / 手动点位（备用）', padding=8)
        way_frame.pack(side='left', fill='both', expand=True)
        columns = ('idx', 'role', 'x', 'y', 'z')
        self.tree = ttk.Treeview(way_frame, columns=columns, show='headings', height=8)
        for col, text, width in [
            ('idx', '#', 40),
            ('role', 'Role', 110),
            ('x', 'X mm', 90),
            ('y', 'Y mm', 90),
            ('z', 'Z mm', 90),
        ]:
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor='center')
        self.tree.pack(fill='both', expand=True)

        wp_buttons = ttk.Frame(way_frame)
        wp_buttons.pack(fill='x', pady=(8, 0))
        ttk.Button(wp_buttons, text='Save Current (P)', command=self.save_current_waypoint).grid(row=0, column=0, padx=3, pady=3)
        ttk.Button(wp_buttons, text='Write YAML', command=self.write_waypoints).grid(row=0, column=1, padx=3, pady=3)
        ttk.Button(wp_buttons, text='Reload YAML', command=lambda: self.load_waypoints(log_missing=True)).grid(row=0, column=2, padx=3, pady=3)
        ttk.Button(wp_buttons, text='Clear', command=self.clear_waypoints).grid(row=0, column=3, padx=3, pady=3)
        ttk.Button(wp_buttons, text='Move Selected', command=self.move_selected).grid(row=1, column=0, padx=3, pady=3)
        ttk.Button(wp_buttons, text='Run 1-8 (A)', command=self.run_translation_waypoints).grid(row=1, column=1, padx=3, pady=3)
        ttk.Button(wp_buttons, text='Rotation Point 9 (R)', command=self.move_to_rotation_hold).grid(row=1, column=2, padx=3, pady=3)

        num_frame = ttk.Frame(way_frame)
        num_frame.pack(fill='x', pady=(8, 0))
        for i in range(1, 10):
            ttk.Button(num_frame, text=str(i), width=4, command=lambda idx=i: self.move_to_waypoint(idx)).pack(side='left', padx=2)

        dashboard_frame = ttk.LabelFrame(outer, text='Discovery Progress / 自动找点进度', padding=6)
        dashboard_frame.pack(fill='both', expand=False, pady=(0, 8))
        self.dashboard_text = tk.Text(dashboard_frame, height=8, wrap='none')
        self.dashboard_text.pack(fill='both', expand=True)

        log_frame = ttk.LabelFrame(outer, text='Log / 日志', padding=6)
        log_frame.pack(fill='both', expand=False)
        self.log_text = tk.Text(log_frame, height=9, wrap='word')
        self.log_text.pack(fill='both', expand=True)

    def bind_keys(self):
        bindings = {
            'e': lambda _e: self.publish_motor(True),
            'd': lambda _e: self.publish_motor(False),
            '0': lambda _e: self.publish_home(),
            'p': lambda _e: self.save_current_waypoint(),
            'a': lambda _e: self.run_translation_waypoints(),
            'r': lambda _e: self.move_to_rotation_hold(),
        }
        for key, callback in bindings.items():
            self.root.bind(key, callback)
        key_jogs = [
            ('i', 'i', (0.0, 1.0, 0.0)),
            ('k', 'k', (0.0, -1.0, 0.0)),
            ('j', 'j', (-1.0, 0.0, 0.0)),
            ('l', 'l', (1.0, 0.0, 0.0)),
            ('u', 'u', (0.0, 0.0, 1.0)),
            ('o', 'o', (0.0, 0.0, -1.0)),
            ('Up', 'Up', (0.0, 1.0, 0.0)),
            ('Down', 'Down', (0.0, -1.0, 0.0)),
            ('Left', 'Left', (-1.0, 0.0, 0.0)),
            ('Right', 'Right', (1.0, 0.0, 0.0)),
            ('Prior', 'PageUp', (0.0, 0.0, 1.0)),
            ('Next', 'PageDown', (0.0, 0.0, -1.0)),
        ]
        for tk_key, key_id, vector in key_jogs:
            self.root.bind('<KeyPress-%s>' % tk_key, lambda _e, v=vector, k=key_id: self.on_jog_key_press(k, *v))
            self.root.bind('<KeyRelease-%s>' % tk_key, lambda _e, k=key_id: self.on_jog_key_release(k))
        for i in range(1, 10):
            self.root.bind(str(i), lambda _e, idx=i: self.move_to_waypoint(idx))

    def make_jog_button(self, parent, text, dx, dy, dz, row, column, pady=4):
        button = ttk.Button(parent, text=text, width=12)
        button.grid(row=row, column=column, padx=4, pady=pady)
        button.bind('<ButtonPress-1>', lambda _e: self.on_jog_button_press(dx, dy, dz))
        button.bind('<ButtonRelease-1>', lambda _e: self.on_jog_button_release())
        button.bind('<Leave>', lambda _e: self.on_jog_button_release(cancel_click=True))
        return button

    def on_jog_key_press(self, key_id, dx, dy, dz):
        if key_id in self._pressed_keys:
            return
        self._pressed_keys.add(key_id)
        self.jog_step(dx, dy, dz)

    def on_jog_key_release(self, key_id):
        self._pressed_keys.discard(key_id)

    def on_jog_button_press(self, dx, dy, dz):
        if self.busy:
            self.log('busy: jog ignored')
            return
        self.pending_button_jog = (float(dx), float(dy), float(dz))
        self.hold_started = False
        if self.hold_start_after_id is not None:
            try:
                self.root.after_cancel(self.hold_start_after_id)
            except tk.TclError:
                pass
        self.hold_start_after_id = self.root.after(
            self.hold_start_delay_ms,
            lambda: self.start_continuous_jog(dx, dy, dz),
        )

    def on_jog_button_release(self, cancel_click=False):
        pending = self.pending_button_jog
        self.pending_button_jog = None
        if self.hold_start_after_id is not None:
            try:
                self.root.after_cancel(self.hold_start_after_id)
            except tk.TclError:
                pass
            self.hold_start_after_id = None
        if self.hold_started:
            self.stop_continuous_jog()
        elif pending is not None and not cancel_click:
            self.jog_step(*pending)

    def start_continuous_jog(self, dx, dy, dz):
        self.hold_start_after_id = None
        if self.busy:
            self.log('busy: jog ignored')
            return
        self.hold_started = True
        self.continuous_jog = (float(dx), float(dy), float(dz))
        if self.continuous_after_id is None:
            self.publish_raw_gcode('G90')
            self.status_var.set('JOGGING')
            self.continuous_jog_tick()

    def stop_continuous_jog(self):
        self.continuous_jog = None
        self.hold_started = False
        if self.hold_start_after_id is not None:
            try:
                self.root.after_cancel(self.hold_start_after_id)
            except tk.TclError:
                pass
            self.hold_start_after_id = None
        if self.continuous_after_id is not None:
            try:
                self.root.after_cancel(self.continuous_after_id)
            except tk.TclError:
                pass
            self.continuous_after_id = None
        if self.safety_descend_active:
            self.status_var.set('DESCENDING TO SAFE Z')
        elif not self.busy:
            self.status_var.set('IDLE')

    def continuous_jog_tick(self):
        self.continuous_after_id = None
        if self.continuous_jog is None:
            return
        if self.busy:
            self.stop_continuous_jog()
            return

        dx_dir, dy_dir, dz_dir = self.continuous_jog
        dx = dx_dir * max(0.0, float(self.hold_xy_step.get()))
        dy = dy_dir * max(0.0, float(self.hold_xy_step.get()))
        dz = dz_dir * max(0.0, float(self.hold_z_step.get()))
        if not self.jog_direct(dx=dx, dy=dy, dz=dz):
            self.stop_continuous_jog()
            return
        self.continuous_after_id = self.root.after(self.continuous_interval_ms, self.continuous_jog_tick)

    def jog_step(self, dx_dir, dy_dir, dz_dir):
        dx = float(dx_dir) * max(0.0, float(self.xy_step.get()))
        dy = float(dy_dir) * max(0.0, float(self.xy_step.get()))
        dz = float(dz_dir) * max(0.0, float(self.z_step.get()))
        self.jog_direct(dx=dx, dy=dy, dz=dz)

    def spin_ros(self):
        if rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.0)
            self.root.after(50, self.spin_ros)

    def flush_logs(self):
        while True:
            try:
                text = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.insert('end', text + '\n')
            self.log_text.see('end')
        self.root.after(100, self.flush_logs)

    def refresh_discovery_dashboard(self):
        try:
            if self.discovery_dashboard_path.exists():
                text = self.discovery_dashboard_path.read_text(encoding='utf-8', errors='replace')
            else:
                text = 'No discovery progress yet.'
            self.dashboard_text.delete('1.0', 'end')
            self.dashboard_text.insert('1.0', text)
        except Exception as exc:
            self.dashboard_text.delete('1.0', 'end')
            self.dashboard_text.insert('1.0', 'dashboard read failed: %s' % exc)
        self.root.after(1000, self.refresh_discovery_dashboard)

    def log(self, text):
        self.log_queue.put(text)
        try:
            self.node.get_logger().info(text)
        except Exception:
            pass

    def set_busy(self, busy, text=None):
        self.busy = busy
        if text:
            self.status_var.set(text)
        else:
            self.status_var.set('BUSY' if busy else 'IDLE')

    def run_threaded(self, func):
        if self.busy:
            self.log('busy: command ignored')
            return
        threading.Thread(target=func, daemon=True).start()

    @staticmethod
    def round_xyz(xyz):
        return [round(float(v), 1) for v in xyz]

    def refresh_all(self):
        self.pose_var.set('X %.1f   Y %.1f   Z %.1f mm' % tuple(self.current_xyz))
        self.limits_var.set(
            'XY box: X[%.0f, %.0f] Y[%.0f, %.0f]\nZ motion: [%.0f, %.0f]\nFirst jog descends to safe Z %.0f\nWaypoint file: %s'
            % (
                self.min_x,
                self.max_x,
                self.min_y,
                self.max_y,
                self.min_z,
                self.max_z,
                self.safe_xy_z,
                self.waypoint_path,
            )
        )
        self.refresh_waypoints()

    def refresh_waypoints(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for idx, wp in enumerate(self.waypoints, start=1):
            role = 'translation' if idx <= 8 else 'rotation'
            x, y, z = wp['xyz']
            self.tree.insert('', 'end', iid=str(idx), values=(idx, role, '%.1f' % x, '%.1f' % y, '%.1f' % z))

    def publish_motor(self, enabled):
        msg = Bool()
        msg.data = bool(enabled)
        self.node.motor_pub.publish(msg)
        self.log('motor %s' % ('ENABLE' if enabled else 'DISABLE'))

    def publish_home(self):
        def work():
            self.set_busy(True, 'HOMING')
            self.node.home_pub.publish(Empty())
            self.current_xyz = list(self.home_xyz)
            self.log('G28 home sent; current pose reset to %s' % self.round_xyz(self.current_xyz))
            self.root.after(0, self.refresh_all)
            time.sleep(4.0)
            self.set_busy(False, 'IDLE')
        self.run_threaded(work)

    def set_current_zero(self):
        self.current_xyz = list(self.home_xyz)
        self.log('current pose manually reset to %s; no motion command sent' % self.round_xyz(self.current_xyz))
        self.refresh_all()

    def hold(self):
        self.publish_motor(True)
        self.publish_move_direct(self.current_xyz)
        self.log('holding current pose %s' % self.round_xyz(self.current_xyz))

    def validate_target(self, xyz, show_popup=False, check_xy=True):
        x, y, z = [float(v) for v in xyz]
        reason = None
        if check_xy and not (self.min_x <= x <= self.max_x):
            reason = 'X %.1f outside [%.1f, %.1f]' % (x, self.min_x, self.max_x)
        elif check_xy and not (self.min_y <= y <= self.max_y):
            reason = 'Y %.1f outside [%.1f, %.1f]' % (y, self.min_y, self.max_y)
        elif not (self.min_z <= z <= self.max_z):
            reason = 'Z %.1f outside [%.1f, %.1f]' % (z, self.min_z, self.max_z)
        elif check_xy and self.use_polygon.get() and not point_in_polygon(x, y, self.safe_polygon):
            reason = 'XY (%.1f, %.1f) outside measured safe polygon' % (x, y)
        if reason:
            self.log('BLOCKED: ' + reason)
            if show_popup:
                messagebox.showwarning('Move blocked', reason)
            return False
        return True

    def publish_move_direct(self, xyz, log_move=True, feedrate=None):
        x, y, z = [float(v) for v in xyz]
        if feedrate is None:
            msg = Point()
            msg.x, msg.y, msg.z = x, y, z
            self.node.move_pub.publish(msg)
        else:
            self.publish_raw_gcode('G1 X%.2f Y%.2f Z%.2f F%.2f' % (x, y, z, float(feedrate)))
        self.current_xyz = [x, y, z]
        if log_move:
            self.log('move -> x=%.1f y=%.1f z=%.1f' % (x, y, z))
        self.root.after(0, self.refresh_all)

    def descend_to_safe_height(self):
        if self.current_xyz[2] <= self.safe_xy_z:
            return False
        target = [self.current_xyz[0], self.current_xyz[1], self.safe_xy_z]
        if not self.validate_target(target, show_popup=False, check_xy=False):
            return False
        self.log('first jog: descending directly to safe Z %.1f' % self.safe_xy_z)
        feedrate = max(1.0, float(self.hold_feedrate.get()))
        travel_mm = abs(self.current_xyz[2] - self.safe_xy_z)
        self.publish_move_direct(target, log_move=False, feedrate=feedrate)
        self.safety_descend_active = True
        self.status_var.set('DESCENDING TO SAFE Z')
        wait_ms = int(max(500.0, min(15000.0, travel_mm / feedrate * 1000.0 + 300.0)))
        self.root.after(wait_ms, self.finish_safety_descend)
        return True

    def finish_safety_descend(self):
        self.safety_descend_active = False
        if not self.busy:
            self.status_var.set('IDLE')

    def publish_raw_gcode(self, text):
        msg = String()
        msg.data = text
        self.node.raw_gcode_pub.publish(msg)

    def publish_charuco_command(self, command):
        msg = String()
        msg.data = str(command)
        self.node.charuco_command_pub.publish(msg)
        self.log('charuco command -> %s' % command)

    def move_to_blocking(self, xyz):
        target = [float(v) for v in xyz]
        current = list(self.current_xyz)
        xy_changes = math.hypot(target[0] - current[0], target[1] - current[1]) > 1e-6
        if current[2] > self.safe_xy_z:
            return self.descend_to_safe_height()
        if xy_changes and target[2] > self.safe_xy_z:
            reason = 'XY target blocked: target Z %.1f is above safe XY height %.1f' % (target[2], self.safe_xy_z)
            self.log(reason)
            messagebox.showwarning('Move blocked', reason)
            return False
        if not self.validate_target(target, show_popup=True):
            return False
        self.publish_move_direct(target)
        time.sleep(1.0)
        return True

    def move_to(self, xyz):
        def work():
            self.set_busy(True, 'MOVING')
            self.move_to_blocking(xyz)
            self.set_busy(False, 'IDLE')
        self.run_threaded(work)

    def jog(self, dx=0.0, dy=0.0, dz=0.0):
        target = [self.current_xyz[0] + dx, self.current_xyz[1] + dy, self.current_xyz[2] + dz]
        if dz > 0.0 and target[2] > self.max_z:
            target[2] = self.max_z
            self.log('Z up capped at %.1f' % self.max_z)
        elif dz < 0.0 and target[2] < self.min_z:
            target[2] = self.min_z
            self.log('Z down capped at %.1f' % self.min_z)
        self.move_to(target)

    def jog_direct(self, dx=0.0, dy=0.0, dz=0.0):
        if self.safety_descend_active:
            return True
        current = list(self.current_xyz)
        target = [current[0] + dx, current[1] + dy, current[2] + dz]
        xy_changes = math.hypot(dx, dy) > 1e-9

        if current[2] > self.safe_xy_z:
            if self.descend_to_safe_height():
                self.stop_continuous_jog()
                return True
            return False

        if dz > 0.0 and target[2] > self.max_z:
            target[2] = self.max_z
        elif dz < 0.0 and target[2] < self.min_z:
            target[2] = self.min_z

        if xy_changes and target[2] > self.safe_xy_z:
            self.log_jog_block_once('XY blocked: target Z must stay <= %.1f' % self.safe_xy_z)
            return False

        if target == current:
            return True
        if not self.validate_target(target, show_popup=False, check_xy=xy_changes):
            return False
        self.publish_move_direct(target, log_move=False, feedrate=max(1.0, float(self.hold_feedrate.get())))
        return True

    def log_jog_block_once(self, text):
        now = time.monotonic()
        if now - self.last_jog_block_log_time > 1.0:
            self.last_jog_block_log_time = now
            self.log(text)

    def save_current_waypoint(self):
        if not self.validate_target(self.current_xyz, show_popup=True):
            return
        if self.current_xyz[2] > self.safe_xy_z:
            messagebox.showwarning(
                'Waypoint blocked',
                'Calibration waypoints must be at or below safe XY height %.1f mm.' % self.safe_xy_z,
            )
            self.log('BLOCKED: waypoint Z %.1f above safe XY height %.1f' % (self.current_xyz[2], self.safe_xy_z))
            return
        if len(self.waypoints) >= 9:
            messagebox.showwarning('Waypoints full', 'Already have 9 waypoints. Clear or edit YAML first.')
            return
        name = 'pt_%02d' % (len(self.waypoints) + 1)
        self.waypoints.append({'name': name, 'xyz': list(self.current_xyz)})
        self.log('saved %s = %s' % (name, self.round_xyz(self.current_xyz)))
        self.refresh_waypoints()

    def write_waypoints(self):
        self.waypoint_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'description': 'Nine manually selected safe Delta hand-eye calibration waypoints. Points 1-8 are translation samples; point 9 is manual rotation hold.',
            'safety': {
                'safe_xy_z_mm': self.safe_xy_z,
                'min_z_mm': self.min_z,
                'max_z_mm': self.max_z,
                'box': {'x_mm': [self.min_x, self.max_x], 'y_mm': [self.min_y, self.max_y]},
                'use_polygon_workspace': bool(self.use_polygon.get()),
                'polygon_margin_mm': self.polygon_margin,
            },
            'waypoints': [
                {'name': wp['name'], 'x_mm': wp['xyz'][0], 'y_mm': wp['xyz'][1], 'z_mm': wp['xyz'][2]}
                for wp in self.waypoints
            ],
        }
        with self.waypoint_path.open('w', encoding='utf-8') as file:
            yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
        self.log('wrote %d waypoints to %s' % (len(self.waypoints), self.waypoint_path))

    def load_waypoints(self, log_missing=True):
        if not self.waypoint_path.exists():
            if log_missing:
                self.log('waypoint file not found: %s' % self.waypoint_path)
            return
        data = yaml.safe_load(self.waypoint_path.read_text(encoding='utf-8')) or {}
        loaded = []
        for idx, item in enumerate(data.get('waypoints', []), start=1):
            xyz = [float(item['x_mm']), float(item['y_mm']), float(item['z_mm'])]
            if xyz[2] > self.safe_xy_z:
                self.log('skipped waypoint #%d above safe XY height: %s' % (idx, self.round_xyz(xyz)))
                continue
            if self.validate_target(xyz):
                loaded.append({'name': str(item.get('name', 'pt_%02d' % idx)), 'xyz': xyz})
            else:
                self.log('skipped unsafe waypoint #%d %s' % (idx, self.round_xyz(xyz)))
        self.waypoints = loaded[:9]
        self.log('loaded %d waypoints from %s' % (len(self.waypoints), self.waypoint_path))
        self.refresh_waypoints()

    def clear_waypoints(self):
        if messagebox.askyesno('Clear waypoints', 'Clear in-memory waypoints? YAML file is not deleted.'):
            self.waypoints = []
            self.refresh_waypoints()
            self.log('cleared in-memory waypoints')

    def selected_index(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning('No selection', 'Select a waypoint row first.')
            return None
        return int(selected[0])

    def move_selected(self):
        idx = self.selected_index()
        if idx is not None:
            self.move_to_waypoint(idx)

    def move_to_waypoint(self, idx):
        if idx < 1 or idx > len(self.waypoints):
            self.log('waypoint %d unavailable; saved count=%d' % (idx, len(self.waypoints)))
            return
        wp = self.waypoints[idx - 1]
        self.log('moving to waypoint %d %s' % (idx, self.round_xyz(wp['xyz'])))
        self.move_to(wp['xyz'])

    def run_translation_waypoints(self):
        if len(self.waypoints) < 8:
            messagebox.showwarning('Not enough points', 'Need at least 8 waypoints for translation calibration.')
            return

        def work():
            self.set_busy(True, 'AUTO 1-8')
            for idx in range(1, 9):
                if not rclpy.ok():
                    break
                wp = self.waypoints[idx - 1]
                self.log('auto waypoint %d/8' % idx)
                if not self.move_to_blocking(wp['xyz']):
                    break
                time.sleep(max(0.0, float(self.auto_delay.get())))
            self.set_busy(False, 'IDLE')
        self.run_threaded(work)

    def move_to_rotation_hold(self):
        if len(self.waypoints) < 9:
            messagebox.showwarning('No point 9', 'Need waypoint 9 for manual rotation hold.')
            return

        def work():
            self.set_busy(True, 'ROTATION HOLD')
            if self.move_to_blocking(self.waypoints[8]['xyz']):
                self.publish_motor(True)
                self.log('holding waypoint 9 for manual board rotation')
            self.set_busy(False, 'IDLE')
        self.run_threaded(work)

    def on_close(self):
        if self.busy and not messagebox.askyesno('Busy', 'A move is running. Close anyway?'):
            return
        self.stop_continuous_jog()
        self.root.destroy()


def main(args=None):
    rclpy.init(args=args)
    node = DeltaGuiNode()
    root = tk.Tk()
    DeltaSafeControlGui(root, node)
    try:
        root.mainloop()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
