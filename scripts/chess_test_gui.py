#!/usr/bin/env python3
"""单层象棋手眼测试 Tkinter 小程序。"""

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from PIL import Image as PilImage
from PIL import ImageTk
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, String


class ChessTestGuiNode(Node):
    def __init__(self):
        super().__init__('chess_test_gui')
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=1)
        self.latest_selected_pixel = None
        self.latest_camera_point = None
        self.latest_delta_target = None
        self.latest_move_status = '空闲'
        self.latest_detection_time = 0.0

        self.home_pub = self.create_publisher(Empty, '/delta_arm/home', 10)
        self.command_pub = self.create_publisher(String, '/chess/handeye_command', 10)

        self.create_subscription(Image, '/chess/detection_image', self.on_image, 10)
        self.create_subscription(PointStamped, '/chess/selected_pixel_center', self.on_selected_pixel, 10)
        self.create_subscription(PointStamped, '/chess/camera_point', self.on_camera_point, 10)
        self.create_subscription(PointStamped, '/chess/delta_target', self.on_delta_target, 10)
        self.create_subscription(String, '/chess/move_status', self.on_move_status, 10)

    def on_image(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.image_queue.full():
                try:
                    self.image_queue.get_nowait()
                except queue.Empty:
                    pass
            self.image_queue.put_nowait(frame)
            self.latest_detection_time = time.time()
        except Exception as exc:
            self.get_logger().warning(f'解码 /chess/detection_image 失败: {exc}')

    def on_selected_pixel(self, msg):
        self.latest_selected_pixel = (
            float(msg.point.x),
            float(msg.point.y),
        )

    def on_camera_point(self, msg):
        self.latest_camera_point = (
            float(msg.point.x),
            float(msg.point.y),
            float(msg.point.z),
        )

    def on_delta_target(self, msg):
        self.latest_delta_target = (
            float(msg.point.x),
            float(msg.point.y),
            float(msg.point.z),
        )

    def on_move_status(self, msg):
        self.latest_move_status = str(msg.data)


class ChessTestGui:
    def __init__(self, root, node):
        self.root = root
        self.node = node
        self.current_photo = None
        self.busy = False
        self.home_settle_sec = 6.0

        self.root.title('象棋手眼测试')
        self.root.geometry('1140x860')
        self.root.minsize(980, 760)
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

        self.image_label = None
        self.status_var = tk.StringVar(value='等待相机画面...')
        self.pixel_var = tk.StringVar(value='像素坐标 u,v: -')
        self.camera_var = tk.StringVar(value='相机坐标 X,Y,Z（毫米）: -')
        self.delta_var = tk.StringVar(value='机械臂目标 X,Y,Z（毫米）: -')
        self.move_var = tk.StringVar(value='运动状态: 空闲')
        self.age_var = tk.StringVar(value='画面延迟: -')

        self.build_ui()
        self.root.after(10, self.spin_ros)
        self.root.after(16, self.refresh_ui)

    def build_ui(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill='both', expand=True)

        image_frame = ttk.LabelFrame(outer, text='象棋识别画面', padding=8)
        image_frame.pack(fill='both', expand=True)
        self.image_label = ttk.Label(image_frame, anchor='center')
        self.image_label.pack(fill='both', expand=True)

        info_frame = ttk.LabelFrame(outer, text='目标信息', padding=8)
        info_frame.pack(fill='x', pady=(10, 0))
        ttk.Label(info_frame, textvariable=self.status_var, font=('Sans', 11, 'bold')).pack(anchor='w')
        ttk.Label(info_frame, textvariable=self.pixel_var).pack(anchor='w', pady=(6, 0))
        ttk.Label(info_frame, textvariable=self.camera_var).pack(anchor='w')
        ttk.Label(info_frame, textvariable=self.delta_var).pack(anchor='w')
        ttk.Label(info_frame, text='画面内只显示相机坐标；像素坐标和机械臂坐标显示在这里。').pack(anchor='w')
        ttk.Label(info_frame, textvariable=self.move_var).pack(anchor='w', pady=(6, 0))
        ttk.Label(info_frame, textvariable=self.age_var).pack(anchor='w')

        button_frame = ttk.Frame(outer)
        button_frame.pack(fill='x', pady=(10, 0))
        ttk.Button(button_frame, text='刷新当前目标', command=self.capture_target).pack(side='left', padx=(0, 8))
        ttk.Button(button_frame, text='直接移动到当前象棋', command=self.go_only).pack(side='left', padx=(0, 8))
        ttk.Button(button_frame, text='回零并移动到当前象棋', command=self.home_and_go).pack(side='left', padx=(0, 8))
        ttk.Button(button_frame, text='仅回零', command=self.home_only).pack(side='left')

    def spin_ros(self):
        if rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.005)
            self.root.after(10, self.spin_ros)

    def refresh_ui(self):
        try:
            frame = self.node.image_queue.get_nowait()
        except queue.Empty:
            frame = None

        if frame is not None:
            rgb = frame[:, :, ::-1]
            image = PilImage.fromarray(rgb)
            width, height = image.size
            max_width = 1080
            max_height = 620
            scale = min(max_width / width, max_height / height, 1.0)
            if scale < 1.0:
                image = image.resize((int(width * scale), int(height * scale)), PilImage.Resampling.LANCZOS)
            self.current_photo = ImageTk.PhotoImage(image=image)
            self.image_label.configure(image=self.current_photo)
            self.status_var.set('实时识别画面。绿色为当前选中象棋。')

        if self.node.latest_selected_pixel is not None:
            self.pixel_var.set(
                '像素坐标 u,v: [%.1f, %.1f]'
                % self.node.latest_selected_pixel
            )
        else:
            self.pixel_var.set('像素坐标 u,v: -')

        if self.node.latest_camera_point is not None:
            camera_point_mm = [v * 1000.0 for v in self.node.latest_camera_point]
            self.camera_var.set(
                '相机坐标 X,Y,Z（毫米）: %s'
                % ([round(v, 1) for v in camera_point_mm],)
            )
        else:
            self.camera_var.set('相机坐标 X,Y,Z（毫米）: -')

        if self.node.latest_delta_target is not None:
            self.delta_var.set(
                '机械臂目标 X,Y,Z（毫米）: %s'
                % ([round(v, 1) for v in self.node.latest_delta_target],)
            )
        else:
            self.delta_var.set('机械臂目标 X,Y,Z（毫米）: -')

        self.move_var.set('运动状态: %s' % self.node.latest_move_status)
        if self.node.latest_detection_time > 0.0:
            age = max(0.0, time.time() - self.node.latest_detection_time)
            self.age_var.set('画面延迟: %.1f 秒' % age)
        else:
            self.age_var.set('画面延迟: -')

        self.root.after(16, self.refresh_ui)

    def publish_command(self, text):
        msg = String()
        msg.data = str(text)
        self.node.command_pub.publish(msg)

    def capture_target(self):
        self.publish_command('capture')
        self.status_var.set('已请求刷新当前象棋目标...')

    def home_only(self):
        self.node.home_pub.publish(Empty())
        self.status_var.set('已请求回零，等待 %.1f 秒...' % self.home_settle_sec)

    def go_only(self):
        self.status_var.set('开始移动到当前象棋目标...')
        self.publish_command('go')

    def home_and_go(self):
        if self.busy:
            self.status_var.set('回零并移动任务正在执行。')
            return

        def worker():
            self.busy = True
            try:
                self.status_var.set('机械臂回零中...')
                self.node.home_pub.publish(Empty())
                time.sleep(self.home_settle_sec)
                self.status_var.set('开始移动到当前象棋目标...')
                self.publish_command('go')
            finally:
                self.busy = False

        threading.Thread(target=worker, daemon=True).start()

    def on_close(self):
        self.root.destroy()


def main(args=None):
    rclpy.init(args=args)
    node = ChessTestGuiNode()
    root = tk.Tk()
    gui = ChessTestGui(root, node)
    try:
        root.mainloop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
