#!/usr/bin/env python3
"""Visual Delta pick demo using a colored target on a fixed work plane."""

import math
import select
import sys
import termios
import tty

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Empty

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


class DeltaVisualPickDemo(Node):
    def __init__(self):
        super().__init__('delta_visual_pick_demo')

        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('delta_move_topic', '/delta_arm/move_to')
        self.declare_parameter('trigger_topic', '/delta_visual_pick_demo/trigger')
        self.declare_parameter(
            'calibration_path',
            '/home/whr/cc_ws/tros_ws/calibration_targets/delta_hand_eye_filtered.yaml',
        )
        self.declare_parameter('work_z_mm', -170.0)
        self.declare_parameter('approach_z_mm', -155.0)
        self.declare_parameter('move_to_work_z', False)
        self.declare_parameter('offset_x_mm', 10.0)
        self.declare_parameter('offset_y_mm', -5.0)
        self.declare_parameter('offset_z_mm', 3.0)
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
        self.delta_move_topic = str(self.get_parameter('delta_move_topic').value)
        self.trigger_topic = str(self.get_parameter('trigger_topic').value)
        self.calibration_path = str(self.get_parameter('calibration_path').value)
        self.work_z_mm = float(self.get_parameter('work_z_mm').value)
        self.approach_z_mm = float(self.get_parameter('approach_z_mm').value)
        self.move_to_work_z = bool(self.get_parameter('move_to_work_z').value)
        self.offset_mm = np.array(
            [
                float(self.get_parameter('offset_x_mm').value),
                float(self.get_parameter('offset_y_mm').value),
                float(self.get_parameter('offset_z_mm').value),
            ],
            dtype=float,
        )
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
        self.latest_target = None
        self.latest_frame = None
        self.bridge = CvBridge()
        self.yolo_model = None

        self.t_delta_camera = load_transform(self.calibration_path)
        self.r_delta_camera = self.t_delta_camera[:3, :3]
        self.t_delta_camera_vec = self.t_delta_camera[:3, 3]

        if self.detector == 'yolo':
            self.load_yolo_model()

        self.move_pub = self.create_publisher(Point, self.delta_move_topic, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.on_camera_info, 10)
        self.create_subscription(Image, self.image_topic, self.on_image, 10)
        self.create_subscription(Empty, self.trigger_topic, self.on_trigger, 10)
        self.create_timer(0.03, self.on_timer)
        self.create_timer(1.0, self.on_status_timer)

        self.get_logger().info('Delta visual pick demo ready')
        self.get_logger().info(f'image: {self.image_topic}')
        self.get_logger().info(f'trigger: {self.trigger_topic}')
        self.get_logger().info(f'detector: {self.detector}')
        self.get_logger().info(f'calibration: {self.calibration_path}')
        self.get_logger().info(
            'fixed plane: work_z=%.1f mm, approach_z=%.1f mm, offset=%s mm'
            % (self.work_z_mm, self.approach_z_mm, np.round(self.offset_mm, 2).tolist())
        )
        self.get_logger().info('keys: SPACE/m/ENTER send move, w write debug image, q quit')

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
        denominator = float(self.r_delta_camera[2, :] @ ray_camera)
        if abs(denominator) < 1e-6:
            return None

        work_z_m = self.work_z_mm / 1000.0
        scale = (work_z_m - self.t_delta_camera_vec[2]) / denominator
        if not math.isfinite(scale) or scale <= 0.0:
            return None

        camera_xyz_m = ray_camera * scale
        delta_xyz_m = self.r_delta_camera @ camera_xyz_m + self.t_delta_camera_vec
        delta_xyz_mm = delta_xyz_m * 1000.0 + self.offset_mm
        return camera_xyz_m, delta_xyz_mm

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
            result = self.pixel_to_delta_on_plane(u, v)
            if result is None:
                continue
            camera_xyz_m, delta_xyz_mm = result
            x, y, w, h = cv2.boundingRect(contour)
            return {
                'u': u,
                'v': v,
                'area': area,
                'bbox': (x, y, w, h),
                'camera_xyz_m': camera_xyz_m,
                'delta_xyz_mm': delta_xyz_mm,
            }, mask
        return None, mask

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
        result = self.pixel_to_delta_on_plane(u, v)
        if result is None:
            return None, mask
        camera_xyz_m, delta_xyz_mm = result
        names = getattr(self.yolo_model, 'names', {})
        label = str(names.get(best['cls_id'], best['cls_id']))
        return {
            'u': u,
            'v': v,
            'area': best['area'],
            'bbox': (
                int(round(x1)),
                int(round(y1)),
                int(round(x2 - x1)),
                int(round(y2 - y1)),
            ),
            'camera_xyz_m': camera_xyz_m,
            'delta_xyz_mm': delta_xyz_mm,
            'label': label,
            'confidence': best['conf'],
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
            cv2.putText(
                canvas,
                'READY: press SPACE to move',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 180, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                self.target_summary_text(target),
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                'delta=(%.1f, %.1f, %.1f) mm'
                % (delta[0], delta[1], self.command_z_mm(delta[2])),
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        small_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        small_mask = cv2.resize(small_mask, (canvas.shape[1] // 4, canvas.shape[0] // 4))
        y0 = canvas.shape[0] - small_mask.shape[0] - 10
        x0 = canvas.shape[1] - small_mask.shape[1] - 10
        canvas[y0:y0 + small_mask.shape[0], x0:x0 + small_mask.shape[1]] = small_mask
        return canvas

    def target_summary_text(self, target):
        text = 'pixel=(%.0f, %.0f) area=%.0f' % (
            target['u'],
            target['v'],
            target['area'],
        )
        if 'label' in target:
            text += ' %s %.2f' % (target['label'], target.get('confidence', 0.0))
        return text

    def command_z_mm(self, transformed_z_mm):
        if self.move_to_work_z:
            return transformed_z_mm
        return self.approach_z_mm

    def send_latest_move(self):
        if self.latest_target is None:
            self.get_logger().warning('no target to move to')
            return
        delta = self.latest_target['delta_xyz_mm']
        msg = Point()
        msg.x = float(delta[0])
        msg.y = float(delta[1])
        msg.z = float(self.command_z_mm(delta[2]))
        self.move_pub.publish(msg)
        self.get_logger().info(
            'sent demo move: x=%.1f y=%.1f z=%.1f mm'
            % (msg.x, msg.y, msg.z)
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
                self.command_z_mm(delta[2]),
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
