#!/usr/bin/env python3
"""
象棋识别ROS2节点
功能：
1. 订阅图像话题，使用YOLO识别象棋
2. 计算每个象棋的中心像素坐标和像素直径
3. 根据已知实际棋子直径，推算像素到实际坐标的转换比例
4. 以参考棋子为坐标原点，计算另一个棋子的相对实际坐标
5. 发布毫米单位的相对坐标到话题
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from ultralytics import YOLO
import cv2
import numpy as np
import time


class ChessDetector(Node):
    def __init__(self):
        super().__init__('chess_detector')
        
        # 参数声明
        self.declare_parameter('model_path', '/home/wyy/下载/yolov8n.pt')  # YOLO模型路径
        self.declare_parameter('image_topic', '/camera/image_raw')  # 图像话题
        self.declare_parameter('chess_real_diameter_mm', 20.0)  # 象棋实际直径，单位：毫米
        self.declare_parameter('confidence_threshold', 0.5)  # 置信度阈值
        self.declare_parameter('chess_class_name', 'chess')  # 象棋在模型中的类别名（需根据你的模型调整）
        self.declare_parameter('reference_chess_position', 'rightmost')  # 参考棋子：rightmost 或 leftmost
        self.declare_parameter('processing_interval_sec', 1.0)  # 图像处理间隔，单位：秒
        self.declare_parameter('device', 'auto')  # YOLO推理设备：auto/cpu/cuda:0
        self.declare_parameter('publish_pixel_center_topic', '/weed_detector/pixel_center')
        self.declare_parameter('top_line_y', 45.0)
        self.declare_parameter('bottom_line_y', 460.0)
        self.declare_parameter('belt_width_mm', 100.0)
        self.declare_parameter('draw_calibration_lines', True)
        
        # 获取参数
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.chess_real_diameter_mm = self.get_parameter('chess_real_diameter_mm').get_parameter_value().double_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.chess_class_name = self.get_parameter('chess_class_name').get_parameter_value().string_value
        self.reference_chess_position = self.get_parameter('reference_chess_position').get_parameter_value().string_value
        self.processing_interval_sec = self.get_parameter('processing_interval_sec').get_parameter_value().double_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.publish_pixel_center_topic = self.get_parameter('publish_pixel_center_topic').get_parameter_value().string_value
        self.top_line_y = self.get_parameter('top_line_y').get_parameter_value().double_value
        self.bottom_line_y = self.get_parameter('bottom_line_y').get_parameter_value().double_value
        self.belt_width_mm = self.get_parameter('belt_width_mm').get_parameter_value().double_value
        self.draw_calibration_lines = self.get_parameter('draw_calibration_lines').get_parameter_value().bool_value
        self.last_processing_time = 0.0
        self.last_detection_log_time = 0.0
        
        # 初始化YOLO模型
        self.get_logger().info(f'加载YOLO模型: {self.model_path}')
        self.model = YOLO(self.model_path)
        
        # 获取象棋类别的ID
        # Ultralytics YOLO的类别名和ID映射
        self.chess_class_id = None
        if hasattr(self.model, 'names'):
            self.get_logger().info(f'模型类别: {self.model.names}')
            for class_id, class_name in self.model.names.items():
                if class_name.lower() == self.chess_class_name.lower():
                    self.chess_class_id = class_id
                    break
        
        if self.chess_class_id is None:
            self.get_logger().warning(
                f'模型中未找到类别 "{self.chess_class_name}"，将显示模型检测到的所有类别。'
            )
        else:
            self.get_logger().info(f'象棋类别ID: {self.chess_class_id}, 类别名: {self.chess_class_name}')
        
        # 初始化CV Bridge
        self.bridge = CvBridge()
        
        # 订阅图像话题
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )
        
        # 发布相对坐标话题
        self.relative_pose_pub = self.create_publisher(
            PointStamped,
            '/chess/relative_position',
            10
        )
        
        # 发布像素坐标和实际坐标转换比例（用于调试和验证）
        self.pixel_scale_pub = self.create_publisher(
            PointStamped,
            '/chess/pixel_scale',
            10
        )

        # 发布YOLO检测框中心像素点，供传送带预测节点按轨道宽度换算成毫米坐标。
        self.pixel_center_pub = self.create_publisher(
            Point,
            self.publish_pixel_center_topic,
            10
        )

        # 发布带检测框的图像，方便用 showimage/rqt_image_view 查看
        self.detection_image_pub = self.create_publisher(
            Image,
            '/chess/detection_image',
            10
        )
        
        # 发布可视化标记（用于RViz2显示检测结果）
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/chess/markers',
            10
        )
        
        # 中间结果存储（用于调试）
        self.last_pixel_coords = []  # 存储最近一次检测到的棋子像素坐标
        self.last_pixel_diameters = []  # 存储最近一次检测到的棋子像素直径
        
        self.get_logger().info('象棋检测节点已启动')
        self.get_logger().info(f'订阅话题: {self.image_topic}')
        self.get_logger().info(f'象棋实际直径: {self.chess_real_diameter_mm} 毫米')
        self.get_logger().info(f'置信度阈值: {self.confidence_threshold}')
        self.get_logger().info(f'参考棋子选择方式: {self.reference_chess_position}')
        self.get_logger().info(f'图像处理间隔: {self.processing_interval_sec} 秒')
        self.get_logger().info(f'YOLO推理设备: {self.device}')
        self.get_logger().info(f'像素中心发布话题: {self.publish_pixel_center_topic}')
        self.get_logger().info(
            f'轨道标定线: top_y={self.top_line_y:.1f}, bottom_y={self.bottom_line_y:.1f}, width={self.belt_width_mm:.1f}mm'
        )

    def image_callback(self, msg):
        """图像回调函数"""
        try:
            now = time.monotonic()
            if now - self.last_processing_time < self.processing_interval_sec:
                return
            self.last_processing_time = now

            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # YOLO推理
            predict_kwargs = {'source': cv_image, 'verbose': False}
            if self.device and self.device.lower() != 'auto':
                predict_kwargs['device'] = self.device
            results = self.model.predict(**predict_kwargs)
            
            # 处理检测结果
            chess_detections = self.extract_chess_detections(results)
            self.log_detection_summary(chess_detections)
            self.publish_pixel_centers(chess_detections)
            self.publish_detection_image(msg.header, cv_image, chess_detections)
            
            if len(chess_detections) < 2:
                self.get_logger().debug(f'检测到 {len(chess_detections)} 个棋子，需要至少2个')
                return
            
            # 计算像素到实际坐标的转换比例，单位：毫米/像素
            pixel_ratio_mm, chess_info = self.calculate_pixel_ratio(chess_detections)
            
            if pixel_ratio_mm is None or len(chess_info) < 2:
                return
            
            # 存储中间结果
            self.last_pixel_coords = [info['center_pixel'] for info in chess_info]
            self.last_pixel_diameters = [info['pixel_diameter'] for info in chess_info]
            
            # 以参考棋子为原点，计算另一个棋子的相对实际坐标
            reference_chess, target_chess = self.select_reference_and_target(chess_info)
            
            # 计算相对像素坐标
            pixel_dx = target_chess['center_pixel'][0] - reference_chess['center_pixel'][0]
            pixel_dy = target_chess['center_pixel'][1] - reference_chess['center_pixel'][1]
            
            # 转换为实际相对坐标（毫米）
            # 注意：Y轴可能需要翻转，因为图像坐标系Y轴向下，实际坐标系可能Y轴向上
            real_dx_mm = pixel_dx * pixel_ratio_mm
            real_dy_mm = -pixel_dy * pixel_ratio_mm  # 翻转Y轴，使其向上为正
            
            # 发布相对坐标
            self.publish_relative_position(
                msg.header, real_dx_mm, real_dy_mm,
                reference_chess, target_chess
            )
            
            # 发布转换比例（调试用）
            self.publish_pixel_scale(msg.header, pixel_ratio_mm)
            
            # 发布可视化标记
            self.publish_markers(msg.header, chess_info, cv_image.shape[:2], reference_chess)
            
            self.get_logger().info(
                f'参考棋子像素: ({reference_chess["center_pixel"][0]:.1f}, {reference_chess["center_pixel"][1]:.1f}), '
                f'目标棋子像素: ({target_chess["center_pixel"][0]:.1f}, {target_chess["center_pixel"][1]:.1f}), '
                f'相对实际坐标: ({real_dx_mm:.2f}, {real_dy_mm:.2f}) 毫米, '
                f'像素比例: {pixel_ratio_mm:.4f} 毫米/像素'
            )
            
        except Exception as e:
            self.get_logger().error(f'图像处理错误: {str(e)}')

    def extract_chess_detections(self, results):
        """提取象棋检测结果"""
        chess_detections = []
        
        for result in results:
            if result.boxes is None:
                continue
                
            boxes = result.boxes.xyxy.cpu().numpy()  # 边界框坐标
            confidences = result.boxes.conf.cpu().numpy()  # 置信度
            class_ids = result.boxes.cls.cpu().numpy()  # 类别ID
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                cls_id = int(cls_id)
                # 如果指定了象棋类别ID，只接受该类别的检测
                if self.chess_class_id is not None and cls_id != self.chess_class_id:
                    continue
                
                # 置信度过滤
                if conf < self.confidence_threshold:
                    continue
                
                # 计算中心像素坐标和直径
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 计算像素直径（使用边界框的宽度和高度，取平均值更稳定）
                box_width = x2 - x1
                box_height = y2 - y1
                pixel_diameter = (box_width + box_height) / 2
                
                chess_detections.append({
                    'center_pixel': (center_x, center_y),
                    'pixel_diameter': pixel_diameter,
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': self.model.names.get(cls_id, str(cls_id)),
                    'bbox': (x1, y1, x2, y2)
                })
        
        # 按X坐标排序，保证检测到的棋子从左到右有序
        chess_detections.sort(key=lambda x: x['center_pixel'][0])
        
        return chess_detections

    def calculate_pixel_ratio(self, chess_detections):
        """计算像素到实际坐标的转换比例"""
        if len(chess_detections) == 0:
            return None, []
        
        # 使用所有检测到的棋子计算像素直径的平均值
        total_pixel_diameter = 0
        chess_info = []
        
        for detection in chess_detections:
            total_pixel_diameter += detection['pixel_diameter']
            chess_info.append({
                'center_pixel': detection['center_pixel'],
                'pixel_diameter': detection['pixel_diameter'],
                'confidence': detection['confidence'],
                'class_id': detection['class_id'],
                'class_name': detection['class_name'],
                'bbox': detection['bbox']
            })
        
        # 计算平均像素直径
        avg_pixel_diameter = total_pixel_diameter / len(chess_detections)
        
        # 计算像素到实际坐标的转换比例
        # pixel_ratio = 实际直径 / 像素直径 = 毫米/像素
        pixel_ratio = self.chess_real_diameter_mm / avg_pixel_diameter
        
        return pixel_ratio, chess_info

    def select_reference_and_target(self, chess_info):
        """选择参考棋子和目标棋子"""
        if self.reference_chess_position.lower() == 'leftmost':
            reference_chess = min(chess_info, key=lambda info: info['center_pixel'][0])
        else:
            reference_chess = max(chess_info, key=lambda info: info['center_pixel'][0])

        target_candidates = [info for info in chess_info if info is not reference_chess]
        target_chess = min(
            target_candidates,
            key=lambda info: abs(info['center_pixel'][0] - reference_chess['center_pixel'][0])
        )
        return reference_chess, target_chess

    def publish_relative_position(self, header, real_dx_mm, real_dy_mm, reference_chess, target_chess):
        """发布相对位置坐标"""
        # 发布相对坐标
        relative_msg = PointStamped()
        relative_msg.header = header
        relative_msg.header.frame_id = 'chess_reference_frame'
        relative_msg.point.x = real_dx_mm
        relative_msg.point.y = real_dy_mm
        relative_msg.point.z = 0.0  # 假设棋子在同一平面上
        
        self.relative_pose_pub.publish(relative_msg)

    def publish_pixel_scale(self, header, pixel_ratio_mm):
        """发布像素转换比例（调试用）"""
        scale_msg = PointStamped()
        scale_msg.header = header
        scale_msg.header.frame_id = 'camera_frame'
        scale_msg.point.x = pixel_ratio_mm  # 毫米/像素
        scale_msg.point.y = 0.0
        scale_msg.point.z = 0.0
        
        self.pixel_scale_pub.publish(scale_msg)

    def publish_pixel_centers(self, chess_detections):
        """发布每个检测目标的中心像素坐标，给传送带预测队列使用."""
        for detection in chess_detections:
            center_x, center_y = detection['center_pixel']
            pixel_msg = Point()
            pixel_msg.x = float(center_x)
            pixel_msg.y = float(center_y)
            pixel_msg.z = 0.0
            self.pixel_center_pub.publish(pixel_msg)

    def log_detection_summary(self, chess_detections):
        """低频打印检测数量，方便确认YOLO是否真的识别到目标."""
        now = time.monotonic()
        if now - self.last_detection_log_time < 1.0:
            return
        self.last_detection_log_time = now

        if not chess_detections:
            self.get_logger().info('本帧未检测到目标')
            return

        centers = []
        for detection in chess_detections[:5]:
            center_x, center_y = detection['center_pixel']
            centers.append('(%.1f, %.1f) %.2f' % (center_x, center_y, float(detection['confidence'])))
        self.get_logger().info(
            '检测到 %d 个目标，中心/置信度: %s'
            % (len(chess_detections), ', '.join(centers))
        )

    def publish_markers(self, header, chess_info, image_shape, reference_chess):
        """发布可视化标记"""
        marker_array = MarkerArray()
        
        for i, info in enumerate(chess_info):
            marker = Marker()
            marker.header = header
            marker.header.frame_id = 'camera_frame'
            marker.ns = 'chess_detection'
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # 位置（使用实际坐标，Z轴假设为0）
            marker.pose.position.x = float(info['center_pixel'][0])
            marker.pose.position.y = float(info['center_pixel'][1])
            marker.pose.position.z = 0.0
            
            # 大小（像素直径）
            marker.scale.x = float(info['pixel_diameter'])
            marker.scale.y = float(info['pixel_diameter'])
            marker.scale.z = 0.01  # 薄片
            
            # 颜色：参考棋子为绿色，目标棋子为红色，其他为蓝色
            if info == reference_chess:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            marker.color.a = 0.8
            
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)

    def publish_detection_image(self, header, image, chess_detections):
        """发布带检测框的图像"""
        display_img = image.copy()
        self.draw_track_calibration(display_img)

        for i, detection in enumerate(chess_detections):
            x1, y1, x2, y2 = detection['bbox']
            center_x, center_y = detection['center_pixel']
            diameter = int(detection['pixel_diameter'])
            confidence = float(detection['confidence'])
            class_name = detection.get('class_name', self.chess_class_name)

            color = (0, 255, 0) if i == 0 else (0, 0, 255)
            cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.circle(display_img, (int(center_x), int(center_y)), 5, color, -1)
            cv2.circle(display_img, (int(center_x), int(center_y)), diameter // 2, color, 2)

            label = f"{class_name} {confidence:.2f}"
            if i == 0:
                label += " Ref"
            text_x = int(x1)
            text_y = max(int(y1) - 8, 18)
            cv2.putText(
                display_img,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        detection_msg = self.bridge.cv2_to_imgmsg(display_img, encoding='bgr8')
        detection_msg.header = header
        self.detection_image_pub.publish(detection_msg)

    def draw_track_calibration(self, display_img):
        """在调试画面上画轨道标定线，与原上位机的上下水平线对应."""
        self.top_line_y = self.get_parameter('top_line_y').get_parameter_value().double_value
        self.bottom_line_y = self.get_parameter('bottom_line_y').get_parameter_value().double_value
        self.belt_width_mm = self.get_parameter('belt_width_mm').get_parameter_value().double_value
        self.draw_calibration_lines = self.get_parameter('draw_calibration_lines').get_parameter_value().bool_value

        if not self.draw_calibration_lines:
            return

        height, width = display_img.shape[:2]
        top_y = max(0, min(height - 1, int(self.top_line_y)))
        bottom_y = max(0, min(height - 1, int(self.bottom_line_y)))
        center_y = int((top_y + bottom_y) / 2)
        center_x = int(width / 2)

        line_color = (0, 255, 255)
        center_color = (255, 0, 255)
        cv2.line(display_img, (0, top_y), (width, top_y), line_color, 2)
        cv2.line(display_img, (0, bottom_y), (width, bottom_y), line_color, 2)
        cv2.line(display_img, (center_x, 0), (center_x, height), center_color, 1)
        cv2.line(display_img, (0, center_y), (width, center_y), center_color, 1)
        cv2.putText(
            display_img,
            f'track {self.belt_width_mm:.0f}mm',
            (8, max(top_y - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            line_color,
            2
        )

    def display_detection(self, image, chess_info, reference_chess):
        """显示检测结果（调试用）"""
        display_img = image.copy()
        
        for i, info in enumerate(chess_info):
            x1, y1, x2, y2 = info['bbox']
            center_x, center_y = info['center_pixel']
            diameter = int(info['pixel_diameter'])
            
            # 绘制边界框
            color = (0, 255, 0) if info == reference_chess else (0, 0, 255)
            cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # 绘制中心点
            cv2.circle(display_img, (int(center_x), int(center_y)), 5, color, -1)
            
            # 绘制直径圆
            cv2.circle(display_img, (int(center_x), int(center_y)), diameter // 2, color, 2)
            
            # 标注文本
            label = f"Chess {i+1} (Ref)" if info == reference_chess else f"Chess {i+1}"
            cv2.putText(display_img, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow('Chess Detection', display_img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    
    chess_detector = ChessDetector()
    
    try:
        rclpy.spin(chess_detector)
    except KeyboardInterrupt:
        chess_detector.get_logger().info('节点被用户中断')
    finally:
        # 清理OpenCV窗口
        cv2.destroyAllWindows()
        chess_detector.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
