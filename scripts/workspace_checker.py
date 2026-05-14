#!/usr/bin/env python3
"""Delta机械臂工作空间边界检查节点."""

import math

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Bool


class WorkspaceChecker:
    """工作空间边界检查器."""

    def __init__(self, boundary_points):
        """初始化工作空间检查器."""
        # 提取XY坐标（Z固定）
        self.original_points = [(x, y) for x, y, z in boundary_points]
        # 按角度排序边界点
        self.boundary = self._sort_by_angle(self.original_points)
        # 计算中心点
        self.center = self._calculate_center()

    def _sort_by_angle(self, points):
        """按相对于原点的角度排序边界点."""
        def angle(point):
            return math.atan2(point[1], point[0])
        return sorted(points, key=angle)

    def _calculate_center(self):
        """计算边界点的中心点."""
        if not self.boundary:
            return (0.0, 0.0)
        x_avg = sum(p[0] for p in self.boundary) / len(self.boundary)
        y_avg = sum(p[1] for p in self.boundary) / len(self.boundary)
        return (x_avg, y_avg)

    def _point_in_polygon(self, point):
        """使用射线法判断点是否在多边形内."""
        x, y = point
        inside = False
        n = len(self.boundary)

        for i in range(n):
            j = (i + 1) % n
            xi, yi = self.boundary[i]
            xj, yj = self.boundary[j]

            if ((yi > y) != (yj > y)) and \
               (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside

        return inside

    def _shrink_point(self, point, margin):
        """将单个点向中心收缩margin毫米."""
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        distance = math.sqrt(dx * dx + dy * dy)

        if distance > margin:
            ratio = (distance - margin) / distance
            return (
                self.center[0] + dx * ratio,
                self.center[1] + dy * ratio
            )
        return self.center

    def check_point(self, x, y, safety_margin=5.0):
        """
        检查坐标是否在工作空间内（包含安全余量）.
        :param x: X坐标（毫米）
        :param y: Y坐标（毫米）
        :param safety_margin: 安全余量（毫米），边界向内收缩的距离
        :return: True=在工作空间内，False=超出工作空间
        """
        # 计算安全边界
        safe_boundary = [
            self._shrink_point(p, safety_margin)
            for p in self.boundary
        ]

        # 检查点是否在安全边界内
        return self._point_in_polygon((x, y), safe_boundary)

    def _point_in_polygon(self, point, polygon):
        """使用射线法判断点是否在指定多边形内."""
        x, y = point
        inside = False
        n = len(polygon)

        for i in range(n):
            j = (i + 1) % n
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            if ((yi > y) != (yj > y)) and \
               (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside

        return inside

    def get_boundary(self):
        """返回排序后的边界点."""
        return self.boundary

    def get_center(self):
        """返回边界中心点."""
        return self.center


class WorkspaceCheckerNode(Node):
    """ROS2工作空间边界检查节点."""

    def __init__(self):
        super().__init__('workspace_checker')

        # Delta机械臂边界点（用户实测数据）
        self.boundary_points = [
            (-40.0, 70.0, -180.0),
            (-50.0, 50.0, -180.0),
            (-60.0, 20.0, -180.0),
            (-70.0, 10.0, -180.0),
            (-80.0, 0.0, -180.0),
            (-90.0, -10.0, -180.0),
            (40.0, 70.0, -180.0),
            (50.0, 50.0, -180.0),
            (60.0, 20.0, -180.0),
            (70.0, 10.0, -180.0),
            (80.0, 0.0, -180.0),
            (90.0, -10.0, -180.0),
            (110.0, -40.0, -180.0),
            (0.0, 120.0, -180.0),
            (0.0, -60.0, -180.0),
            (-110.0, -40.0, -180.0),
        ]

        # 初始化工作空间检查器
        self.checker = WorkspaceChecker(self.boundary_points)

        # 参数
        self.declare_parameter('safety_margin', 5.0)
        self.safety_margin = float(self.get_parameter('safety_margin').value)

        # 订阅目标坐标。target_position 只做检查，pick_target_raw 会检查并转发。
        self.target_sub = self.create_subscription(
            Point,
            '/delta_arm/target_position',
            self.on_target_position,
            10
        )
        self.raw_pick_sub = self.create_subscription(
            Point,
            '/delta_arm/pick_target_raw',
            self.on_pick_target_raw,
            10
        )

        # 发布检查结果
        self.result_pub = self.create_publisher(
            Bool,
            '/delta_arm/position_valid',
            10
        )
        self.valid_pick_pub = self.create_publisher(
            Point,
            '/delta_arm/pick_target',
            10
        )

        # 服务
        # 可以添加服务接口供其他节点查询

        self.get_logger().info('工作空间边界检查节点已启动')
        self.get_logger().info(f'边界点数量: {len(self.boundary_points)}')
        self.get_logger().info(f'安全余量: {self.safety_margin}mm')
        self.get_logger().info(f'工作空间中心: {self.checker.get_center()}')
        self.get_logger().info('订阅: /delta_arm/target_position, /delta_arm/pick_target_raw')
        self.get_logger().info('发布: /delta_arm/position_valid, /delta_arm/pick_target')

    def on_target_position(self, msg):
        """处理目标位置消息."""
        x = msg.x
        y = msg.y
        z = msg.z

        # 检查坐标是否在工作空间内
        is_valid = self.checker.check_point(x, y, self.safety_margin)

        # 发布检查结果
        result_msg = Bool()
        result_msg.data = is_valid
        self.result_pub.publish(result_msg)

        # 日志输出
        if is_valid:
            self.get_logger().info(f'坐标({x:.1f}, {y:.1f}, {z:.1f}) 在工作空间内')
        else:
            self.get_logger().warn(f'坐标({x:.1f}, {y:.1f}, {z:.1f}) 超出工作空间！')

        return is_valid

    def on_pick_target_raw(self, msg):
        """检查抓取目标，合法才转发给target_commander."""
        is_valid = self.on_target_position(msg)
        if is_valid:
            self.valid_pick_pub.publish(msg)
            self.get_logger().info(
                f'已转发抓取目标: ({msg.x:.1f}, {msg.y:.1f}, {msg.z:.1f})'
            )
        else:
            self.get_logger().warn(
                f'已拦截非法抓取目标: ({msg.x:.1f}, {msg.y:.1f}, {msg.z:.1f})'
            )


def main(args=None):
    rclpy.init(args=args)
    node = WorkspaceCheckerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
