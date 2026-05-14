#!/usr/bin/env python3
"""Delta机械臂工作空间边界可视化工具."""

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class WorkspaceVisualizer:
    """工作空间可视化工具."""

    def __init__(self, boundary_points):
        """初始化可视化工具."""
        self.boundary_points = boundary_points
        self.xy_points = [(x, y) for x, y, z in boundary_points]
        self.sorted_points = self._sort_by_angle(self.xy_points)
        self.center = self._calculate_center()

    def _sort_by_angle(self, points):
        """按相对于原点的角度排序边界点."""
        def angle(point):
            return math.atan2(point[1], point[0])
        return sorted(points, key=angle)

    def _calculate_center(self):
        """计算边界点的中心点."""
        x_avg = sum(p[0] for p in self.sorted_points) / len(self.sorted_points)
        y_avg = sum(p[1] for p in self.sorted_points) / len(self.sorted_points)
        return (x_avg, y_avg)

    def _shrink_boundary(self, margin=5.0):
        """将边界向内收缩margin毫米."""
        shrunk = []
        for x, y in self.sorted_points:
            dx = x - self.center[0]
            dy = y - self.center[1]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance > margin:
                ratio = (distance - margin) / distance
                shrunk.append((
                    self.center[0] + dx * ratio,
                    self.center[1] + dy * ratio
                ))
            else:
                shrunk.append(self.center)
        return shrunk

    def plot_workspace(self, safety_margin=5.0):
        """绘制工作空间边界图."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')

        # 提取排序后的边界点坐标
        x_coords = [p[0] for p in self.sorted_points]
        y_coords = [p[1] for p in self.sorted_points]

        # 闭合多边形（首尾相连）
        x_coords.append(self.sorted_points[0][0])
        y_coords.append(self.sorted_points[0][1])

        # 绘制原始边界
        ax.plot(x_coords, y_coords, 'b-', linewidth=2, label='原始边界')
        
        # 绘制边界点
        ax.scatter(x_coords[:-1], y_coords[:-1], c='red', s=50, label='边界点')

        # 绘制安全边界
        if safety_margin > 0:
            safe_points = self._shrink_boundary(safety_margin)
            safe_x = [p[0] for p in safe_points] + [safe_points[0][0]]
            safe_y = [p[1] for p in safe_points] + [safe_points[0][1]]
            ax.plot(safe_x, safe_y, 'g--', linewidth=2, label=f'安全边界 (-{safety_margin}mm)')

        # 绘制中心点
        ax.scatter(self.center[0], self.center[1], c='green', s=100, marker='*', label='中心点')

        # 绘制原点
        ax.scatter(0, 0, c='blue', s=80, marker='o', label='原点(0,0)')

        # 设置坐标轴
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('Delta机械臂工作空间边界图 (Z=-180mm)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        # 设置坐标轴范围（留出边距）
        all_x = x_coords + safe_x if safety_margin > 0 else x_coords
        all_y = y_coords + safe_y if safety_margin > 0 else y_coords
        x_min, x_max = min(all_x) - 20, max(all_x) + 20
        y_min, y_max = min(all_y) - 20, max(all_y) + 20
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # 添加标注
        for i, (x, y) in enumerate(self.sorted_points):
            ax.annotate(f'P{i+1}', (x+3, y+3), fontsize=8)

        plt.show()

    def print_boundary_info(self):
        """打印边界信息."""
        print("\n=== Delta机械臂工作空间信息 ===")
        print(f"边界点数量: {len(self.boundary_points)}")
        print(f"工作空间中心: ({self.center[0]:.1f}, {self.center[1]:.1f})")
        
        # 计算尺寸范围
        x_coords = [p[0] for p in self.sorted_points]
        y_coords = [p[1] for p in self.sorted_points]
        print(f"X轴范围: [{min(x_coords):.1f}, {max(x_coords):.1f}] mm")
        print(f"Y轴范围: [{min(y_coords):.1f}, {max(y_coords):.1f}] mm")
        
        # 计算近似面积（使用鞋带公式）
        area = self._calculate_area()
        print(f"工作空间面积: {area:.1f} mm²")

    def _calculate_area(self):
        """使用鞋带公式计算多边形面积."""
        n = len(self.sorted_points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.sorted_points[i][0] * self.sorted_points[j][1]
            area -= self.sorted_points[j][0] * self.sorted_points[i][1]
        return abs(area) / 2.0


def main():
    """主函数."""
    # 用户提供的16个边界点
    boundary_points = [
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

    # 创建可视化工具
    visualizer = WorkspaceVisualizer(boundary_points)

    # 打印边界信息
    visualizer.print_boundary_info()

    # 绘制工作空间图
    visualizer.plot_workspace(safety_margin=5.0)


if __name__ == '__main__':
    main()