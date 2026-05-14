#!/usr/bin/env python3
"""Delta机械臂工作空间边界ASCII可视化工具."""

import math


class WorkspaceASCIIPlot:
    """ASCII工作空间可视化工具."""

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

    def _point_in_polygon(self, point, polygon):
        """使用射线法判断点是否在多边形内."""
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

    def plot_ascii(self, safety_margin=5.0, scale=0.5):
        """绘制ASCII工作空间图."""
        # 计算绘图区域
        x_coords = [p[0] for p in self.sorted_points]
        y_coords = [p[1] for p in self.sorted_points]
        
        x_min, x_max = min(x_coords) - 10, max(x_coords) + 10
        y_min, y_max = min(y_coords) - 10, max(y_coords) + 10

        # 计算画布大小
        width = int((x_max - x_min) * scale) + 1
        height = int((y_max - y_min) * scale) + 1

        # 创建画布
        grid = [[' ' for _ in range(width)] for _ in range(height)]

        # 坐标转换函数
        def to_grid(x, y):
            gx = int((x - x_min) * scale)
            gy = int((y_max - y) * scale)
            return gx, gy

        # 绘制安全边界内部（填充）
        safe_points = self._shrink_boundary(safety_margin)
        for gy in range(height):
            for gx in range(width):
                x = x_min + gx / scale
                y = y_max - gy / scale
                if self._point_in_polygon((x, y), safe_points):
                    grid[gy][gx] = '·'

        # 绘制原始边界内部（填充）
        for gy in range(height):
            for gx in range(width):
                x = x_min + gx / scale
                y = y_max - gy / scale
                if self._point_in_polygon((x, y), self.sorted_points):
                    if grid[gy][gx] == ' ':
                        grid[gy][gx] = '░'

        # 绘制原始边界线
        for i in range(len(self.sorted_points)):
            x1, y1 = self.sorted_points[i]
            x2, y2 = self.sorted_points[(i + 1) % len(self.sorted_points)]
            self._draw_line(grid, x1, y1, x2, y2, x_min, y_max, scale, '#')

        # 绘制安全边界线
        for i in range(len(safe_points)):
            x1, y1 = safe_points[i]
            x2, y2 = safe_points[(i + 1) % len(safe_points)]
            self._draw_line(grid, x1, y1, x2, y2, x_min, y_max, scale, ':')

        # 绘制边界点
        for i, (x, y) in enumerate(self.sorted_points):
            gx, gy = to_grid(x, y)
            if 0 <= gx < width and 0 <= gy < height:
                grid[gy][gx] = str(i + 1) if i < 9 else 'X'

        # 绘制中心点
        gx, gy = to_grid(self.center[0], self.center[1])
        if 0 <= gx < width and 0 <= gy < height:
            grid[gy][gx] = '◎'

        # 绘制原点
        gx, gy = to_grid(0, 0)
        if 0 <= gx < width and 0 <= gy < height:
            grid[gy][gx] = '●'

        # 添加坐标轴标签
        print("\n" + "=" * 60)
        print("Delta机械臂工作空间ASCII图 (Z=-180mm)")
        print("=" * 60)
        print(f"X范围: [{x_min:.0f}, {x_max:.0f}] mm")
        print(f"Y范围: [{y_min:.0f}, {y_max:.0f}] mm")
        print("图例:")
        print("  ● : 原点(0,0)")
        print("  ◎ : 工作空间中心")
        print("  # : 原始边界线")
        print("  : : 安全边界线 (-5mm)")
        print("  ░ : 原始边界内")
        print("  · : 安全边界内")
        print("  1-9,X : 边界点编号")
        print("-" * 60)

        # 打印画布
        y_label = y_max
        for row in grid:
            y_str = f"{y_label:6.0f} |"
            print(y_str + ''.join(row))
            y_label -= 1 / scale

        # 添加X轴标签
        print("      +" + "-" * width)
        x_ticks = [x_min + i * (x_max - x_min) / 5 for i in range(6)]
        x_labels = "        "
        for xt in x_ticks:
            x_labels += f"{xt:6.0f}"
        print(x_labels)
        print(" " * 10 + "X轴 (mm)")

    def _draw_line(self, grid, x1, y1, x2, y2, x_min, y_max, scale, char):
        """绘制线段."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        width = len(grid[0])
        height = len(grid)

        while True:
            gx = int((x1 - x_min) * scale)
            gy = int((y_max - y1) * scale)
            if 0 <= gx < width and 0 <= gy < height:
                if grid[gy][gx] == ' ':
                    grid[gy][gx] = char

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

    def print_info(self):
        """打印边界信息."""
        print("\n=== Delta机械臂工作空间信息 ===")
        print(f"边界点数量: {len(self.boundary_points)}")
        print(f"工作空间中心: ({self.center[0]:.1f}, {self.center[1]:.1f})")
        
        x_coords = [p[0] for p in self.sorted_points]
        y_coords = [p[1] for p in self.sorted_points]
        print(f"X轴范围: [{min(x_coords):.1f}, {max(x_coords):.1f}] mm")
        print(f"Y轴范围: [{min(y_coords):.1f}, {max(y_coords):.1f}] mm")
        
        area = self._calculate_area()
        print(f"工作空间面积: {area:.1f} mm²")
        print()

    def _calculate_area(self):
        """使用鞋带公式计算多边形面积."""
        n = len(self.sorted_points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.sorted_points[i][0] * self.sorted_points[j][1]
            area -= self.sorted_points[j][0] * self.sorted_points[i][1]
        return abs(area) / 2.0

    def print_boundary_points(self):
        """打印所有边界点."""
        print("\n=== 边界点列表 ===")
        for i, (x, y, z) in enumerate(self.boundary_points):
            print(f"P{i+1}: ({x:6.1f}, {y:6.1f}, {z:6.1f})")


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
    plotter = WorkspaceASCIIPlot(boundary_points)

    # 打印边界信息
    plotter.print_info()

    # 打印边界点列表
    plotter.print_boundary_points()

    # 绘制ASCII图
    plotter.plot_ascii(safety_margin=5.0, scale=0.3)


if __name__ == '__main__':
    main()