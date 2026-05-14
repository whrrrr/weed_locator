#!/usr/bin/env python3
"""Generate an SVG plot for the measured Delta workspace boundary."""

import math
from pathlib import Path


BOUNDARY_POINTS = [
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


def sort_by_angle(points):
    cx = sum(x for x, _ in points) / len(points)
    cy = sum(y for _, y in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx)), (cx, cy)


def shrink_boundary(points, center, margin):
    cx, cy = center
    shrunk = []
    for x, y in points:
        dx = x - cx
        dy = y - cy
        dist = math.hypot(dx, dy)
        if dist <= margin:
            shrunk.append((cx, cy))
        else:
            ratio = (dist - margin) / dist
            shrunk.append((cx + dx * ratio, cy + dy * ratio))
    return shrunk


def polygon_area(points):
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def main():
    margin = 5.0
    raw_points = [(x, y) for x, y, _ in BOUNDARY_POINTS]
    boundary, center = sort_by_angle(raw_points)
    safe_boundary = shrink_boundary(boundary, center, margin)

    xs = [x for x, _ in boundary]
    ys = [y for _, y in boundary]
    pad = 30
    x_min, x_max = min(xs) - pad, max(xs) + pad
    y_min, y_max = min(ys) - pad, max(ys) + pad
    width = 900
    height = 700

    def sx(x):
        return (x - x_min) / (x_max - x_min) * width

    def sy(y):
        return height - (y - y_min) / (y_max - y_min) * height

    def points_attr(points):
        return " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)

    axis_x = sx(0)
    axis_y = sy(0)
    cx, cy = center
    area = polygon_area(boundary)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial, sans-serif;font-size:14px}.small{font-size:12px}.label{font-weight:bold}</style>',
        f'<line x1="{axis_x:.1f}" y1="0" x2="{axis_x:.1f}" y2="{height}" stroke="#d0d0d0" stroke-width="1"/>',
        f'<line x1="0" y1="{axis_y:.1f}" x2="{width}" y2="{axis_y:.1f}" stroke="#d0d0d0" stroke-width="1"/>',
        f'<polygon points="{points_attr(boundary)}" fill="#8ecae6" fill-opacity="0.28" stroke="#126782" stroke-width="3"/>',
        f'<polygon points="{points_attr(safe_boundary)}" fill="#90be6d" fill-opacity="0.25" stroke="#2d6a4f" stroke-width="2" stroke-dasharray="8 6"/>',
    ]

    for idx, (x, y) in enumerate(boundary, start=1):
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="#d62828"/>')
        parts.append(f'<text class="small" x="{sx(x)+7:.1f}" y="{sy(y)-7:.1f}">P{idx} ({x:.0f},{y:.0f})</text>')

    parts.extend([
        f'<circle cx="{sx(0):.1f}" cy="{sy(0):.1f}" r="6" fill="#1d4ed8"/>',
        f'<text x="{sx(0)+8:.1f}" y="{sy(0)+20:.1f}">origin (0,0)</text>',
        f'<circle cx="{sx(cx):.1f}" cy="{sy(cy):.1f}" r="7" fill="#f59e0b"/>',
        f'<text x="{sx(cx)+8:.1f}" y="{sy(cy)-10:.1f}">center ({cx:.1f},{cy:.1f})</text>',
        '<rect x="18" y="18" width="360" height="116" fill="#ffffff" stroke="#dddddd"/>',
        '<text class="label" x="34" y="44">Delta workspace boundary, Z=-180mm</text>',
        f'<text x="34" y="70">X range: [{min(xs):.0f}, {max(xs):.0f}] mm</text>',
        f'<text x="34" y="92">Y range: [{min(ys):.0f}, {max(ys):.0f}] mm</text>',
        f'<text x="34" y="114">Polygon area: {area:.0f} mm^2</text>',
        f'<text x="34" y="136">Dashed green: safety margin {margin:.0f}mm</text>',
        '</svg>',
    ])

    out = Path('/home/whr/cc_ws/tros_ws/workspace_boundary_z180.svg')
    out.write_text("\n".join(parts), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
