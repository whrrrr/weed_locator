#!/usr/bin/env python3
"""Split a Delta hand-eye calibration YAML into per-Z calibration files."""

import argparse
import copy
import math
from pathlib import Path

import numpy as np
import yaml


def sample_points(sample):
    delta = sample.get('delta_xyz_mm') or sample.get('delta_xyz') or sample.get('robot_xyz_mm')
    camera = sample.get('camera_xyz_m') or sample.get('camera_xyz')
    if delta is None or camera is None:
        return None
    return np.array(camera, dtype=float) * 1000.0, np.array(delta, dtype=float)


def fit_rigid_transform(camera_points_mm, delta_points_mm):
    camera_center = camera_points_mm.mean(axis=0)
    delta_center = delta_points_mm.mean(axis=0)
    camera_zero = camera_points_mm - camera_center
    delta_zero = delta_points_mm - delta_center

    u, _, vt = np.linalg.svd(camera_zero.T @ delta_zero)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T

    translation_mm = delta_center - rotation @ camera_center
    predicted = (rotation @ camera_points_mm.T).T + translation_mm
    residuals = np.linalg.norm(predicted - delta_points_mm, axis=1)

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation_mm / 1000.0
    return transform, residuals


def z_key(delta_xyz_mm, tolerance_mm):
    z = float(delta_xyz_mm[2])
    if tolerance_mm <= 0.0:
        return int(round(z))
    return int(round(z / tolerance_mm) * tolerance_mm)


def split_by_z(data, tolerance_mm):
    layers = {}
    for sample in data.get('samples', []):
        points = sample_points(sample)
        if points is None:
            continue
        _, delta_xyz_mm = points
        layers.setdefault(z_key(delta_xyz_mm, tolerance_mm), []).append(sample)
    return layers


def write_layer(source_data, source_path, output_dir, prefix, z_mm, samples):
    parsed = [sample_points(sample) for sample in samples]
    parsed = [item for item in parsed if item is not None]
    if len(parsed) < 3:
        return None

    camera_points_mm = np.array([item[0] for item in parsed])
    delta_points_mm = np.array([item[1] for item in parsed])
    transform, residuals = fit_rigid_transform(camera_points_mm, delta_points_mm)

    result = copy.deepcopy(source_data)
    result['source_calibration_path'] = str(source_path)
    result['subset'] = f'z{abs(int(round(z_mm)))}'
    result['samples'] = samples
    result['sample_count'] = len(samples)
    result['T_delta_camera'] = [[float(value) for value in row] for row in transform]
    result['fit_error'] = {
        'rmse_mm': float(math.sqrt(float(np.mean(residuals ** 2)))),
        'mean_mm': float(np.mean(residuals)),
        'max_mm': float(np.max(residuals)),
    }

    output_path = output_dir / f'{prefix}_z{abs(int(round(z_mm)))}.yaml'
    output_path.write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True), encoding='utf-8')
    return output_path, result['fit_error'], len(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_yaml', type=Path)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--prefix', default=None)
    parser.add_argument('--z-tolerance-mm', type=float, default=1.0)
    args = parser.parse_args()

    source_path = args.input_yaml.expanduser()
    data = yaml.safe_load(source_path.read_text(encoding='utf-8')) or {}
    output_dir = (args.output_dir or source_path.parent).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or source_path.stem

    layers = split_by_z(data, args.z_tolerance_mm)
    if not layers:
        raise RuntimeError(f'no valid samples found in {source_path}')

    summary_lines = [f'source: {source_path}', '']
    for z_mm in sorted(layers):
        result = write_layer(data, source_path, output_dir, prefix, z_mm, layers[z_mm])
        if result is None:
            summary_lines.append(f'z={z_mm}: skipped, fewer than 3 valid samples')
            continue
        output_path, error, count = result
        summary_lines.append(
            '%s: n=%d, rmse=%.2f mm, mean=%.2f mm, max=%.2f mm'
            % (output_path.name, count, error['rmse_mm'], error['mean_mm'], error['max_mm'])
        )

    summary_path = output_dir / f'{prefix}_z_summary.txt'
    summary_path.write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    print(summary_path)
    print('\n'.join(summary_lines))


if __name__ == '__main__':
    main()
