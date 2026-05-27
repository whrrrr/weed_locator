#!/usr/bin/env python3
"""Fit an empirical camera_xyz -> delta_xyz model from ChArUco samples."""

import argparse
import math
from pathlib import Path

import numpy as np
import yaml


def sample_points(sample):
    camera = sample.get('camera_xyz_m') or sample.get('camera_xyz')
    delta = sample.get('delta_xyz_mm') or sample.get('robot_xyz_mm') or sample.get('delta_xyz')
    if camera is None or delta is None:
        return None
    return np.array(camera, dtype=float) * 1000.0, np.array(delta, dtype=float)


def polynomial_terms(points_norm, degree):
    x = points_norm[:, 0]
    y = points_norm[:, 1]
    z = points_norm[:, 2]
    columns = [
        np.ones(points_norm.shape[0]),
        x,
        y,
        z,
    ]
    names = ['1', 'x', 'y', 'z']
    if degree >= 2:
        columns.extend([x * x, y * y, z * z, x * y, x * z, y * z])
        names.extend(['x2', 'y2', 'z2', 'xy', 'xz', 'yz'])
    if degree >= 3:
        columns.extend([
            x * x * x,
            y * y * y,
            z * z * z,
            x * x * y,
            x * x * z,
            y * y * x,
            y * y * z,
            z * z * x,
            z * z * y,
            x * y * z,
        ])
        names.extend(['x3', 'y3', 'z3', 'x2y', 'x2z', 'y2x', 'y2z', 'z2x', 'z2y', 'xyz'])
    return np.column_stack(columns), names


def design_matrix(camera_points_mm, mean_mm, scale_mm, degree):
    normalized = (camera_points_mm - mean_mm) / scale_mm
    return polynomial_terms(normalized, degree)


def fit_model(camera_points_mm, delta_points_mm, degree, ridge):
    mean_mm = camera_points_mm.mean(axis=0)
    scale_mm = camera_points_mm.std(axis=0)
    scale_mm = np.where(scale_mm < 1e-6, 1.0, scale_mm)
    phi, term_names = design_matrix(camera_points_mm, mean_mm, scale_mm, degree)

    if ridge > 0.0:
        regularizer = ridge * np.eye(phi.shape[1])
        regularizer[0, 0] = 0.0
        coefficients = np.linalg.solve(phi.T @ phi + regularizer, phi.T @ delta_points_mm)
    else:
        coefficients, *_ = np.linalg.lstsq(phi, delta_points_mm, rcond=None)

    predicted = phi @ coefficients
    residuals = predicted - delta_points_mm
    errors = np.linalg.norm(residuals, axis=1)
    return {
        'degree': degree,
        'ridge': ridge,
        'camera_mean_mm': mean_mm,
        'camera_scale_mm': scale_mm,
        'term_names': term_names,
        'coefficients': coefficients,
        'predicted': predicted,
        'residuals': residuals,
        'errors': errors,
    }


def leave_one_out(camera_points_mm, delta_points_mm, degree, ridge):
    if len(camera_points_mm) < 8:
        return None

    predictions = []
    actuals = []
    for index in range(len(camera_points_mm)):
        mask = np.ones(len(camera_points_mm), dtype=bool)
        mask[index] = False
        model = fit_model(camera_points_mm[mask], delta_points_mm[mask], degree, ridge)
        phi, _ = design_matrix(
            camera_points_mm[index:index + 1],
            model['camera_mean_mm'],
            model['camera_scale_mm'],
            degree,
        )
        predictions.append((phi @ model['coefficients'])[0])
        actuals.append(delta_points_mm[index])

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    errors = np.linalg.norm(predictions - actuals, axis=1)
    return error_summary(errors)


def error_summary(errors):
    return {
        'rmse_mm': float(math.sqrt(float(np.mean(errors ** 2)))),
        'mean_mm': float(np.mean(errors)),
        'median_mm': float(np.median(errors)),
        'max_mm': float(np.max(errors)),
    }


def load_samples(path):
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    parsed = []
    for sample in data.get('samples', []):
        points = sample_points(sample)
        if points is not None:
            parsed.append(points)
    if not parsed:
        raise RuntimeError(f'no usable samples in {path}')
    camera_points_mm = np.array([item[0] for item in parsed], dtype=float)
    delta_points_mm = np.array([item[1] for item in parsed], dtype=float)
    return data, camera_points_mm, delta_points_mm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_yaml', type=Path)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--degree', type=int, default=2, choices=[1, 2, 3])
    parser.add_argument('--ridge', type=float, default=1e-6)
    parser.add_argument('--no-loocv', action='store_true')
    args = parser.parse_args()

    source_path = args.input_yaml.expanduser()
    output_path = args.output.expanduser() if args.output else source_path.with_name(f'{source_path.stem}_poly{args.degree}_model.yaml')
    _data, camera_points_mm, delta_points_mm = load_samples(source_path)

    model = fit_model(camera_points_mm, delta_points_mm, args.degree, args.ridge)
    train_error = error_summary(model['errors'])
    loocv_error = None if args.no_loocv else leave_one_out(camera_points_mm, delta_points_mm, args.degree, args.ridge)

    output = {
        'description': 'Empirical polynomial model: delta_xyz_mm = f(camera_xyz_mm)',
        'source_calibration_path': str(source_path),
        'sample_count': int(len(camera_points_mm)),
        'input_units': 'camera_xyz_mm',
        'output_units': 'delta_xyz_mm',
        'model_type': 'polynomial',
        'degree': int(args.degree),
        'ridge': float(args.ridge),
        'camera_mean_mm': [float(value) for value in model['camera_mean_mm']],
        'camera_scale_mm': [float(value) for value in model['camera_scale_mm']],
        'term_names': model['term_names'],
        'coefficients': [[float(value) for value in row] for row in model['coefficients']],
        'train_error': train_error,
        'loocv_error': loocv_error,
        'camera_bounds_mm': {
            'min': [float(value) for value in camera_points_mm.min(axis=0)],
            'max': [float(value) for value in camera_points_mm.max(axis=0)],
        },
        'delta_bounds_mm': {
            'min': [float(value) for value in delta_points_mm.min(axis=0)],
            'max': [float(value) for value in delta_points_mm.max(axis=0)],
        },
    }
    output_path.write_text(yaml.safe_dump(output, sort_keys=False, allow_unicode=True), encoding='utf-8')

    print(output_path)
    print('samples:', len(camera_points_mm))
    print('train: rmse={rmse_mm:.2f} mean={mean_mm:.2f} median={median_mm:.2f} max={max_mm:.2f} mm'.format(**train_error))
    if loocv_error:
        print('loocv: rmse={rmse_mm:.2f} mean={mean_mm:.2f} median={median_mm:.2f} max={max_mm:.2f} mm'.format(**loocv_error))


if __name__ == '__main__':
    main()
