#!/usr/bin/env python3
"""ROS 2 node for hobby-servo PWM angle control on RDK X5."""

import math
import time
from pathlib import Path

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from std_msgs.msg import Float32


PWM_MAP = {
    33: ('/sys/class/pwm/pwmchip0', 1),
    32: ('/sys/class/pwm/pwmchip0', 0),
}


class KS3518PwmServoNode(Node):
    """Drive a 50 Hz hobby servo with RDK sysfs hardware PWM."""

    def __init__(self):
        super().__init__('ks3518_pwm_servo')

        self.declare_parameter('pin', 33)
        self.declare_parameter('frequency_hz', 50.0)
        self.declare_parameter('min_angle_deg', 0.0)
        self.declare_parameter('max_angle_deg', 180.0)
        self.declare_parameter('min_pulse_ms', 0.5)
        self.declare_parameter('max_pulse_ms', 2.5)
        self.declare_parameter('angle', 90.0)
        self.declare_parameter('angle_topic', '/ks3518/angle')
        self.declare_parameter('refresh_rate_hz', 10.0)
        self.declare_parameter('signal_high_voltage', 3.3)

        self.pin = int(self.get_parameter('pin').value)
        self.frequency_hz = float(self.get_parameter('frequency_hz').value)
        self.min_angle_deg = float(self.get_parameter('min_angle_deg').value)
        self.max_angle_deg = float(self.get_parameter('max_angle_deg').value)
        self.min_pulse_ms = float(self.get_parameter('min_pulse_ms').value)
        self.max_pulse_ms = float(self.get_parameter('max_pulse_ms').value)
        self.angle_topic = str(self.get_parameter('angle_topic').value)
        self.refresh_rate_hz = float(self.get_parameter('refresh_rate_hz').value)
        self.signal_high_voltage = float(self.get_parameter('signal_high_voltage').value)

        self._validate_config()

        self.pwm_path = self._export_pwm(self.pin)
        self.period_ns = int(round(1_000_000_000.0 / self.frequency_hz))
        self.current_angle = None

        initial_angle = float(self.get_parameter('angle').value)
        self._initialize_pwm()
        self.current_angle = self._clamp_angle(initial_angle)
        self._write_angle_pwm(self.current_angle)

        self.state_pub = self.create_publisher(Float32, '/ks3518/state_angle', 10)
        self.angle_sub = self.create_subscription(
            Float32,
            self.angle_topic,
            self._angle_callback,
            10,
        )
        self.add_on_set_parameters_callback(self._parameters_callback)
        self.refresh_timer = None
        if self.refresh_rate_hz > 0.0:
            self.refresh_timer = self.create_timer(
                1.0 / self.refresh_rate_hz,
                self._refresh_pwm,
            )

        self.get_logger().info(
            f'KS-3518 PWM servo ready: BOARD pin {self.pin}, '
            f'{self.frequency_hz:.1f} Hz, angle {self.current_angle:.1f} deg, '
            f'sysfs={self.pwm_path}'
        )
        self.get_logger().info(
            'Command with: ros2 topic pub --once '
            f'{self.angle_topic} std_msgs/msg/Float32 "{{data: 90.0}}"'
        )
        self._log_pwm_info(self.current_angle)

    def _validate_config(self):
        if self.pin not in PWM_MAP:
            raise ValueError(f'pin must be one of {sorted(PWM_MAP)}')
        if self.frequency_hz <= 0:
            raise ValueError('frequency_hz must be positive')
        if self.max_angle_deg <= self.min_angle_deg:
            raise ValueError('max_angle_deg must be greater than min_angle_deg')
        if self.max_pulse_ms <= self.min_pulse_ms:
            raise ValueError('max_pulse_ms must be greater than min_pulse_ms')

        period_ms = 1000.0 / self.frequency_hz
        if self.max_pulse_ms >= period_ms:
            raise ValueError('max_pulse_ms must be shorter than the PWM period')

    def _write_text(self, path, value):
        Path(path).write_text(str(value))

    def _read_text(self, path):
        return Path(path).read_text().strip()

    def _export_pwm(self, pin):
        chip_path, channel = PWM_MAP[pin]
        chip = Path(chip_path)
        pwm_path = chip / f'pwm{channel}'
        if not pwm_path.exists():
            self._write_text(chip / 'export', channel)
            deadline = time.monotonic() + 2.0
            while not pwm_path.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f'timeout exporting {pwm_path}')
                time.sleep(0.01)
        return pwm_path

    def _initialize_pwm(self):
        self._write_text(self.pwm_path / 'enable', 0)
        self._write_text(self.pwm_path / 'duty_cycle', 1)
        self._write_text(self.pwm_path / 'period', self.period_ns)
        self._write_text(self.pwm_path / 'enable', 1)
        time.sleep(0.05)

    def _clamp_angle(self, angle_deg):
        return max(self.min_angle_deg, min(self.max_angle_deg, float(angle_deg)))

    def _angle_to_pulse_ms(self, angle_deg):
        angle = self._clamp_angle(angle_deg)
        span = self.max_angle_deg - self.min_angle_deg
        ratio = (angle - self.min_angle_deg) / span
        return self.min_pulse_ms + ratio * (self.max_pulse_ms - self.min_pulse_ms)

    def _angle_to_duty_cycle(self, angle_deg):
        pulse_ms = self._angle_to_pulse_ms(angle_deg)
        period_ms = 1000.0 / self.frequency_hz
        return (pulse_ms / period_ms) * 100.0

    def _angle_to_duty_ns(self, angle_deg):
        pulse_ms = self._angle_to_pulse_ms(angle_deg)
        return max(1, int(round(pulse_ms * 1_000_000.0)))

    def _write_angle_pwm(self, angle_deg):
        duty_ns = self._angle_to_duty_ns(angle_deg)
        # Keep PWM enabled and update duty in-place; this avoids one-step
        # delayed output on the RDK PWM driver.
        self._write_text(self.pwm_path / 'duty_cycle', duty_ns)
        time.sleep(0.01)
        self._write_text(self.pwm_path / 'duty_cycle', duty_ns)

    def set_angle(self, angle_deg):
        if not math.isfinite(float(angle_deg)):
            raise ValueError('angle must be finite')

        angle = self._clamp_angle(angle_deg)
        self._write_angle_pwm(angle)
        self.current_angle = angle

        msg = Float32()
        msg.data = float(angle)
        self.state_pub.publish(msg)

        self._log_pwm_info(angle)

    def _log_pwm_info(self, angle):
        duty_cycle = self._angle_to_duty_cycle(angle)
        pulse_ms = self._angle_to_pulse_ms(angle)
        average_voltage = self.signal_high_voltage * duty_cycle / 100.0
        period_ms = 1000.0 / self.frequency_hz
        duty_ns = self._read_text(self.pwm_path / 'duty_cycle')
        enable = self._read_text(self.pwm_path / 'enable')
        self.get_logger().info(
            f'angle={angle:.1f} deg, period={period_ms:.3f} ms, '
            f'pulse_high={pulse_ms:.3f} ms, duty={duty_cycle:.3f}%, '
            f'estimated_multimeter_dc={average_voltage:.3f} V '
            f'(high={self.signal_high_voltage:.1f} V), '
            f'sysfs_duty_ns={duty_ns}, enable={enable}'
        )

    def _refresh_pwm(self):
        if self.current_angle is None:
            return
        self._write_angle_pwm(self.current_angle)

    def _angle_callback(self, msg):
        try:
            self.set_angle(msg.data)
        except ValueError as exc:
            self.get_logger().error(str(exc))

    def _parameters_callback(self, params):
        for param in params:
            if param.name == 'angle':
                try:
                    self.set_angle(param.value)
                except ValueError as exc:
                    return SetParametersResult(successful=False, reason=str(exc))
        return SetParametersResult(successful=True)

    def destroy_node(self):
        try:
            if hasattr(self, 'pwm_path'):
                self._write_text(self.pwm_path / 'duty_cycle', 1)
                self._write_text(self.pwm_path / 'enable', 1)
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KS3518PwmServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
