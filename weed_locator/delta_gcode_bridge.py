#!/usr/bin/env python3
"""
Delta ESP32 G-code bridge node.

Responsibilities:
- Open the ESP32 serial port
- Publish serial feedback to ROS
- Accept raw G-code and basic motion/device commands from ROS topics
"""

import time

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, String

try:
    import serial
    from serial import SerialException
except ImportError:  # pragma: no cover
    serial = None
    SerialException = Exception


class DeltaGcodeBridge(Node):
    """ROS2 <-> ESP32 G-code serial bridge."""

    def __init__(self):
        super().__init__('delta_gcode_bridge')

        self.declare_parameter(
            'port',
            '/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0',
        )
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('read_timeout', 0.05)
        self.declare_parameter('startup_delay_sec', 2.0)
        self.declare_parameter('default_feedrate', 80.0)
        self.declare_parameter('command_interval_sec', 0.05)
        self.declare_parameter('auto_absolute_mode', True)

        self.port = self.get_parameter('port').value
        self.baudrate = int(self.get_parameter('baudrate').value)
        self.read_timeout = float(self.get_parameter('read_timeout').value)
        self.startup_delay_sec = float(self.get_parameter('startup_delay_sec').value)
        self.default_feedrate = float(self.get_parameter('default_feedrate').value)
        self.command_interval_sec = float(self.get_parameter('command_interval_sec').value)
        self.auto_absolute_mode = bool(self.get_parameter('auto_absolute_mode').value)

        self.serial_conn = None
        self.last_write_time = 0.0

        self.status_pub = self.create_publisher(String, '/delta_arm/status', 20)
        self.gcode_echo_pub = self.create_publisher(String, '/delta_arm/last_gcode', 20)

        self.create_subscription(String, '/delta_arm/gcode_raw', self.on_raw_gcode, 20)
        self.create_subscription(Point, '/delta_arm/move_to', self.on_move_to, 20)
        self.create_subscription(Bool, '/delta_arm/pump_cmd', self.on_pump_cmd, 20)
        self.create_subscription(Bool, '/delta_arm/valve_cmd', self.on_valve_cmd, 20)
        self.create_subscription(Bool, '/delta_arm/motor_enable', self.on_motor_enable_cmd, 20)
        self.create_subscription(Empty, '/delta_arm/home', self.on_home_cmd, 20)

        self.read_timer = self.create_timer(0.02, self.poll_serial)

        self.open_serial()

        self.get_logger().info('delta_gcode_bridge 已启动')
        self.get_logger().info(f'订阅: /delta_arm/gcode_raw, /delta_arm/move_to, /delta_arm/pump_cmd, /delta_arm/valve_cmd, /delta_arm/motor_enable, /delta_arm/home')
        self.get_logger().info('发布: /delta_arm/status, /delta_arm/last_gcode')

    def open_serial(self):
        """Open the serial port to the ESP32."""
        if serial is None:
            self.get_logger().error('未安装 pyserial，无法打开串口')
            return

        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=self.read_timeout)
            self.get_logger().info(f'已打开串口 {self.port} @ {self.baudrate}')
            time.sleep(self.startup_delay_sec)
            self.flush_input()
        except SerialException as exc:
            self.serial_conn = None
            self.get_logger().error(f'打开串口失败: {exc}')

    def flush_input(self):
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.reset_input_buffer()
            except SerialException as exc:
                self.get_logger().warning(f'清空串口输入缓冲失败: {exc}')

    def publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def publish_last_gcode(self, text: str):
        msg = String()
        msg.data = text
        self.gcode_echo_pub.publish(msg)

    def ensure_command_gap(self):
        delta = time.time() - self.last_write_time
        if delta < self.command_interval_sec:
            time.sleep(self.command_interval_sec - delta)

    def send_line(self, line: str):
        """Send one G-code line terminated with CR."""
        if not line:
            return False

        if not self.serial_conn or not self.serial_conn.is_open:
            self.get_logger().error('串口未打开，无法发送 G-code')
            return False

        try:
            self.ensure_command_gap()
            payload = line.strip().replace('\n', '').replace('\r', '') + '\r'
            self.serial_conn.write(payload.encode('utf-8'))
            self.serial_conn.flush()
            self.last_write_time = time.time()
            self.publish_last_gcode(payload.strip())
            self.get_logger().info(f'发送: {payload.strip()}')
            return True
        except SerialException as exc:
            self.get_logger().error(f'发送串口数据失败: {exc}')
            return False

    def send_gcode_block(self, block: str):
        """Send a multi-line G-code block."""
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        for line in lines:
            if not self.send_line(line):
                break

    def on_raw_gcode(self, msg: String):
        self.send_gcode_block(msg.data)

    def on_move_to(self, msg: Point):
        lines = []
        if self.auto_absolute_mode:
            lines.append('G90')
        lines.append(
            f'G1 X{msg.x:.2f} Y{msg.y:.2f} Z{msg.z:.2f} F{self.default_feedrate:.2f}'
        )
        self.send_gcode_block('\n'.join(lines))

    def on_pump_cmd(self, msg: Bool):
        self.send_line('M121' if msg.data else 'M122')

    def on_valve_cmd(self, msg: Bool):
        self.send_line('M1' if msg.data else 'M2')

    def on_motor_enable_cmd(self, msg: Bool):
        self.send_line('M17' if msg.data else 'M18')

    def on_home_cmd(self, _msg: Empty):
        self.send_line('G28')

    def poll_serial(self):
        """Read and republish any serial feedback from the ESP32."""
        if not self.serial_conn or not self.serial_conn.is_open:
            return

        try:
            waiting = self.serial_conn.in_waiting
            if waiting <= 0:
                return

            raw = self.serial_conn.read(waiting).decode('utf-8', errors='ignore')
            for line in raw.splitlines():
                text = line.strip()
                if not text:
                    continue
                self.publish_status(text)
                self.get_logger().info(f'ESP32: {text}')
        except SerialException as exc:
            self.get_logger().error(f'读取串口数据失败: {exc}')

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.get_logger().info('已关闭 ESP32 串口')


def main(args=None):
    rclpy.init(args=args)
    node = DeltaGcodeBridge()

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
