#!/usr/bin/env python3
"""Keyboard tuner for camera track calibration lines."""

import select
import sys
import termios
import tty

import rclpy
from rcl_interfaces.msg import Parameter, ParameterType
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node


class TrackLineTuner(Node):
    def __init__(self):
        super().__init__('track_line_tuner')

        self.declare_parameter('top_line_y', 45.0)
        self.declare_parameter('bottom_line_y', 460.0)
        self.declare_parameter('step', 5.0)
        self.declare_parameter('fine_step', 1.0)
        self.declare_parameter('nodes', ['chess_detector', 'conveyor_pick_scheduler'])

        self.top_line_y = float(self.get_parameter('top_line_y').value)
        self.bottom_line_y = float(self.get_parameter('bottom_line_y').value)
        self.step = float(self.get_parameter('step').value)
        self.fine_step = float(self.get_parameter('fine_step').value)
        self.node_names = list(self.get_parameter('nodes').value)

        self.param_clients = {
            name: self.create_client(SetParameters, f'/{name}/set_parameters')
            for name in self.node_names
        }

        self.print_help()
        self.push_params()

    def print_help(self):
        print('')
        print('Track line tuner')
        print('  w/s: top line up/down')
        print('  i/k: bottom line up/down')
        print('  u/j: both lines up/down')
        print('  a/d: fine top up/down')
        print('  h/l: fine bottom up/down')
        print('  p: print current lines')
        print('  q: quit')
        print('')

    def spin_keyboard(self):
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.05)
                if not select.select([sys.stdin], [], [], 0.0)[0]:
                    continue
                key = sys.stdin.read(1)
                if key == 'q':
                    break
                self.handle_key(key)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def handle_key(self, key):
        changed = True
        if key == 'w':
            self.top_line_y -= self.step
        elif key == 's':
            self.top_line_y += self.step
        elif key == 'i':
            self.bottom_line_y -= self.step
        elif key == 'k':
            self.bottom_line_y += self.step
        elif key == 'u':
            self.top_line_y -= self.step
            self.bottom_line_y -= self.step
        elif key == 'j':
            self.top_line_y += self.step
            self.bottom_line_y += self.step
        elif key == 'a':
            self.top_line_y -= self.fine_step
        elif key == 'd':
            self.top_line_y += self.fine_step
        elif key == 'h':
            self.bottom_line_y -= self.fine_step
        elif key == 'l':
            self.bottom_line_y += self.fine_step
        elif key == 'p':
            changed = False
        else:
            changed = False

        self.clamp_lines()
        self.print_status()
        if changed:
            self.push_params()

    def clamp_lines(self):
        self.top_line_y = max(0.0, self.top_line_y)
        self.bottom_line_y = max(self.top_line_y + 1.0, self.bottom_line_y)

    def print_status(self):
        print(
            f'top_line_y={self.top_line_y:.1f}, '
            f'bottom_line_y={self.bottom_line_y:.1f}, '
            f'pixel_span={self.bottom_line_y - self.top_line_y:.1f}'
        )

    def push_params(self):
        for name, client in self.param_clients.items():
            if not client.wait_for_service(timeout_sec=0.1):
                self.get_logger().warn(f'参数服务暂不可用: /{name}/set_parameters')
                continue

            req = SetParameters.Request()
            req.parameters = [
                self.make_double_param('top_line_y', self.top_line_y),
                self.make_double_param('bottom_line_y', self.bottom_line_y),
            ]
            future = client.call_async(req)
            future.add_done_callback(lambda fut, node_name=name: self.on_set_done(node_name, fut))

    def on_set_done(self, node_name, future):
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().warn(f'更新 {node_name} 参数失败: {exc}')
            return

        failed = [r.reason for r in result.results if not r.successful]
        if failed:
            self.get_logger().warn(f'更新 {node_name} 参数被拒绝: {failed}')

    @staticmethod
    def make_double_param(name, value):
        param = Parameter()
        param.name = name
        param.value.type = ParameterType.PARAMETER_DOUBLE
        param.value.double_value = float(value)
        return param


def main(args=None):
    rclpy.init(args=args)
    node = TrackLineTuner()
    try:
        node.spin_keyboard()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
