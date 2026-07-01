#!/usr/bin/env python3
"""Keyboard tuner for SO101 joint-state display offsets.

This node never commands the robot. It subscribes to real /joint_states,
adds per-joint display offsets, and republishes /display_joint_states for
robot_state_publisher/RViz.
"""

import math
import select
import sys
import termios
import threading
import tty
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState


DEFAULT_JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex']


def parse_joint_list(raw):
    return [item.strip() for item in str(raw).split(',') if item.strip()]


class LeRobotJointOffsetTuner(Node):
    def __init__(self):
        super().__init__('lerobot_joint_offset_tuner')
        self.declare_parameter('input_topic', '/joint_states')
        self.declare_parameter('output_topic', '/display_joint_states')
        self.declare_parameter('joints', ','.join(DEFAULT_JOINTS))
        self.declare_parameter('joint_name_map', 'elbow_flex:wrist_flex')
        self.declare_parameter(
            'save_path',
            '/home/whr/cc_ws/tros_ws/calibration_targets/so101_joint_display_offsets.yaml',
        )
        self.declare_parameter('step_deg', 1.0)
        self.declare_parameter('fine_step_deg', 0.2)
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('load_on_start', True)

        self.joints = parse_joint_list(self.get_parameter('joints').value)
        self.name_map = self.parse_name_map(self.get_parameter('joint_name_map').value)
        self.offsets_deg = {joint: 0.0 for joint in self.joints}
        self.selected_index = 0
        self.latest_msg = None
        self.latest_adjusted_msg = None
        self.lock = threading.RLock()

        if bool(self.get_parameter('load_on_start').value):
            self.load_offsets(silent=True)

        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        self.sub = self.create_subscription(JointState, input_topic, self.joint_state_callback, 10)
        self.pub = self.create_publisher(JointState, output_topic, 10)
        rate = float(self.get_parameter('publish_rate').value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.publish_adjusted)

        self.print_help()

    def parse_name_map(self, raw):
        mapping = {}
        for item in str(raw).split(','):
            item = item.strip()
            if not item:
                continue
            if ':' not in item:
                self.get_logger().warning(f'ignoring malformed joint_name_map item: {item!r}')
                continue
            source, target = [part.strip() for part in item.split(':', 1)]
            if source and target:
                mapping[source] = target
        return mapping

    def joint_state_callback(self, msg):
        with self.lock:
            self.latest_msg = msg

    def publish_adjusted(self):
        with self.lock:
            if self.latest_msg is None:
                return
            msg = JointState()
            msg.header = self.latest_msg.header
            msg.header.stamp = self.get_clock().now().to_msg()

            mapped_targets = set(self.name_map.values())
            names = []
            positions = []
            velocities = []
            efforts = []
            used_names = set()
            for idx, source_name in enumerate(self.latest_msg.name):
                # If a real joint is remapped onto a URDF joint name, drop the
                # original joint with that same URDF name. In this modified
                # SO101, real elbow_flex drives URDF wrist_flex, while the real
                # wrist_flex motor is not part of the kinematic model.
                if source_name in mapped_targets and source_name not in self.name_map:
                    continue

                display_name = self.name_map.get(source_name, source_name)
                if display_name in used_names:
                    continue
                used_names.add(display_name)
                names.append(display_name)
                positions.append(self.latest_msg.position[idx])
                if idx < len(self.latest_msg.velocity):
                    velocities.append(self.latest_msg.velocity[idx])
                if idx < len(self.latest_msg.effort):
                    efforts.append(self.latest_msg.effort[idx])

            msg.name = names
            msg.position = positions
            msg.velocity = velocities
            msg.effort = efforts

            name_to_index = {name: idx for idx, name in enumerate(msg.name)}
            for joint, offset_deg in self.offsets_deg.items():
                display_joint = self.name_map.get(joint, joint)
                idx = name_to_index.get(display_joint)
                if idx is not None and idx < len(msg.position):
                    msg.position[idx] += math.radians(offset_deg)
            self.latest_adjusted_msg = msg
            self.pub.publish(msg)

    def selected_joint(self):
        if not self.joints:
            return ''
        return self.joints[self.selected_index % len(self.joints)]

    def adjust_selected(self, delta_deg):
        joint = self.selected_joint()
        if not joint:
            return
        self.offsets_deg[joint] = self.offsets_deg.get(joint, 0.0) + delta_deg
        self.print_status()

    def print_help(self):
        print()
        print('LeRobot joint offset tuner')
        print('  It reads /joint_states and publishes adjusted /display_joint_states for RViz.')
        print('  1/2/3...: select joint')
        print('  a/d or -/+: offset -/+ step')
        print('  z/x: offset -/+ fine step')
        print('  0: reset selected joint offset')
        print('  R: reset all offsets')
        print('  p: save offsets')
        print('  l: load offsets')
        print('  h: help')
        print('  q: quit')
        self.print_status()

    def print_status(self):
        step = float(self.get_parameter('step_deg').value)
        fine = float(self.get_parameter('fine_step_deg').value)
        selected = self.selected_joint()
        parts = []
        for idx, joint in enumerate(self.joints):
            mark = '*' if joint == selected else ' '
            parts.append(f'{mark}{idx + 1}:{joint}={self.offsets_deg.get(joint, 0.0):+.2f}deg')
        print('offsets:', ' | '.join(parts), f'  step={step:g}deg fine={fine:g}deg')

    def save_offsets(self):
        path = Path(str(self.get_parameter('save_path').value)).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'description': 'SO101 display/ROS joint offsets. adjusted = raw + offset_deg',
            'input_topic': str(self.get_parameter('input_topic').value),
            'output_topic': str(self.get_parameter('output_topic').value),
            'joint_name_map': dict(self.name_map),
            'offsets_deg': {joint: float(value) for joint, value in self.offsets_deg.items()},
        }
        with self.lock:
            if self.latest_msg is not None:
                data['raw_input_pose'] = self.joint_state_to_yaml(self.latest_msg)
            if self.latest_adjusted_msg is not None:
                data['adjusted_display_pose'] = self.joint_state_to_yaml(self.latest_adjusted_msg)
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')
        print(f'saved offsets: {path}')

    def joint_state_to_yaml(self, msg):
        joints = []
        for idx, name in enumerate(msg.name):
            if idx >= len(msg.position):
                continue
            rad = float(msg.position[idx])
            joints.append({
                'name': name,
                'position_rad': rad,
                'position_deg': math.degrees(rad),
            })
        return {
            'names': list(msg.name),
            'joints': joints,
        }

    def load_offsets(self, silent=False):
        path = Path(str(self.get_parameter('save_path').value)).expanduser()
        if not path.exists():
            if not silent:
                print(f'offset file does not exist: {path}')
            return
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        loaded = data.get('offsets_deg', {})
        for joint in self.joints:
            if joint in loaded:
                self.offsets_deg[joint] = float(loaded[joint])
        if not silent:
            print(f'loaded offsets: {path}')
            self.print_status()

    def handle_key(self, key):
        if key == 'q':
            return False
        if key == 'h':
            self.print_help()
            return True
        if key == 'p':
            self.save_offsets()
            return True
        if key == 'l':
            self.load_offsets()
            return True
        if key == 'R':
            for joint in self.joints:
                self.offsets_deg[joint] = 0.0
            self.print_status()
            return True
        if key == '0':
            self.offsets_deg[self.selected_joint()] = 0.0
            self.print_status()
            return True
        if key.isdigit():
            idx = int(key) - 1
            if 0 <= idx < len(self.joints):
                self.selected_index = idx
                self.print_status()
            return True

        step = float(self.get_parameter('step_deg').value)
        fine = float(self.get_parameter('fine_step_deg').value)
        if key in ('d', '+', '='):
            self.adjust_selected(step)
        elif key in ('a', '-'):
            self.adjust_selected(-step)
        elif key == 'x':
            self.adjust_selected(fine)
        elif key == 'z':
            self.adjust_selected(-fine)
        return True

    def spin_keyboard(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.02)
                ready, _, _ = select.select([sys.stdin], [], [], 0.02)
                if not ready:
                    continue
                key = sys.stdin.read(1)
                if not self.handle_key(key):
                    break
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def main(args=None):
    rclpy.init(args=args)
    node = LeRobotJointOffsetTuner()
    try:
        node.spin_keyboard()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
