#!/home/whr/miniconda3/envs/lerobot/bin/python
"""Tune real SO101 joint angles and RViz display offsets in one keyboard node."""

import math
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState

from weed_locator.srv import WriteJoints


DEFAULT_TUNE_JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex']
DEFAULT_COMMAND_JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex']


def parse_joint_list(raw):
    return [item.strip() for item in str(raw).split(',') if item.strip()]


class LeRobotRealSimAngleTuner(Node):
    """One node for two separate actions:

    - sim/display tuning: add per-joint offsets and publish /display_joint_states
    - real tuning: send small relative commands through /lerobot/write_joints
    """

    def __init__(self):
        super().__init__('lerobot_real_sim_angle_tuner')

        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('display_joint_state_topic', '/display_joint_states')
        self.declare_parameter('write_joints_service', '/lerobot/write_joints')
        self.declare_parameter('save_path', '/home/whr/cc_ws/tros_ws/calibration_targets/so101_joint_display_offsets.yaml')
        self.declare_parameter('load_on_start', True)
        self.declare_parameter('tune_joints', ','.join(DEFAULT_TUNE_JOINTS))
        self.declare_parameter('command_joints', ','.join(DEFAULT_COMMAND_JOINTS))
        self.declare_parameter('joint_name_map', 'elbow_flex:wrist_flex')
        self.declare_parameter('sim_step_deg', 1.0)
        self.declare_parameter('sim_fine_step_deg', 0.2)
        self.declare_parameter('real_step_deg', 1.0)
        self.declare_parameter('real_fine_step_deg', 0.2)
        self.declare_parameter('max_real_step_deg', 5.0)
        self.declare_parameter('max_pending_real_error_deg', 8.0)
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('service_timeout_sec', 2.0)

        self.tune_joints = parse_joint_list(self.get_parameter('tune_joints').value)
        self.command_joints = parse_joint_list(self.get_parameter('command_joints').value)
        self.name_map = self.parse_name_map(self.get_parameter('joint_name_map').value)
        self.offsets_deg = {joint: 0.0 for joint in self.tune_joints}
        self.selected_index = 0
        self.latest_msg = None
        self.latest_adjusted_msg = None
        self.command_targets_deg = None
        self.last_real_command_time = 0.0
        self.lock = threading.RLock()

        if bool(self.get_parameter('load_on_start').value):
            self.load_offsets(silent=True)

        self.sub = self.create_subscription(
            JointState,
            str(self.get_parameter('joint_state_topic').value),
            self.on_joint_state,
            10,
        )
        self.pub = self.create_publisher(
            JointState,
            str(self.get_parameter('display_joint_state_topic').value),
            10,
        )
        self.write_client = self.create_client(
            WriteJoints,
            str(self.get_parameter('write_joints_service').value),
        )
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

    def on_joint_state(self, msg):
        with self.lock:
            self.latest_msg = msg

    def selected_joint(self):
        if not self.tune_joints:
            return ''
        return self.tune_joints[self.selected_index % len(self.tune_joints)]

    def publish_adjusted(self):
        with self.lock:
            if self.latest_msg is None:
                return
            msg = self.adjusted_joint_state(self.latest_msg)
            self.latest_adjusted_msg = msg
        self.pub.publish(msg)

    def adjusted_joint_state(self, source_msg):
        msg = JointState()
        msg.header = source_msg.header
        msg.header.stamp = self.get_clock().now().to_msg()

        mapped_targets = set(self.name_map.values())
        names = []
        positions = []
        velocities = []
        efforts = []
        used_names = set()
        for idx, source_name in enumerate(source_msg.name):
            # If real elbow_flex is displayed as URDF wrist_flex, drop the real
            # wrist_flex motor from the display chain. It is not part of IK.
            if source_name in mapped_targets and source_name not in self.name_map:
                continue
            display_name = self.name_map.get(source_name, source_name)
            if display_name in used_names:
                continue
            used_names.add(display_name)
            names.append(display_name)
            positions.append(float(source_msg.position[idx]))
            if idx < len(source_msg.velocity):
                velocities.append(float(source_msg.velocity[idx]))
            if idx < len(source_msg.effort):
                efforts.append(float(source_msg.effort[idx]))

        msg.name = names
        msg.position = positions
        msg.velocity = velocities
        msg.effort = efforts

        name_to_index = {name: idx for idx, name in enumerate(msg.name)}
        for joint, offset_deg in self.offsets_deg.items():
            display_joint = self.name_map.get(joint, joint)
            idx = name_to_index.get(display_joint)
            if idx is not None and idx < len(msg.position):
                msg.position[idx] += math.radians(float(offset_deg))
        return msg

    def latest_positions_by_name(self):
        with self.lock:
            msg = self.latest_msg
            if msg is None:
                return None
            return {
                name: float(msg.position[idx])
                for idx, name in enumerate(msg.name)
                if idx < len(msg.position)
            }

    def sync_real_targets_to_current(self):
        positions = self.latest_positions_by_name()
        if positions is None:
            print('cannot sync real targets: no /joint_states yet')
            return False
        missing = [name for name in self.command_joints if name not in positions]
        if missing:
            print(f'cannot sync real targets: /joint_states missing command joints: {missing}')
            return False
        self.command_targets_deg = {
            name: math.degrees(positions[name])
            for name in self.command_joints
        }
        print(
            'synced real targets:',
            ', '.join(f'{name}={self.command_targets_deg[name]:+.2f}deg' for name in self.command_joints),
        )
        return True

    def adjust_sim_offset(self, delta_deg):
        joint = self.selected_joint()
        if not joint:
            return
        self.offsets_deg[joint] = self.offsets_deg.get(joint, 0.0) + float(delta_deg)
        self.print_status()

    def jog_real_joint(self, delta_deg):
        joint = self.selected_joint()
        if joint not in self.command_joints:
            print(f'real jog rejected: {joint} is not in command_joints={self.command_joints}')
            return

        max_step = float(self.get_parameter('max_real_step_deg').value)
        if abs(float(delta_deg)) > max_step:
            print(f'real jog rejected: {delta_deg:.3f} deg exceeds max_real_step_deg={max_step:.3f}')
            return

        positions = self.latest_positions_by_name()
        if positions is None:
            print('real jog rejected: no /joint_states yet')
            return
        missing = [name for name in self.command_joints if name not in positions]
        if missing:
            print(f'real jog rejected: /joint_states missing command joints: {missing}')
            return
        if self.command_targets_deg is None:
            if not self.sync_real_targets_to_current():
                return
        if not self.write_client.wait_for_service(timeout_sec=float(self.get_parameter('service_timeout_sec').value)):
            print('real jog rejected: /lerobot/write_joints service is not available')
            return

        before = float(self.command_targets_deg[joint])
        self.command_targets_deg[joint] = before + float(delta_deg)

        actual_deg = math.degrees(positions[joint])
        pending_error = self.command_targets_deg[joint] - actual_deg
        max_pending = float(self.get_parameter('max_pending_real_error_deg').value)
        if abs(pending_error) > max_pending:
            self.command_targets_deg[joint] = before
            print(
                f'real jog rejected: target-current error {pending_error:+.2f} deg '
                f'exceeds max_pending_real_error_deg={max_pending:.2f}. Press c to resync if needed.'
            )
            return

        targets_deg = [float(self.command_targets_deg[name]) for name in self.command_joints]
        idx = self.command_joints.index(joint)

        request = WriteJoints.Request()
        request.target_positions = [float(value) for value in targets_deg]
        future = self.write_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=float(self.get_parameter('service_timeout_sec').value))
        result = future.result()
        if result is None or not result.success:
            self.command_targets_deg[joint] = before
            print('real jog failed: /lerobot/write_joints returned failure')
            return
        self.last_real_command_time = time.monotonic()
        print(
            f'real jog sent: {joint} target {before:+.2f} -> {targets_deg[idx]:+.2f} deg '
            f'({delta_deg:+.2f}), actual={actual_deg:+.2f}, pending={pending_error:+.2f}'
        )

    def print_help(self):
        print()
        print('LeRobot real/sim angle tuner')
        print('  One terminal, two separate controls:')
        print('    SIM/RViz offset: a/d = -/+ step, z/x = -/+ fine')
        print('    REAL motor jog:  j/l = -/+ step, n/m = -/+ fine')
        print('    c: sync real command targets to current /joint_states')
        print('  1/2/3...: select joint')
        print('  p: save raw + offsets + adjusted display pose')
        print('  r: reload offsets from file')
        print('  0: reset selected sim offset')
        print('  R: reset all sim offsets')
        print('  h: help')
        print('  q: quit')
        print('  NOTE: only j/l/n/m move the real robot. a/d/z/x only change RViz display.')
        self.print_status()

    def print_status(self):
        sim_step = float(self.get_parameter('sim_step_deg').value)
        sim_fine = float(self.get_parameter('sim_fine_step_deg').value)
        real_step = float(self.get_parameter('real_step_deg').value)
        real_fine = float(self.get_parameter('real_fine_step_deg').value)
        selected = self.selected_joint()
        parts = []
        for idx, joint in enumerate(self.tune_joints):
            mark = '*' if joint == selected else ' '
            parts.append(f'{mark}{idx + 1}:{joint} sim_offset={self.offsets_deg.get(joint, 0.0):+.2f}deg')
        print(' | '.join(parts))
        print(f'  sim step={sim_step:g}/{sim_fine:g} deg, real step={real_step:g}/{real_fine:g} deg')

    def joint_state_to_yaml(self, msg, source_map=None):
        joints = []
        source_map = source_map or {}
        for idx, name in enumerate(msg.name):
            if idx >= len(msg.position):
                continue
            rad = float(msg.position[idx])
            item = {
                'name': name,
                'position_rad': rad,
                'position_deg': math.degrees(rad),
            }
            if name in source_map:
                item['source_joint'] = source_map[name]
            joints.append(item)
        return {
            'names': list(msg.name),
            'joints': joints,
        }

    def display_source_map(self):
        mapping = {}
        for joint in self.tune_joints:
            mapping[self.name_map.get(joint, joint)] = joint
        return mapping

    def save_offsets(self):
        path = Path(str(self.get_parameter('save_path').value)).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            raw = self.latest_msg
            adjusted = self.latest_adjusted_msg

        data = {
            'description': 'SO101 display offsets. adjusted_display = raw_input + offsets_deg.',
            'source_topic': str(self.get_parameter('joint_state_topic').value),
            'output_topic': str(self.get_parameter('display_joint_state_topic').value),
            'joints_used_for_ik': list(self.tune_joints),
            'command_joints': list(self.command_joints),
            'joint_name_map': dict(self.name_map),
            'offsets_deg': {joint: float(self.offsets_deg.get(joint, 0.0)) for joint in self.tune_joints},
        }
        if raw is not None:
            data['raw_input_pose'] = self.joint_state_to_yaml(raw)
        if adjusted is not None:
            data['adjusted_display_pose'] = self.joint_state_to_yaml(adjusted, self.display_source_map())
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')
        print(f'saved: {path}')

    def load_offsets(self, silent=False):
        path = Path(str(self.get_parameter('save_path').value)).expanduser()
        if not path.exists():
            if not silent:
                print(f'offset file does not exist: {path}')
            return
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        loaded = data.get('offsets_deg', {})
        for joint in self.tune_joints:
            if joint in loaded:
                self.offsets_deg[joint] = float(loaded[joint])
        if not silent:
            print(f'loaded: {path}')
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
        if key == 'c':
            self.sync_real_targets_to_current()
            return True
        if key == 'r':
            self.load_offsets()
            return True
        if key == 'R':
            for joint in self.tune_joints:
                self.offsets_deg[joint] = 0.0
            self.print_status()
            return True
        if key == '0':
            self.offsets_deg[self.selected_joint()] = 0.0
            self.print_status()
            return True
        if key.isdigit():
            idx = int(key) - 1
            if 0 <= idx < len(self.tune_joints):
                self.selected_index = idx
                self.print_status()
            return True

        sim_step = float(self.get_parameter('sim_step_deg').value)
        sim_fine = float(self.get_parameter('sim_fine_step_deg').value)
        real_step = float(self.get_parameter('real_step_deg').value)
        real_fine = float(self.get_parameter('real_fine_step_deg').value)

        if key == 'a':
            self.adjust_sim_offset(-sim_step)
        elif key == 'd':
            self.adjust_sim_offset(sim_step)
        elif key == 'z':
            self.adjust_sim_offset(-sim_fine)
        elif key == 'x':
            self.adjust_sim_offset(sim_fine)
        elif key == 'j':
            self.jog_real_joint(-real_step)
        elif key == 'l':
            self.jog_real_joint(real_step)
        elif key == 'n':
            self.jog_real_joint(-real_fine)
        elif key == 'm':
            self.jog_real_joint(real_fine)
        return True

    def spin_keyboard(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.03)
                readable, _, _ = select.select([sys.stdin], [], [], 0.03)
                if not readable:
                    continue
                key = sys.stdin.read(1)
                if not self.handle_key(key):
                    break
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def main(args=None):
    rclpy.init(args=args)
    node = LeRobotRealSimAngleTuner()
    try:
        node.spin_keyboard()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
