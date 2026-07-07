#!/home/whr/miniconda3/envs/lerobot/bin/python
"""Keyboard Cartesian jog for the modified SO101 no-elbow arm.

The real hardware still reports the third motor as ``elbow_flex``.  In the
no-elbow URDF that motor drives the URDF ``wrist_flex`` joint, with display
offsets measured by visual alignment.  This node keeps those two spaces
separate:

  raw real joints -> display/URDF joints -> IK -> raw real command joints
"""

import math
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import JointState

from weed_locator.srv import WriteJoints


LEROBOT_SRC = Path('/home/whr/lerobot/src')
if LEROBOT_SRC.exists():
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.model.kinematics import RobotKinematics


DEFAULT_IK_JOINTS = ['shoulder_pan', 'shoulder_lift', 'wrist_flex']
DEFAULT_COMMAND_JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex']

KEY_DELTAS = {
    'w': np.array([1.0, 0.0, 0.0]),
    's': np.array([-1.0, 0.0, 0.0]),
    'a': np.array([0.0, 1.0, 0.0]),
    'd': np.array([0.0, -1.0, 0.0]),
    'r': np.array([0.0, 0.0, 1.0]),
    'f': np.array([0.0, 0.0, -1.0]),
}


def parse_list(raw):
    return [item.strip() for item in str(raw).split(',') if item.strip()]


def parse_name_map(raw):
    mapping = {}
    for item in str(raw).split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            continue
        source, target = [part.strip() for part in item.split(':', 1)]
        if source and target:
            mapping[source] = target
    return mapping


def wrap_degrees(angle):
    return ((float(angle) + 180.0) % 360.0) - 180.0


def nearest_equivalent_degrees(angle, reference):
    angle = float(angle)
    reference = float(reference)
    return angle + round((reference - angle) / 360.0) * 360.0


class TerminalMode:
    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError('keyboard jog needs an interactive terminal')
        self.settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


class NoElbowIKKeyboardJog(Node):
    def __init__(self):
        super().__init__('lerobot_no_elbow_ik_keyboard_jog')

        self.declare_parameter('step_m', 0.015)
        self.declare_parameter('relative_frame', 'base')
        self.declare_parameter('execute', True)
        self.declare_parameter('target_frame', 'gripper_frame_link')
        self.declare_parameter('position_weight', 1.0)
        self.declare_parameter('orientation_weight', 0.0)
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('display_joint_state_topic', '/display_joint_states')
        self.declare_parameter('write_joints_service', '/lerobot/write_joints')
        self.declare_parameter('offset_path', '/home/whr/cc_ws/tros_ws/calibration_targets/so101_joint_display_offsets.yaml')
        self.declare_parameter('ik_joint_names', ','.join(DEFAULT_IK_JOINTS))
        self.declare_parameter('command_joint_names', ','.join(DEFAULT_COMMAND_JOINTS))
        self.declare_parameter('joint_name_map', 'elbow_flex:wrist_flex')
        self.declare_parameter('max_joint_delta_deg', 18.0)
        self.declare_parameter('max_position_error_m', 0.02)
        self.declare_parameter('verify_after_execute', True)
        self.declare_parameter('settle_sec', 0.8)
        self.declare_parameter('timeout_sec', 3.0)
        self.declare_parameter('min_command_interval_sec', 0.25)
        self.declare_parameter('display_publish_rate', 30.0)
        self.declare_parameter('use_command_cache', True)
        self.declare_parameter('max_cache_error_deg', 6.0)
        self.declare_parameter('continuous_mode', True)
        self.declare_parameter('command_period_sec', 0.18)
        self.declare_parameter('key_timeout_sec', 0.32)
        self.declare_parameter('continuous_log_every', 1)
        self.declare_parameter('aux_joint_name', 'wrist_flex')
        self.declare_parameter('aux_step_deg', 3.0)
        self.declare_parameter('smooth_execute', True)
        self.declare_parameter('smooth_rate_hz', 10.0)
        self.declare_parameter('smooth_profile', 'smootherstep')
        self.declare_parameter('min_smooth_duration_sec', 0.25)
        self.declare_parameter('joint_speed_limits_deg_s', 'shoulder_pan:2.0,shoulder_lift:0.5,wrist_flex:0.5')

        self.step_m = float(self.get_parameter('step_m').value)
        self.ik_joints = parse_list(self.get_parameter('ik_joint_names').value)
        self.command_joints = parse_list(self.get_parameter('command_joint_names').value)
        self.name_map = parse_name_map(self.get_parameter('joint_name_map').value)
        self.display_to_source = {target: source for source, target in self.name_map.items()}
        self.offsets_deg = self.load_offsets()
        self.latest_msg = None
        self.joint_state_count = 0
        self.last_command_time = 0.0
        self.command_q_deg = None

        self.create_subscription(
            JointState,
            str(self.get_parameter('joint_state_topic').value),
            self.on_joint_state,
            10,
        )
        self.display_pub = self.create_publisher(
            JointState,
            str(self.get_parameter('display_joint_state_topic').value),
            10,
        )
        rate = float(self.get_parameter('display_publish_rate').value)
        self.create_timer(1.0 / max(rate, 1.0), self.publish_display_state)

        self.write_client = self.create_client(
            WriteJoints,
            str(self.get_parameter('write_joints_service').value),
        )
        self.kinematics = self.make_kinematics()

    def load_offsets(self):
        path = Path(str(self.get_parameter('offset_path').value)).expanduser()
        if not path.exists():
            raise RuntimeError(f'offset file not found: {path}')
        with path.open('r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        offsets = {str(k): float(v) for k, v in (data.get('offsets_deg') or {}).items()}
        file_map = data.get('joint_name_map') or {}
        if file_map:
            self.name_map.update({str(k): str(v) for k, v in file_map.items()})
            self.display_to_source = {target: source for source, target in self.name_map.items()}
        if not offsets:
            raise RuntimeError(f'offset file has no offsets_deg: {path}')
        return offsets

    def make_kinematics(self):
        pkg_dir = get_package_share_directory('weed_locator')
        urdf_path = str(Path(pkg_dir) / 'config' / 'SO101' / 'so101_no_elbow.urdf')
        self.get_logger().info(f'IK URDF: {urdf_path}')
        return RobotKinematics(
            urdf_path,
            target_frame_name=str(self.get_parameter('target_frame').value),
            joint_names=self.ik_joints,
        )

    def on_joint_state(self, msg):
        self.latest_msg = msg
        self.joint_state_count += 1

    def wait_for_joint_state(self):
        deadline = time.monotonic() + float(self.get_parameter('timeout_sec').value)
        while rclpy.ok() and self.latest_msg is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        return self.latest_msg

    def spin_for(self, duration_sec):
        deadline = time.monotonic() + max(0.0, float(duration_sec))
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.03, max(0.0, deadline - time.monotonic())))

    def raw_positions_deg(self, msg):
        return {
            name: math.degrees(float(msg.position[idx]))
            for idx, name in enumerate(msg.name)
            if idx < len(msg.position)
        }

    def source_for_display_joint(self, display_joint):
        return self.display_to_source.get(display_joint, display_joint)

    def raw_to_display_deg(self, raw_deg, display_joint):
        source_joint = self.source_for_display_joint(display_joint)
        if source_joint not in raw_deg:
            raise RuntimeError(f'/joint_states missing source joint {source_joint!r} for {display_joint!r}')
        return wrap_degrees(float(raw_deg[source_joint]) + float(self.offsets_deg.get(source_joint, 0.0)))

    def display_to_raw_deg(self, display_deg, display_joint, raw_deg=None):
        source_joint = self.source_for_display_joint(display_joint)
        raw_target = float(display_deg) - float(self.offsets_deg.get(source_joint, 0.0))
        if raw_deg is not None and source_joint in raw_deg:
            raw_target = nearest_equivalent_degrees(raw_target, raw_deg[source_joint])
        return raw_target

    def current_ik_degrees(self, msg):
        raw_deg = self.raw_positions_deg(msg)
        return np.array([self.raw_to_display_deg(raw_deg, joint) for joint in self.ik_joints], dtype=float)

    def publish_display_state(self):
        msg = self.latest_msg
        if msg is None:
            return
        try:
            raw_deg = self.raw_positions_deg(msg)
        except Exception:
            return

        mapped_targets = set(self.name_map.values())
        display = JointState()
        display.header = msg.header
        display.header.stamp = self.get_clock().now().to_msg()
        used = set()
        for name in msg.name:
            if name in mapped_targets and name not in self.name_map:
                continue
            display_name = self.name_map.get(name, name)
            if display_name in used:
                continue
            used.add(display_name)
            if name not in raw_deg:
                continue
            display.name.append(display_name)
            display.position.append(math.radians(wrap_degrees(raw_deg[name] + self.offsets_deg.get(name, 0.0))))

        self.display_pub.publish(display)

    def apply_delta(self, current_pose, delta):
        frame = str(self.get_parameter('relative_frame').value).lower().strip()
        target_pose = current_pose.copy()
        if frame == 'tool':
            target_pose[:3, 3] += current_pose[:3, :3] @ delta
        else:
            target_pose[:3, 3] += delta
        return target_pose

    def solve_target(self, delta):
        msg = self.wait_for_joint_state()
        if msg is None:
            raise RuntimeError('timed out waiting for /joint_states')

        q_actual = self.current_ik_degrees(msg)
        q_current = q_actual
        if bool(self.get_parameter('use_command_cache').value):
            if self.command_q_deg is None:
                self.command_q_deg = q_actual.copy()
            cache_error = float(np.max(np.abs(self.command_q_deg - q_actual)))
            max_cache_error = float(self.get_parameter('max_cache_error_deg').value)
            if cache_error <= max_cache_error:
                q_current = self.command_q_deg.copy()
            else:
                self.get_logger().warning(
                    f'command cache reset: target/actual joint error {cache_error:.2f} deg '
                    f'exceeds {max_cache_error:.2f} deg'
                )
                self.command_q_deg = q_actual.copy()

        t_current = self.kinematics.forward_kinematics(q_current)
        t_target = self.apply_delta(t_current, delta)
        q_target = self.kinematics.inverse_kinematics(
            q_current,
            t_target,
            position_weight=float(self.get_parameter('position_weight').value),
            orientation_weight=float(self.get_parameter('orientation_weight').value),
        )
        q_target = np.array(q_target, dtype=float)
        if not np.all(np.isfinite(q_target)):
            raise RuntimeError(f'IK returned non-finite target: {q_target.tolist()}')

        t_result = self.kinematics.forward_kinematics(q_target)
        delta_deg = q_target - q_current
        pos_error = float(np.linalg.norm(t_result[:3, 3] - t_target[:3, 3]))
        max_delta = float(np.max(np.abs(delta_deg)))
        return q_current, q_target, t_current, t_target, pos_error, max_delta

    def command_targets_from_ik(self, q_target):
        msg = self.latest_msg
        if msg is None:
            raise RuntimeError('no /joint_states available for command conversion')
        raw_deg = self.raw_positions_deg(msg)
        q_by_display = {joint: float(q_target[idx]) for idx, joint in enumerate(self.ik_joints)}

        targets = []
        for command_joint in self.command_joints:
            display_joint = self.name_map.get(command_joint, command_joint)
            if display_joint in q_by_display:
                targets.append(self.display_to_raw_deg(q_by_display[display_joint], display_joint, raw_deg))
            else:
                if command_joint not in raw_deg:
                    raise RuntimeError(f'/joint_states missing command joint {command_joint!r}')
                targets.append(float(raw_deg[command_joint]))
        return targets

    def send_command(self, q_target):
        if not self.write_client.wait_for_service(timeout_sec=float(self.get_parameter('timeout_sec').value)):
            raise RuntimeError('/lerobot/write_joints service is not available')
        request = WriteJoints.Request()
        targets = [float(v) for v in self.command_targets_from_ik(q_target)]
        active_count = max(len(self.command_joints), 4)
        while len(targets) < active_count:
            targets.append(float('nan'))
        request.target_positions = targets
        future = self.write_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=float(self.get_parameter('timeout_sec').value))
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError('failed to execute /lerobot/write_joints command')

    def parse_speed_limits(self):
        raw = str(self.get_parameter('joint_speed_limits_deg_s').value)
        limits = {}
        for item in raw.split(','):
            item = item.strip()
            if not item or ':' not in item:
                continue
            joint, value = [part.strip() for part in item.split(':', 1)]
            try:
                speed = float(value)
            except ValueError:
                continue
            if speed > 0.0:
                limits[joint] = speed
        return limits

    def smooth_fraction(self, t):
        t = max(0.0, min(1.0, float(t)))
        profile = str(self.get_parameter('smooth_profile').value).lower().strip()
        if profile == 'linear':
            return t
        if profile == 'smoothstep':
            return t * t * (3.0 - 2.0 * t)
        # smootherstep: smoother acceleration at both ends.
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def smooth_duration_sec(self, q_start, q_target):
        speed_limits = self.parse_speed_limits()
        duration = float(self.get_parameter('min_smooth_duration_sec').value)
        for idx, joint in enumerate(self.ik_joints):
            delta = abs(float(q_target[idx]) - float(q_start[idx]))
            speed = speed_limits.get(joint)
            if speed is not None and speed > 0.0:
                duration = max(duration, delta / speed)
        return duration

    def send_smooth_command(self, q_start, q_target):
        if not bool(self.get_parameter('smooth_execute').value):
            self.send_command(q_target)
            return

        q_start = np.array(q_start, dtype=float)
        q_target = np.array(q_target, dtype=float)
        duration = self.smooth_duration_sec(q_start, q_target)
        rate = max(1.0, float(self.get_parameter('smooth_rate_hz').value))
        steps = max(1, int(round(duration * rate)))
        period = 1.0 / rate
        start_time = time.monotonic()

        self.get_logger().info(
            f'smooth trajectory: duration={duration:.2f}s, rate={rate:.1f}Hz, '
            f'steps={steps}, delta_deg={np.round(q_target - q_start, 2).tolist()}'
        )

        for step in range(1, steps + 1):
            t = step / steps
            s = self.smooth_fraction(t)
            q_cmd = q_start + (q_target - q_start) * s
            due = start_time + step * period
            now = time.monotonic()
            if now < due:
                self.spin_for(due - now)
            self.send_command(q_cmd)
            self.command_q_deg = q_cmd.copy()

    def send_aux_joint_delta(self, delta_deg):
        aux_joint = str(self.get_parameter('aux_joint_name').value)
        msg = self.wait_for_joint_state()
        if msg is None:
            self.get_logger().error('timed out waiting for /joint_states')
            return
        raw_deg = self.raw_positions_deg(msg)
        if aux_joint not in raw_deg:
            self.get_logger().error(f'/joint_states missing aux joint {aux_joint!r}')
            return
        if not self.write_client.wait_for_service(timeout_sec=float(self.get_parameter('timeout_sec').value)):
            self.get_logger().error('/lerobot/write_joints service is not available')
            return
        active_count = max(len(self.command_joints), 4)
        targets = [float('nan')] * active_count
        aux_index = active_count - 1
        targets[aux_index] = float(raw_deg[aux_joint]) + float(delta_deg)
        request = WriteJoints.Request()
        request.target_positions = targets
        future = self.write_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=float(self.get_parameter('timeout_sec').value))
        result = future.result()
        if result is None or not result.success:
            self.get_logger().error(f'failed to move aux joint {aux_joint}')
            return
        self.get_logger().info(
            f'aux {aux_joint}: {raw_deg[aux_joint]:+.2f} -> {targets[aux_index]:+.2f} deg'
        )

    def verify_executed_pose(self, q_start, q_target, start_pose, target_pose):
        start_count = self.joint_state_count
        self.spin_for(float(self.get_parameter('settle_sec').value))
        if self.joint_state_count == start_count:
            self.get_logger().warning('no fresh /joint_states received after command')
            return

        q_actual = self.current_ik_degrees(self.latest_msg)
        t_actual = self.kinematics.forward_kinematics(q_actual)
        actual_delta = t_actual[:3, 3] - start_pose[:3, 3]
        desired_delta = target_pose[:3, 3] - start_pose[:3, 3]
        target_error = float(np.linalg.norm(t_actual[:3, 3] - target_pose[:3, 3]))
        progress = 0.0
        desired_norm_sq = float(np.dot(desired_delta, desired_delta))
        if desired_norm_sq > 1e-12:
            progress = float(np.dot(actual_delta, desired_delta) / desired_norm_sq)

        self.get_logger().info(
            'actual delta xyz mm: '
            f'{np.round(actual_delta * 1000.0, 1).tolist()}, '
            f'target error: {target_error * 1000.0:.1f} mm, '
            f'progress: {progress * 100.0:.0f}%'
        )
        self.get_logger().info(
            f'target joint delta deg {self.ik_joints}: '
            f'{np.round(q_target - q_start, 2).tolist()}'
        )
        self.get_logger().info(
            f'actual joint delta deg {self.ik_joints}: '
            f'{np.round(q_actual - q_start, 2).tolist()}'
        )

    def jog(self, key, enforce_interval=True, verify=None):
        now = time.monotonic()
        if enforce_interval and now - self.last_command_time < float(self.get_parameter('min_command_interval_sec').value):
            return

        delta = KEY_DELTAS[key] * self.step_m
        try:
            q_current, q_target, t_current, t_target, pos_error, max_delta = self.solve_target(delta)
        except Exception as exc:
            self.get_logger().error(str(exc))
            return

        max_allowed_delta = float(self.get_parameter('max_joint_delta_deg').value)
        max_allowed_error = float(self.get_parameter('max_position_error_m').value)
        if max_delta > max_allowed_delta or pos_error > max_allowed_error:
            self.get_logger().error(
                f'rejected {key}: max_delta={max_delta:.2f} deg '
                f'(limit {max_allowed_delta:.2f}), ik_error={pos_error * 1000.0:.2f} mm '
                f'(limit {max_allowed_error * 1000.0:.2f})'
            )
            return

        if bool(self.get_parameter('execute').value):
            try:
                self.send_smooth_command(q_current, q_target)
            except Exception as exc:
                self.get_logger().error(str(exc))
                return
            status = 'sent'
        else:
            status = 'dry-run'
        self.command_q_deg = q_target.copy()

        self.last_command_time = time.monotonic()
        self.get_logger().info(
            f'{status} {key}: target delta mm {np.round(delta * 1000.0, 1).tolist()}, '
            f'ik_error={pos_error * 1000.0:.2f} mm, max_joint={max_delta:.2f} deg'
        )
        should_verify = bool(self.get_parameter('verify_after_execute').value) if verify is None else bool(verify)
        if should_verify and bool(self.get_parameter('execute').value):
            self.verify_executed_pose(q_current, q_target, t_current, t_target)

    def print_help(self):
        print('')
        print('LeRobot SO101 no-elbow IK keyboard jog')
        print(f'  step: {self.step_m * 1000.0:.1f} mm, execute: {bool(self.get_parameter("execute").value)}')
        print(f'  IK joints: {self.ik_joints}')
        print(f'  command joints: {self.command_joints}')
        print(f'  continuous: {bool(self.get_parameter("continuous_mode").value)}')
        print('  w/s: +x/-x    a/d: +y/-y    r/f: +z/-z')
        print('  o/p: aux fourth motor +/-')
        print('  c: sync command cache to current real joints')
        print('  +/-: step up/down             q: quit')
        print('')

    def adjust_step(self, scale):
        self.step_m = max(0.005, min(0.10, self.step_m * scale))
        print(f'step: {self.step_m * 1000.0:.1f} mm')

    def initialize(self):
        if self.wait_for_joint_state() is None:
            raise RuntimeError('timed out waiting for /joint_states')
        if bool(self.get_parameter('execute').value):
            if not self.write_client.wait_for_service(timeout_sec=float(self.get_parameter('timeout_sec').value)):
                raise RuntimeError('/lerobot/write_joints service is not available')

    def read_available_keys(self):
        keys = []
        while select.select([sys.stdin], [], [], 0.0)[0]:
            keys.append(sys.stdin.read(1).lower())
        if not keys:
            rclpy.spin_once(self, timeout_sec=0.01)
        return keys

    def sync_command_cache(self):
        msg = self.wait_for_joint_state()
        if msg is not None:
            self.command_q_deg = self.current_ik_degrees(msg)
            print(f'command cache synced: {np.round(self.command_q_deg, 2).tolist()}')

    def handle_non_motion_key(self, key):
        if key in ('+', '='):
            self.adjust_step(1.5)
            return True
        if key in ('-', '_'):
            self.adjust_step(1.0 / 1.5)
            return True
        if key == 'c':
            self.sync_command_cache()
            return True
        if key == 'o':
            self.send_aux_joint_delta(float(self.get_parameter('aux_step_deg').value))
            return True
        if key == 'p':
            self.send_aux_joint_delta(-float(self.get_parameter('aux_step_deg').value))
            return True
        return False

    def run_continuous(self):
        active_key = None
        last_key_time = 0.0
        next_command_time = 0.0
        command_count = 0
        command_period = max(0.08, float(self.get_parameter('command_period_sec').value))
        key_timeout = max(command_period, float(self.get_parameter('key_timeout_sec').value))
        print('  continuous mode: hold w/s/a/d/r/f to move, release to stop')
        print(f'  command period={command_period:.2f}s, key timeout={key_timeout:.2f}s\n')

        with TerminalMode():
            while rclpy.ok():
                now = time.monotonic()
                for key in self.read_available_keys():
                    if key in ('q', '\x03'):
                        print('quit')
                        return
                    if key in KEY_DELTAS:
                        active_key = key
                        last_key_time = now
                    elif key in ('o', 'p'):
                        self.handle_non_motion_key(key)
                    else:
                        self.handle_non_motion_key(key)

                if active_key is not None and now - last_key_time > key_timeout:
                    active_key = None

                if active_key is not None and now >= next_command_time:
                    self.jog(active_key, enforce_interval=False, verify=False)
                    command_count += 1
                    log_every = max(1, int(self.get_parameter('continuous_log_every').value))
                    if command_count % log_every == 0:
                        self.get_logger().info(f'continuous active key: {active_key}')
                    next_command_time = time.monotonic() + command_period

                rclpy.spin_once(self, timeout_sec=0.01)

    def run(self):
        self.initialize()
        self.print_help()
        if bool(self.get_parameter('continuous_mode').value):
            self.run_continuous()
            return
        with TerminalMode():
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.02)
                readable, _, _ = select.select([sys.stdin], [], [], 0.02)
                if not readable:
                    continue
                key = sys.stdin.read(1).lower()
                if key == 'q' or key == '\x03':
                    print('quit')
                    return
                if self.handle_non_motion_key(key):
                    continue
                if key in KEY_DELTAS:
                    self.jog(key)


def main(args=None):
    rclpy.init(args=args)
    node = NoElbowIKKeyboardJog()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        node.get_logger().error(str(exc))
        raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
