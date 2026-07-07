#!/home/whr/miniconda3/envs/lerobot/bin/python
"""Probe how smooth the SO101 bus servos can move with streamed targets.

This sends a single joint through a small motion using either linear
interpolation or a smoothstep S-curve.  The default mode is manual: every motion
segment waits for Enter before moving, so you can stop if the arm is near an
interference position.  Use --auto only when the workspace is known safe.
"""

import argparse
import math
import sys
import select
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from weed_locator.srv import WriteJoints


JOINT_ORDER = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper',
]


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def profile_value(t, profile):
    t = max(0.0, min(1.0, float(t)))
    if profile == 'linear':
        return t
    if profile == 'smoothstep':
        return smoothstep(t)
    if profile == 'smootherstep':
        return smootherstep(t)
    raise ValueError(f'unknown profile: {profile}')


class ServoSmoothnessProbe(Node):
    def __init__(self, args):
        super().__init__('lerobot_servo_smoothness_probe')
        self.args = args
        self.latest_joint_state = None
        self.joint_state_count = 0
        self.create_subscription(JointState, args.joint_state_topic, self.on_joint_state, 10)
        self.client = self.create_client(WriteJoints, args.write_joints_service)

    def on_joint_state(self, msg):
        self.latest_joint_state = msg
        self.joint_state_count += 1

    def wait_ready(self):
        deadline = time.monotonic() + self.args.timeout
        while rclpy.ok() and self.latest_joint_state is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.latest_joint_state is None:
            raise RuntimeError(f'timed out waiting for {self.args.joint_state_topic}')
        if not self.client.wait_for_service(timeout_sec=self.args.timeout):
            raise RuntimeError(f'{self.args.write_joints_service} service is not available')

    def current_degrees(self, joint):
        msg = self.latest_joint_state
        if msg is None:
            raise RuntimeError('no joint state yet')
        by_name = dict(zip(msg.name, msg.position, strict=False))
        if joint not in by_name:
            raise RuntimeError(f'{self.args.joint_state_topic} missing joint {joint!r}; got {list(msg.name)}')
        if joint == 'gripper':
            return math.degrees(float(by_name[joint]))
        return math.degrees(float(by_name[joint]))

    def send_joint(self, joint, target_deg):
        targets = [float('nan')] * len(JOINT_ORDER)
        targets[JOINT_ORDER.index(joint)] = float(target_deg)

        request = WriteJoints.Request()
        request.target_positions = targets
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.args.timeout)
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError(f'write_joints failed for {joint}={target_deg:.3f} deg')

    def stream_segment(self, joint, start_deg, end_deg, duration, rate_hz, profile):
        period = 1.0 / float(rate_hz)
        steps = max(1, int(round(duration * rate_hz)))
        start_time = time.monotonic()
        max_send_late_ms = 0.0

        for i in range(steps + 1):
            t = i / steps
            s = profile_value(t, profile)
            target = start_deg + (end_deg - start_deg) * s
            due = start_time + i * period
            now = time.monotonic()
            if now < due:
                time.sleep(due - now)
            else:
                max_send_late_ms = max(max_send_late_ms, (now - due) * 1000.0)

            self.send_joint(joint, target)
            rclpy.spin_once(self, timeout_sec=0.0)

        return max_send_late_ms

    def run_once(self, profile, rate_hz, duration):
        joint = self.args.joint
        center = self.current_degrees(joint)
        amplitude = float(self.args.amplitude_deg)
        low = center - amplitude
        high = center + amplitude

        print('')
        print(f'=== {joint}: profile={profile}, rate={rate_hz:g}Hz, duration={duration:g}s, amplitude=±{amplitude:g}deg ===')
        print(f'center={center:.2f}deg, range=[{low:.2f}, {high:.2f}]')
        print('watch/feel: 是否一格一格、是否抖、是否明显滞后、是否过冲')

        late = 0.0
        for cycle in range(1, self.args.cycles + 1):
            if not self.confirm_segment(cycle, 'center -> high -> center', center, high):
                return
            print(f'cycle {cycle}/{self.args.cycles}: center -> high')
            late = max(late, self.stream_segment(joint, center, high, duration, rate_hz, profile))
            print(f'cycle {cycle}/{self.args.cycles}: high -> center')
            late = max(late, self.stream_segment(joint, high, center, duration, rate_hz, profile))
            time.sleep(self.args.pause)

            if not self.args.full_cycle:
                continue

            if not self.confirm_segment(cycle, 'center -> low -> center', center, low):
                return
            print(f'cycle {cycle}/{self.args.cycles}: center -> low')
            late = max(late, self.stream_segment(joint, center, low, duration, rate_hz, profile))
            print(f'cycle {cycle}/{self.args.cycles}: low -> center')
            late = max(late, self.stream_segment(joint, low, center, duration, rate_hz, profile))
            time.sleep(self.args.pause)

        print(f'done: max send scheduling late={late:.1f}ms')

    def confirm_segment(self, cycle, label, start_deg, end_deg):
        if self.args.auto:
            return True
        print(
            f'cycle {cycle}/{self.args.cycles}: next {label} '
            f'({start_deg:.2f} -> {end_deg:.2f} deg). '
            'Enter=move, s=skip, q=quit: ',
            end='',
            flush=True,
        )
        while True:
            key = sys.stdin.readline().strip().lower()
            if key in ('', 'y', 'yes'):
                return True
            if key in ('s', 'skip'):
                print('skip')
                return False
            if key in ('q', 'quit', 'exit'):
                print('quit')
                return False
            print('请输入 Enter / s / q: ', end='', flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint', choices=JOINT_ORDER[:-1], default='shoulder_pan')
    parser.add_argument('--amplitude-deg', type=float, default=3.0)
    parser.add_argument('--duration', type=float, default=1.2, help='seconds for center->edge')
    parser.add_argument('--rate-hz', type=float, default=30.0)
    parser.add_argument(
        '--profile',
        choices=['linear', 'smoothstep', 'smootherstep', 'all'],
        default='smootherstep',
    )
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--pause', type=float, default=0.25)
    parser.add_argument('--timeout', type=float, default=3.0)
    parser.add_argument('--joint-state-topic', default='/joint_states')
    parser.add_argument('--write-joints-service', default='/lerobot/write_joints')
    parser.add_argument(
        '--sweep',
        action='store_true',
        help='manually try several rates and profiles: 10/20/30/50Hz linear+smootherstep',
    )
    parser.add_argument('--auto', action='store_true', help='run without per-segment Enter confirmation')
    parser.add_argument('--full-cycle', action='store_true', help='also test center->low->center; default only center->high->center')
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = ServoSmoothnessProbe(args)
    try:
        node.wait_ready()
        if args.sweep:
            for rate in (10.0, 20.0, 30.0, 50.0):
                for profile in ('linear', 'smootherstep'):
                    node.run_once(profile, rate, args.duration)
        else:
            profiles = ['linear', 'smoothstep', 'smootherstep'] if args.profile == 'all' else [args.profile]
            for profile in profiles:
                node.run_once(profile, args.rate_hz, args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
