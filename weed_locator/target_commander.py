#!/usr/bin/env python3
"""
Simple target commander for the Delta ESP32 bridge.

Input:
- /delta_arm/pick_target (geometry_msgs/Point)

Output:
- /delta_arm/move_to
- /delta_arm/pump_cmd
- /delta_arm/valve_cmd
"""

import threading
import time

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, String


class TargetCommander(Node):
    """Translate one target point into a small pick sequence."""

    def __init__(self):
        super().__init__('target_commander')

        self.declare_parameter('pre_grasp_offset_z', 40.0)
        self.declare_parameter('lift_offset_z', 60.0)
        self.declare_parameter('safe_z', -140.0)
        self.declare_parameter('drop_x', 0.0)
        self.declare_parameter('drop_y', 80.0)
        self.declare_parameter('drop_z', -180.0)
        self.declare_parameter('command_delay_sec', 1.0)
        self.declare_parameter('pump_settle_sec', 0.5)
        self.declare_parameter('release_settle_sec', 0.5)
        self.declare_parameter('auto_home_first', False)
        self.declare_parameter('home_settle_sec', 3.0)
        self.declare_parameter('pc_relative_mode', True)
        self.declare_parameter('relative_probe_z', 19.0)
        self.declare_parameter('relative_travel_z', 10.0)
        self.declare_parameter('relative_lower_before_xy', True)
        self.declare_parameter('go_work_origin_before_pick', True)
        self.declare_parameter('work_origin_x', 0.0)
        self.declare_parameter('work_origin_y', -55.0)
        self.declare_parameter('work_origin_z', -195.0)

        self.pre_grasp_offset_z = float(self.get_parameter('pre_grasp_offset_z').value)
        self.lift_offset_z = float(self.get_parameter('lift_offset_z').value)
        self.safe_z = float(self.get_parameter('safe_z').value)
        self.drop_x = float(self.get_parameter('drop_x').value)
        self.drop_y = float(self.get_parameter('drop_y').value)
        self.drop_z = float(self.get_parameter('drop_z').value)
        self.command_delay_sec = float(self.get_parameter('command_delay_sec').value)
        self.pump_settle_sec = float(self.get_parameter('pump_settle_sec').value)
        self.release_settle_sec = float(self.get_parameter('release_settle_sec').value)
        self.auto_home_first = bool(self.get_parameter('auto_home_first').value)
        self.home_settle_sec = float(self.get_parameter('home_settle_sec').value)
        self.pc_relative_mode = bool(self.get_parameter('pc_relative_mode').value)
        self.relative_probe_z = float(self.get_parameter('relative_probe_z').value)
        self.relative_travel_z = float(self.get_parameter('relative_travel_z').value)
        self.relative_lower_before_xy = bool(self.get_parameter('relative_lower_before_xy').value)
        self.go_work_origin_before_pick = bool(self.get_parameter('go_work_origin_before_pick').value)
        self.work_origin_x = float(self.get_parameter('work_origin_x').value)
        self.work_origin_y = float(self.get_parameter('work_origin_y').value)
        self.work_origin_z = float(self.get_parameter('work_origin_z').value)

        self.move_pub = self.create_publisher(Point, '/delta_arm/move_to', 20)
        self.pump_pub = self.create_publisher(Bool, '/delta_arm/pump_cmd', 20)
        self.valve_pub = self.create_publisher(Bool, '/delta_arm/valve_cmd', 20)
        self.motor_pub = self.create_publisher(Bool, '/delta_arm/motor_enable', 20)
        self.home_pub = self.create_publisher(Empty, '/delta_arm/home', 20)
        self.raw_gcode_pub = self.create_publisher(String, '/delta_arm/gcode_raw', 20)

        self.create_subscription(Point, '/delta_arm/pick_target', self.on_pick_target, 20)

        self.sequence_lock = threading.Lock()
        self.busy = False

        self.get_logger().info('target_commander 已启动')
        self.get_logger().info('订阅: /delta_arm/pick_target')
        self.get_logger().info('发布: /delta_arm/move_to, /delta_arm/gcode_raw, /delta_arm/pump_cmd, /delta_arm/valve_cmd, /delta_arm/motor_enable, /delta_arm/home')

    def publish_move(self, x: float, y: float, z: float):
        msg = Point()
        msg.x = float(x)
        msg.y = float(y)
        msg.z = float(z)
        self.move_pub.publish(msg)
        self.get_logger().info(f'动作: move_to({msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f})')

    def publish_pump(self, enabled: bool):
        msg = Bool()
        msg.data = enabled
        self.pump_pub.publish(msg)
        self.get_logger().info(f'动作: pump {"ON" if enabled else "OFF"}')

    def publish_valve(self, enabled: bool):
        msg = Bool()
        msg.data = enabled
        self.valve_pub.publish(msg)
        self.get_logger().info(f'动作: valve {"OPEN" if enabled else "CLOSE"}')

    def publish_motor_enable(self, enabled: bool):
        msg = Bool()
        msg.data = enabled
        self.motor_pub.publish(msg)
        self.get_logger().info(f'动作: motor {"ENABLE" if enabled else "DISABLE"}')

    def publish_home(self):
        self.home_pub.publish(Empty())
        self.get_logger().info('动作: home')

    def publish_raw_gcode(self, block: str):
        msg = String()
        msg.data = block
        self.raw_gcode_pub.publish(msg)
        self.get_logger().info('动作: raw_gcode\n' + block)

    def on_pick_target(self, msg: Point):
        with self.sequence_lock:
            if self.busy:
                self.get_logger().warning('当前仍在执行上一条抓取流程，忽略新目标')
                return
            self.busy = True

        worker = threading.Thread(
            target=self.run_pick_sequence,
            args=(msg.x, msg.y, msg.z),
            daemon=True,
        )
        worker.start()

    def run_pick_sequence(self, x: float, y: float, z: float):
        try:
            if self.pc_relative_mode:
                self.run_pc_relative_sequence(x, y)
                return

            if self.auto_home_first:
                self.publish_home()
                time.sleep(self.home_settle_sec)

            pre_grasp_z = z + self.pre_grasp_offset_z
            lift_z = z + self.lift_offset_z
            drop_pre_z = self.drop_z + self.pre_grasp_offset_z

            if pre_grasp_z > self.safe_z:
                pre_grasp_z = self.safe_z
            if lift_z > self.safe_z:
                lift_z = self.safe_z
            if drop_pre_z > self.safe_z:
                drop_pre_z = self.safe_z

            self.publish_motor_enable(True)
            time.sleep(self.command_delay_sec)

            # Make sure release valve is closed before suction.
            self.publish_valve(False)
            time.sleep(self.command_delay_sec)

            self.publish_move(x, y, pre_grasp_z)
            time.sleep(self.command_delay_sec)

            self.publish_move(x, y, z)
            time.sleep(self.command_delay_sec)

            self.publish_pump(True)
            time.sleep(self.pump_settle_sec)

            self.publish_move(x, y, lift_z)
            time.sleep(self.command_delay_sec)

            self.publish_move(self.drop_x, self.drop_y, drop_pre_z)
            time.sleep(self.command_delay_sec)

            self.publish_move(self.drop_x, self.drop_y, self.drop_z)
            time.sleep(self.command_delay_sec)

            self.publish_valve(True)
            time.sleep(self.release_settle_sec)

            self.publish_pump(False)
            time.sleep(self.command_delay_sec)

            self.publish_valve(False)
            time.sleep(self.command_delay_sec)

            self.publish_move(self.drop_x, self.drop_y, self.safe_z)
            time.sleep(self.command_delay_sec)

            self.get_logger().info('抓取放置流程执行完毕')
        finally:
            with self.sequence_lock:
                self.busy = False

    def run_pc_relative_sequence(self, x: float, y: float):
        """Replicate the original PC upper-computer relative G-code sequence."""
        if self.auto_home_first:
            self.publish_home()
            time.sleep(self.home_settle_sec)

        self.publish_motor_enable(True)
        time.sleep(self.command_delay_sec)

        if self.go_work_origin_before_pick:
            self.publish_move(self.work_origin_x, self.work_origin_y, self.work_origin_z)
            time.sleep(self.command_delay_sec)

        self.publish_valve(False)
        time.sleep(self.command_delay_sec)

        probe = abs(self.relative_probe_z)
        travel = max(0.0, min(abs(self.relative_travel_z), probe))
        final_probe = probe - travel
        end_x = self.drop_x
        end_y = self.drop_y

        if self.relative_lower_before_xy:
            # Lower to a travel plane before XY moves. This avoids the small XY
            # workspace near the high home position on this Delta arm.
            lines = [
                'G91',
                f'G0 Z-{travel:.0f}',
                f'G0 X{int(x)} Y{int(y)}',
            ]
            if final_probe > 0.0:
                lines.append(f'G0 Z-{final_probe:.0f}')
            lines.extend([
                'M121',
            ])
            if final_probe > 0.0:
                lines.append(f'G0 Z{final_probe:.0f}')
            lines.append(f'G0 X{int(end_x - x)} Y{int(end_y - y)}')
            if final_probe > 0.0:
                lines.append(f'G0 Z-{final_probe:.0f}')
            lines.extend([
                'M1',
            ])
            if final_probe > 0.0:
                lines.append(f'G0 Z{final_probe:.0f}')
            lines.extend([
                f'G0 X{int(-end_x)} Y{int(-end_y)}',
                f'G0 Z{travel:.0f}',
                'M2',
            ])
            block = '\n'.join(lines)
        else:
            # Same structure as CameraDetect.py:
            # G91; move to target; down/up; move to drop; down; release; up; return origin.
            block = '\n'.join([
                'G91',
                f'G0 X{int(x)} Y{int(y)}',
                f'G0 Z-{probe:.0f}',
                'M121',
                f'G0 Z{probe:.0f}',
                f'G0 X{int(end_x - x)} Y{int(end_y - y)}',
                f'G0 Z-{probe:.0f}',
                'M1',
                f'G0 Z{probe:.0f}',
                f'G0 X{int(-end_x)} Y{int(-end_y)}',
                'M2',
            ])
        self.publish_raw_gcode(block)
        time.sleep(self.command_delay_sec)
        self.get_logger().info('PC相对模式抓取放置流程已下发')


def main(args=None):
    rclpy.init(args=args)
    node = TargetCommander()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
