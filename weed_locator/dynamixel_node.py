#!/usr/bin/env python3
"""
Dynamixel 基础通信节点
- 初始化串口连接
- 读取6个关节的当前位置
- 发布测试指令让单关节动一下
- 提供服务: 读写关节位置
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from weed_locator.srv import ReadJoints, WriteJoints, MoveJoint
import serial
import time
import numpy as np


class DynamixelController(Node):
    """Dynamixel 串口通信控制器"""

    def __init__(self):
        super().__init__('dynamixel_controller')

        # 参数声明
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 1000000)
        self.declare_parameter('joint_ids', [1, 2, 3, 4, 5, 6])
        self.declare_parameter('publish_rate', 10.0)

        self.port = self.get_parameter('port').value
        self.baudrate = self.get_parameter('baudrate').value
        self.joint_ids = self.get_parameter('joint_ids').value
        self.publish_rate = self.get_parameter('publish_rate').value

        # 关节位置限制 [min, max]
        self.joint_limits = [
            [800, 3200],   # ID1: 基座
            [1000, 3000],  # ID2: 肩
            [1000, 3000],  # ID3: 肘
            [500, 3500],   # ID4: 腕俯仰
            [500, 3500],   # ID5: 腕旋转
            [500, 3500],   # ID6: 夹爪
        ]

        # 初始化串口
        self.ser = None
        self.init_serial()

        # 关节位置发布者
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)

        # 定时器: 读取和发布关节位置
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)

        # 当前关节位置
        self.current_positions = [2048.0] * 6

        # 创建服务
        self.read_joints_srv = self.create_service(
            ReadJoints, '/dynamixel/read_joints', self.read_joints_callback)
        self.write_joints_srv = self.create_service(
            WriteJoints, '/dynamixel/write_joints', self.write_joints_callback)
        self.move_joint_srv = self.create_service(
            MoveJoint, '/dynamixel/move_joint', self.move_joint_callback)

        self.get_logger().info('服务已创建: /dynamixel/read_joints, /dynamixel/write_joints, /dynamixel/move_joint')

        self.get_logger().info('Dynamixel 控制器已初始化')
        self.get_logger().info(f'串口: {self.port}, 波特率: {self.baudrate}')

        # 运行测试: 动一下关节1
        self.test_motion()

    def init_serial(self):
        """初始化串口连接"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            time.sleep(0.1)  # 等待稳定
            self.get_logger().info(f'串口 {self.port} 已打开')
        except serial.SerialException as e:
            self.get_logger().error(f'无法打开串口 {self.port}: {e}')
            raise

    def checksum(self, data):
        """计算校验和"""
        return (~sum(data)) & 0xFF

    def write_position(self, servo_id, position):
        """发送位置指令到指定舵机"""
        pos = int(np.clip(position, 0, 4095))
        data = [servo_id, 0x05, 0x03, 0x2A, pos & 0xFF, (pos >> 8) & 0xFF]
        cs = self.checksum(data)
        packet = bytes([0xFF, 0xFF] + data + [cs])
        self.ser.write(packet)
        time.sleep(0.001)

    def read_position(self, servo_id):
        """读取指定舵机的当前位置 (寄存器0x38=当前位置)"""
        # 读指令: FF FF ID SIZE CMD ADDR LEN CHECKSUM
        data = [servo_id, 0x04, 0x02, 0x38, 0x02]  # 读2字节
        cs = self.checksum(data)
        packet = bytes([0xFF, 0xFF] + data + [cs])
        self.ser.write(packet)
        time.sleep(0.002)

        # 读取响应
        if self.ser.in_waiting >= 7:
            response = self.ser.read(self.ser.in_waiting)
            if len(response) >= 7 and response[0] == 0xFF and response[1] == 0xFF:
                pos = response[5] | (response[6] << 8)
                return pos
        return None

    def read_all_positions(self):
        """读取所有6个关节的位置"""
        positions = []
        for sid in self.joint_ids:
            pos = self.read_position(sid)
            if pos is not None:
                positions.append(float(pos))
            else:
                # 读取失败时使用上次位置
                idx = self.joint_ids.index(sid)
                positions.append(self.current_positions[idx])
        return positions

    def test_motion(self):
        """测试运动: 让关节1来回动一下"""
        self.get_logger().info('执行测试运动: 关节1')

        # 保存原始位置
        original_pos = self.current_positions[0]

        # 移动到位置 1500
        self.get_logger().info('移动关节1到 1500...')
        for _ in range(20):
            self.write_position(1, 1500)
            time.sleep(0.01)

        time.sleep(0.5)

        # 移动到位置 2500
        self.get_logger().info('移动关节1到 2500...')
        for _ in range(20):
            self.write_position(1, 2500)
            time.sleep(0.01)

        time.sleep(0.5)

        # 回到原始位置
        self.get_logger().info(f'关节1回到原始位置 {int(original_pos)}...')
        for _ in range(20):
            self.write_position(1, original_pos)
            time.sleep(0.01)

        self.get_logger().info('测试运动完成')

    def read_joints_callback(self, request, response):
        """读取所有关节位置服务"""
        positions = self.read_all_positions()
        response.positions = positions
        response.success = True
        return response

    def write_joints_callback(self, request, response):
        """写入所有关节目标位置服务"""
        try:
            for i, sid in enumerate(self.joint_ids):
                pos = int(np.clip(request.target_positions[i],
                                  self.joint_limits[i][0],
                                  self.joint_limits[i][1]))
                for _ in range(10):
                    self.write_position(sid, pos)
                    time.sleep(0.005)
            response.success = True
        except Exception as e:
            self.get_logger().error(f'写入关节失败: {e}')
            response.success = False
        return response

    def move_joint_callback(self, request, response):
        """单关节移动服务"""
        joint_id = request.joint_id
        target = request.target_position

        if joint_id not in self.joint_ids:
            self.get_logger().error(f'无效的关节ID: {joint_id}')
            response.success = False
            return response

        idx = self.joint_ids.index(joint_id)
        pos = int(np.clip(target, self.joint_limits[idx][0], self.joint_limits[idx][1]))

        for _ in range(20):
            self.write_position(joint_id, pos)
            time.sleep(0.005)

        response.success = True
        return response

    def timer_callback(self):
        """定时读取并发布关节位置"""
        # 读取所有关节位置
        positions = self.read_all_positions()
        self.current_positions = positions

        # 发布 JointState 消息
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f'joint_{i}' for i in self.joint_ids]
        msg.position = positions

        self.joint_pub.publish(msg)

    def close(self):
        """关闭串口连接"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.get_logger().info('串口已关闭')


def main(args=None):
    rclpy.init(args=args)

    try:
        controller = DynamixelController()
        rclpy.spin(controller)
    except Exception as e:
        print(f'错误: {e}')
    finally:
        if 'controller' in locals():
            controller.close()
        rclpy.shutdown()


if __name__ == '__main__':
    main()