#!/usr/bin/env python3
"""
基于placo的运动学求解服务
使用SO101 URDF和placo的KinematicsSolver进行逆运动学求解
支持发布JointState供RVIZ仿真显示
"""

import rclpy
from rclpy.node import Node
import numpy as np
import placo
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from weed_locator.srv import SolveIK


class IKSolver(Node):
    """基于placo的逆运动学求解器"""

    def __init__(self):
        super().__init__('ik_solver')
        
        # URDF文件路径 - 只使用share目录
        import os
        from ament_index_python.packages import get_package_share_directory
        package_share_dir = get_package_share_directory('weed_locator')
        urdf_path = os.path.join(package_share_dir, 'config', 'SO101', 'so101_new_calib.urdf')
        
        self.get_logger().info(f'Loading robot from {urdf_path}')
        
        # 加载机器人模型
        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        
        # 获取关节名称列表
        self.joint_names = list(self.robot.joint_names())
        self.get_logger().info(f'Joint names: {self.joint_names}')
        
        # 末端执行器名称 - 必须与URDF中的link名称一致
        self.ee_frame = 'gripper_link'
        
        # 禁用关节限位（如果误差太大可以尝试启用）
        self.solver.enable_joint_limits(False)
        
        # 注意：不使用 mask_fbase(True)，因为这会导致四元数w被固定
        # 失去浮动基座的灵活性，导致IK求解误差大
        
        # 初始化状态q (13个值 = 浮动基座7参数 + 6个关节)
        # state.q 结构: [fbase_xyz(3), fbase_quat_xyz(3), quat_w(1), joint(6)]
        # 浮动基座初始为单位姿态(位置0,0,0 四元数0,0,0,1)
        self.robot.state.q = np.zeros(13)
        self.robot.state.q[6] = 1.0  # quaternion w at index 6
        self.robot.update_kinematics()
        
        self.get_logger().info('IK Solver initialized with placo')
        
        # 注意：在仿真模式下，不发布/joint_states
        # 因为fake_dynamixel_node会发布，而RVIZ只需要一个来源
        # 如果需要调试，可以取消注释下面的代码
        # self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        
        # 创建服务
        self.srv = self.create_service(SolveIK, '/weed_locator/solve_ik', self.solve_ik_callback)
        self.get_logger().info('IK service ready: /weed_locator/solve_ik')
        
        # 注意：不再定时发布关节状态，由fake_dynamixel_node负责发布
        # self.timer = self.create_timer(0.1, self.publish_joint_state)
        # self.get_logger().info('Publishing joint states to /joint_states')
    
    def publish_joint_state(self):
        """发布关节状态用于RVIZ显示"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [self.robot.get_joint(name) for name in self.joint_names]
        self.joint_state_pub.publish(msg)

    def solve_ik_callback(self, request, response):
        """逆运动学求解回调"""
        try:
            # 获取目标位置
            target_position = np.array([
                request.target_pose.position.x,
                request.target_pose.position.y,
                request.target_pose.position.z
            ])
            
            self.get_logger().info(f'IK request: target={target_position}')
            
            # 重置状态到初始姿态，确保每次求解都从相同的起点开始
            # 否则上次求解的结果会作为下次的初始猜测，导致误差累积
            self.robot.state.q = np.zeros(13)
            self.robot.state.q[6] = 1.0  # quaternion w
            self.robot.update_kinematics()
            
            # 清除旧任务
            self.solver.clear()
            
            # 添加位置任务
            task = self.solver.add_position_task(self.ee_frame, target_position)
            
            # 求解IK
            q_solution = self.solver.solve(False)
            
            self.get_logger().info(f'q_solution length: {len(q_solution)}, model.nq: {self.robot.model.nq}')
            
            # 正确处理 q_solution 结构：
            # q_solution = [fbase_xyz(3), fbase_quat_xyz(3), joint(6)] = 12 values
            # state.q = [fbase_xyz(3), fbase_quat_xyz(3), quat_w(1), joint(6)] = 13 values
            if len(q_solution) == 12 and self.robot.model.nq == 13:
                q_full = np.zeros(13)
                q_full[0:3] = q_solution[0:3]  # fbase xyz
                q_full[3:6] = q_solution[3:6]  # fbase quat xyz
                q_full[6] = 1.0  # quaternion w
                q_full[7:13] = q_solution[6:12]  # joint angles
                self.robot.state.q = q_full
            else:
                self.robot.state.q = q_solution.copy()
            
            # 再次更新运动学
            self.robot.update_kinematics()
            
            # 通过关节名称获取6个关节角度（避开浮动基座）
            joint_positions = [self.robot.get_joint(name) for name in self.joint_names]
            
            # 验证结果 - 计算末端位置（相对于universe）
            T_result = self.robot.get_T_a_b("universe", self.ee_frame)
            actual_position = T_result[:3, 3]
            position_error = np.linalg.norm(actual_position - target_position)
            
            self.get_logger().info(f'IK solved: actual={actual_position}, target={target_position}, error={position_error:.4f}m')
            self.get_logger().info(f'Joint angles: {[f"{q:.3f}" for q in joint_positions]}')
            
            response.success = True
            response.message = f'IK solved with position error: {position_error:.4f}m'
            response.joint_positions = joint_positions
            
        except Exception as e:
            self.get_logger().error(f'IK solving failed: {str(e)}')
            response.success = False
            response.message = str(e)
            response.joint_positions = []
        
        return response


def main(args=None):
    rclpy.init(args=args)
    node = IKSolver()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
