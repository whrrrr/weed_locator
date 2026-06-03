#!/usr/bin/env python3
"""
SO101 运动学模块 - 基于 Argo-Robot/controls
参考: https://github.com/Argo-Robot/controls
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def inv_homog_mat(T):
    """高效求逆 4x4 齐次变换矩阵"""
    R_inv = T[:3, :3].T
    t_inv = -R_inv @ T[:3, 3]
    return np.block([[R_inv, t_inv.reshape(3, 1)], [0, 0, 0, 1]])


def calc_lin_err(T1, T2):
    """计算位置误差"""
    return T1[:3, 3] - T2[:3, 3]


def calc_ang_err(T1, T2):
    """计算角度误差 (使用 Rodrigues' 公式)"""
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    R_rel = R2.T @ R1
    trace = np.trace(R_rel)
    trace = np.clip(trace, -1, 3)
    angle = np.arccos((trace - 1) / 2)
    if angle < 1e-6:
        return np.zeros(3)
    log_rot = angle / (2 * np.sin(angle)) * np.array([
        R_rel[2, 1] - R_rel[1, 2],
        R_rel[0, 2] - R_rel[2, 0],
        R_rel[1, 0] - R_rel[0, 1]
    ])
    return log_rot


def calc_dh_matrix(dh, theta):
    """计算标准 DH 齐次变换矩阵"""
    theta_i, d_i, a_i, alpha_i = dh
    ct = np.cos(theta_i + theta)
    st = np.sin(theta_i + theta)
    ca = np.cos(alpha_i)
    sa = np.sin(alpha_i)
    return np.array([
        [ct, -st * ca, st * sa, a_i * ct],
        [st, ct * ca, -ct * sa, a_i * st],
        [0, sa, ca, d_i],
        [0, 0, 0, 1]
    ])


def dls_right_pseudoinv(J, lambda_val=0.001):
    """阻尼最小二乘右伪逆"""
    return J.T @ np.linalg.inv(J @ J.T + lambda_val**2 * np.eye(J.shape[0]))


class SO101Kinematics:
    """SO101 运动学求解器 - 基于 DH 方法"""
    
    def __init__(self):
        # DH 参数表 [theta, d, a, alpha]
        # 来源: https://github.com/Argo-Robot/controls
        self.dh_table = [
            [0, 0.0542, 0.0304, np.pi / 2],      # Joint 1: 基座
            [0, 0.0, 0.188, 0.0],                 # Joint 2: 肩部，改装后大臂
            [0, 0.0, 0.230, 0.0],                 # Joint 3: 肘部，改装后小臂
            [0, 0.0, 0.0, -np.pi / 2],            # Joint 4: 腕俯仰
            [0, 0.0609, 0.0, 0.0],                # Joint 5: 腕旋转
        ]
        
        # 世界坐标系到基座的变换 (需要根据实际机器人调整)
        self.worldTbase = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, -0.0453],
            [0.0, 0.0, 1.0, 0.0647],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # 末端执行器坐标系
        self.nTtool = np.array([
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # 机械关节限制
        self.mech_limits_low = np.array([-2.2, -3.1416, 0.0, -2.0, -3.1416, -0.2])
        self.mech_limits_up = np.array([2.2, 0.2, 3.1416, 1.8, 3.1416, 2.0])
        
        # DH 到机械角度的偏移 (14.45°)
        self.beta = np.deg2rad(14.45)
    
    def mechanical_to_dh(self, q_mech):
        """
        将机械角度转换为 DH 角度
        
        q_mech: 机械关节角度 [theta1, theta2, theta3, theta4, theta5, theta6]
        返回: DH 角度 [theta1, theta2, theta3, theta4, theta5]
        """
        q_dh = np.zeros(5)
        q_dh[0] = q_mech[0]
        q_dh[1] = -q_mech[1] - self.beta
        q_dh[2] = -q_mech[2] + self.beta
        q_dh[3] = -q_mech[3] - np.pi / 2
        q_dh[4] = -q_mech[4] - np.pi / 2
        return q_dh
    
    def dh_to_mechanical(self, q_dh):
        """
        将 DH 角度转换为机械角度
        
        q_dh: DH 角度 [theta1, theta2, theta3, theta4, theta5]
        返回: 机械关节角度 [theta1, theta2, theta3, theta4, theta5, theta6]
        """
        q_mech = np.zeros(6)
        q_mech[0] = q_dh[0]
        q_mech[1] = -q_dh[1] - self.beta
        q_mech[2] = -q_dh[2] + self.beta
        q_mech[3] = -q_dh[3] - np.pi / 2
        q_mech[4] = -q_dh[4] - np.pi / 2
        q_mech[5] = 0.0  # 夹爪初始角度
        return q_mech
    
    def forward_kinematics(self, q_dh):
        """
        正向运动学: 根据关节角度计算末端位置
        
        q_dh: DH 角度数组 (5个关节)
        返回: 4x4 齐次变换矩阵
        """
        T = np.eye(4)
        for i, dh in enumerate(self.dh_table):
            T = T @ calc_dh_matrix(dh, q_dh[i])
        return self.worldTbase @ T @ self.nTtool
    
    def inverse_kinematics(self, q_start_mech, target_position, use_orientation=False, k=0.5, n_iter=100):
        """
        逆向运动学: 根据目标位置计算关节角度
        
        q_start_mech: 起始机械角度
        target_position: 目标位置 [x, y, z]
        use_orientation: 是否使用姿态控制
        k: 阻尼系数
        n_iter: 最大迭代次数
        
        返回: 机械关节角度数组
        """
        # 转换起始角度为 DH 角度
        q_dh = self.mechanical_to_dh(q_start_mech)[:5]
        
        # 构建目标变换矩阵 (只控制位置)
        desired_T = np.eye(4)
        desired_T[:3, 3] = target_position
        
        # 转换到 baseTn 坐标系
        desired_baseTn = inv_homog_mat(self.worldTbase) @ desired_T @ inv_homog_mat(self.nTtool)
        
        # 迭代求解
        for _ in range(n_iter):
            # 当前正向运动学
            T_current = np.eye(4)
            for i, dh in enumerate(self.dh_table):
                T_current = T_current @ calc_dh_matrix(dh, q_dh[i])
            
            # 计算误差
            err_lin = calc_lin_err(T_current, desired_baseTn)
            
            # 检查是否收敛
            if np.linalg.norm(err_lin) < 1e-4:
                break
            
            # 计算几何雅可比矩阵
            J = self._compute_jacobian(q_dh)
            
            # 阻尼最小二乘 (只用位置部分)
            J_pos = J[:3, :]
            J_pinv = dls_right_pseudoinv(J_pos)
            delta_q = k * J_pinv @ err_lin
            
            # 更新角度
            q_dh = q_dh - delta_q
        
        # 转换回机械角度 (已经是6个元素，包含夹爪)
        q_mech = self.dh_to_mechanical(q_dh)
        
        return q_mech
    
    def _compute_jacobian(self, q_dh):
        """计算几何雅可比矩阵 (简化版)"""
        n_joints = len(self.dh_table)
        J = np.zeros((6, n_joints))
        
        # 计算当前末端位置
        T_middle = [np.eye(4)]
        for i, dh in enumerate(self.dh_table):
            T_middle.append(T_middle[-1] @ calc_dh_matrix(dh, q_dh[i]))
        
        end_effector = T_middle[-1] @ self.nTtool
        p_end = end_effector[:3, 3]
        
        # 计算每个关节的雅可比列
        for i in range(n_joints):
            # 关节 i 的旋转轴
            if i == 0:
                z_axis = np.array([0, 0, 1])
                pJoint = np.zeros(3)
            else:
                T_i = T_middle[i]
                z_axis = T_i[:3, 2]
                pJoint = T_i[:3, 3]
            
            # 线速度部分
            J[:3, i] = np.cross(z_axis, p_end - pJoint)
            # 角速度部分
            J[3:, i] = z_axis
        
        return J
    
    def check_joint_limits(self, q_mech):
        """检查关节角度是否超出限制"""
        for i in range(len(q_mech)):
            if q_mech[i] < self.mech_limits_low[i] or q_mech[i] > self.mech_limits_up[i]:
                raise ValueError(f"关节 {i} 角度 {np.degrees(q_mech[i]):.1f}° 超出限制 [{np.degrees(self.mech_limits_low[i]):.1f}°, {np.degrees(self.mech_limits_up[i]):.1f}°]")


# 测试代码
if __name__ == "__main__":
    kin = SO101Kinematics()
    
    # 初始位置 (机械角度)
    q_init = np.array([-np.pi/2, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, 0])
    print(f"初始机械角度: {np.degrees(q_init)}")
    
    # 正向运动学
    T_start = kin.forward_kinematics(kin.mechanical_to_dh(q_init))
    print(f"初始末端位置: {T_start[:3, 3]}")
    
    # 测试位置
    target = np.array([0.0, 0.0, 0.1])  # 目标位置
    print(f"\n目标位置: {target}")
    
    try:
        q_solution = kin.inverse_kinematics(q_init, target)
        print(f"求解角度: {np.degrees(q_solution)}")
        
        # 验证
        T_verify = kin.forward_kinematics(kin.mechanical_to_dh(q_solution))
        print(f"验证位置: {T_verify[:3, 3]}")
    except Exception as e:
        print(f"求解失败: {e}")
