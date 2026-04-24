#!/usr/bin/env python3
"""检查机械臂末端执行器位置"""

import numpy as np
import placo

# URDF路径
urdf_path = '/home/whr/cc_ws/tros_ws/install/weed_locator/share/weed_locator/config/SO101/so101_new_calib.urdf'

print(f"Loading robot from: {urdf_path}")
robot = placo.RobotWrapper(urdf_path)

print(f"\n=== Robot model info ===")
print(f"model.nq = {robot.model.nq}")

# 初始化状态为0
robot.state.q = np.zeros(13)
robot.state.q[6] = 1.0  # 四元数w
robot.update_kinematics()

# 获取末端执行器位置
T = robot.get_T_a_b("universe", "gripper_link")
print(f"\n=== End effector at zero position ===")
print(f"Position (x,y,z): {T[:3, 3]}")
print(f"Rotation matrix:\n{T[:3, :3]}")

# 测试不同的目标位置
print(f"\n=== Testing target positions ===")
test_positions = [
    [0.05, 0, 0.1],
    [0.1, 0, 0.1],
    [0.15, 0, 0.1],
    [0.2, 0, 0.1],
]

solver = placo.KinematicsSolver(robot)
solver.mask_fbase(True)

for target in test_positions:
    solver.clear()
    task = solver.add_position_task('gripper_link', np.array(target))
    q_solution = solver.solve(False)
    
    if len(q_solution) == 12:
        q_solution = np.insert(q_solution, 6, 1.0)
    
    robot.state.q = q_solution
    robot.update_kinematics()
    
    T_result = robot.get_T_a_b("universe", "gripper_link")
    actual = T_result[:3, 3]
    error = np.linalg.norm(actual - np.array(target))
    
    print(f"Target: {target} -> Actual: {actual}, Error: {error:.4f}m")

print("\n=== DONE ===")
