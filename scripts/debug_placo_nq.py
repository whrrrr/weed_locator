#!/usr/bin/env python3
"""调试脚本：检查placo robot的关节结构和nq构成"""

import numpy as np
import placo

# URDF路径
urdf_path = '/home/whr/cc_ws/tros_ws/install/weed_locator/share/weed_locator/config/SO101/so101_new_calib.urdf'

print(f"Loading robot from: {urdf_path}")
robot = placo.RobotWrapper(urdf_path)

print(f"\n=== Robot Info ===")
print(f"model.nq = {robot.model.nq}")
print(f"model.name = {robot.model.name}")

print(f"\n=== State.q ===")
print(f"len(robot.state.q) = {len(robot.state.q)}")
print(f"robot.state.q = {robot.state.q}")

print(f"\n=== Testing mask_fbase and solver ===")
robot.state.q = np.zeros(13)
robot.update_kinematics()

solver = placo.KinematicsSolver(robot)
solver.mask_fbase(True)
print(f"After mask_fbase(True): solver created successfully")

# 再看看solve返回什么
solver.clear()
task = solver.add_position_task('gripper_link', np.array([0.1, 0.0, 0.15]))
q_solution = solver.solve(False)
print(f"solve() returned {len(q_solution)} values: {q_solution}")

print("\n=== DONE ===")
