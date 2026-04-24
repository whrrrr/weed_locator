cmake_minimum_required(VERSION 3.8)
project(weed_locator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  srv/ReadJoints.srv
  srv/WriteJoints.srv
  srv/MoveJoint.srv
  srv/SolveIK.srv
  DEPENDENCIES geometry_msgs
)

# 安装 Python 包到正确的 site-packages 目录
install(DIRECTORY weed_locator/
  DESTINATION lib/python3.10/site-packages/weed_locator
  FILES_MATCHING PATTERN "*.py"
)

# 同时安装到 local 目录（ROS2 Python 运行时查找的位置）
install(DIRECTORY weed_locator/
  DESTINATION local/lib/python3.10/dist-packages/weed_locator
  FILES_MATCHING PATTERN "*.py"
)

# 复制 .so 文件到 Python site-packages（编译后自动执行）
install(CODE "
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_py/weed_locator
    ${CMAKE_INSTALL_PREFIX}/lib/python3.10/site-packages/weed_locator
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_py/weed_locator
    ${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/weed_locator
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/libweed_locator__rosidl_typesupport_c.so
    ${CMAKE_INSTALL_PREFIX}/lib/python3.10/site-packages/weed_locator/
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/libweed_locator__rosidl_typesupport_introspection_c.so
    ${CMAKE_INSTALL_PREFIX}/lib/python3.10/site-packages/weed_locator/
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/libweed_locator__rosidl_typesupport_fastrtps_c.so
    ${CMAKE_INSTALL_PREFIX}/lib/python3.10/site-packages/weed_locator/
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/libweed_locator__rosidl_typesupport_cpp.so
    ${CMAKE_INSTALL_PREFIX}/lib/python3.10/site-packages/weed_locator/
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/libweed_locator__rosidl_typesupport_introspection_cpp.so
    ${CMAKE_INSTALL_PREFIX}/lib/python3.10/site-packages/weed_locator/
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/libweed_locator__rosidl_typesupport_fastrtps_cpp.so
    ${CMAKE_INSTALL_PREFIX}/lib/python3.10/site-packages/weed_locator/
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/libweed_locator__rosidl_generator_c.so
    ${CMAKE_INSTALL_PREFIX}/lib/python3.10/site-packages/weed_locator/
  )
")

# 安装可执行脚本（去掉 .py 后缀）
install(PROGRAMS weed_locator/dynamixel_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME dynamixel_node
)
install(PROGRAMS weed_locator/weed_fusion_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME weed_fusion_node
)
install(PROGRAMS weed_locator/ik_service.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME ik_service
)

# 安装 scripts 目录下的 Python 脚本
install(PROGRAMS scripts/hand_eye_calibration.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME hand_eye_calibration
)
install(PROGRAMS scripts/target_tracking_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME target_tracking_node
)
install(PROGRAMS scripts/arm_keyboard_control.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME arm_keyboard_control
)
install(PROGRAMS scripts/test_ik_service.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME test_ik_service
)
install(PROGRAMS scripts/fake_dynamixel_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME fake_dynamixel_node
)
install(PROGRAMS scripts/joint_state_bridge.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME joint_state_bridge
)

# 安装 launch 文件
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# 安装 config 文件 (包括 SO101 URDF 和 mesh) - 安装到share目录供package://使用
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
