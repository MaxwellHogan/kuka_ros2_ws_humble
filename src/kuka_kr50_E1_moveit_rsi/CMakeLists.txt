cmake_minimum_required(VERSION 3.8)
project(kuka_kr50_E1_moveit_rsi)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_core REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(kr50_E1_moveitcpp_driver src/kr50_E1_moveitcpp_driver.cpp)
add_executable(kr50_E1_benchmark src/kr50_E1_benchmark.cpp)

ament_target_dependencies(
  kr50_E1_moveitcpp_driver
  rclcpp
  moveit_ros_planning_interface
  moveit_core
  geometry_msgs
  sensor_msgs
)

ament_target_dependencies(
  kr50_E1_benchmark
  rclcpp
  moveit_ros_planning_interface
  moveit_core
  geometry_msgs
  sensor_msgs
)

install(TARGETS kr50_E1_moveitcpp_driver
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS kr50_E1_benchmark
  DESTINATION lib/${PROJECT_NAME}
)

# install(TARGETS kr50_E1_moveitcpp_driver_v1
#   DESTINATION lib/${PROJECT_NAME}
# )

install(PROGRAMS
  scripts/move_done_listener.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
