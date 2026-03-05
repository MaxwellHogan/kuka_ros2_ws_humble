cmake_minimum_required(VERSION 3.8)
project(string_republisher)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(string_republisher_node src/string_republisher_node.cpp)
ament_target_dependencies(string_republisher_node rclcpp std_msgs)

install(TARGETS
  string_republisher_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
