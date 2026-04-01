import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_rviz = LaunchConfiguration("use_rviz")

    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Enable or disable RViz",
    )

    startup_launch = os.path.join(
        get_package_share_directory("kuka_rsi_driver"),
        "launch",
        "startup.launch.py",
    )

    # These assume your custom config files are installed with kuka_rsi_driver
    arm1_controller_config = os.path.join(
        get_package_share_directory("kuka_rsi_driver"),
        "config",
        "ros2_controller_config_rsi_only_arm1.yaml",
    )
    arm1_jtc_config = os.path.join(
        get_package_share_directory("kuka_rsi_driver"),
        "config",
        "joint_trajectory_controller_config_arm1.yaml",
    )

    arm2_controller_config = os.path.join(
        get_package_share_directory("kuka_rsi_driver"),
        "config",
        "ros2_controller_config_rsi_only_arm2.yaml",
    )
    arm2_jtc_config = os.path.join(
        get_package_share_directory("kuka_rsi_driver"),
        "config",
        "joint_trajectory_controller_config_arm2.yaml",
    )

    # Replace this if your custom RViz config lives in a different package/file
    rviz_config = os.path.join(
        get_package_share_directory("kuka_resources"),
        "config",
        "view_multi_KSS_urdf.rviz",
    )

    arm1_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(startup_launch),
        launch_arguments={
            "namespace": "arm1",
            "driver_version": "rsi_only",
            "mode": "hardware",
            "client_ip": "192.168.253.100",
            "client_port": "59152",
            "controller_ip": "192.168.253.10",
            "robot_model": "kr50_r2100",
            "use_gpio": "false",
            "robot_family": "radars",
            "controller_config": arm1_controller_config,
            "jtc_config": arm1_jtc_config,
        }.items(),
    )

    arm2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(startup_launch),
        launch_arguments={
            "namespace": "arm2",
            "driver_version": "rsi_only",
            "mode": "hardware",
            "client_ip": "192.168.253.100",
            "client_port": "59153",
            "controller_ip": "192.168.253.20",
            "robot_model": "kr50_r2100",
            "use_gpio": "false",
            "robot_family": "radars",
            "controller_config": arm2_controller_config,
            "jtc_config": arm2_jtc_config,
            "y": "3.170",
        }.items(),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="multi_robot_rviz",
        arguments=["-d", rviz_config],
        output="screen",
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        use_rviz_arg,
        arm1_launch,
        arm2_launch,
        rviz_node,
    ])