
import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, "r") as file:
            return yaml.safe_load(file)
    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        return None


def launch_setup(context, *args, **kwargs):
    robot_model = LaunchConfiguration("robot_model")
    robot_family_path = LaunchConfiguration("robot_family_support")
    ns = LaunchConfiguration("namespace")
    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    z = LaunchConfiguration("z")
    roll = LaunchConfiguration("roll")
    pitch = LaunchConfiguration("pitch")
    yaw = LaunchConfiguration("yaw")

    if ns.perform(context) == "":
        tf_prefix = ""
    else:
        tf_prefix = ns.perform(context) + "_"

    moveit_config = (
        MoveItConfigsBuilder("kuka_kr")
        .robot_description(
            file_path=get_package_share_directory(robot_family_path.perform(context))
            + f"/urdf/{robot_model.perform(context)}.urdf.xacro",
            mappings={
                "x": x.perform(context),
                "y": y.perform(context),
                "z": z.perform(context),
                "roll": roll.perform(context),
                "pitch": pitch.perform(context),
                "yaw": yaw.perform(context),
                "prefix": tf_prefix,
            },
        )
        .robot_description_semantic(
            f"urdf/{robot_model.perform(context)}_arm.srdf"
        )
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_scene_monitor(
            publish_robot_description=True, publish_robot_description_semantic=True
        )
        .joint_limits(
            file_path=get_package_share_directory(robot_family_path.perform(context))
            + f"/config/{robot_model.perform(context)}_joint_limits.yaml"
        )
        .to_moveit_configs()

    )

    rviz_config_file = (
        get_package_share_directory("kuka_resources") + "/config/planning_6_axis.rviz"
    )

    robot_description_kinematics = {
        "robot_description_kinematics": {
            "manipulator": {"kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin"}
        }
    }

    # Planning Functionality
    planning_pipelines_config = {
        "default_planning_pipeline": "ompl",
        "planning_pipelines": ["pilz", "ompl"],
        "pilz": {
            "planning_plugin": "pilz_industrial_motion_planner/CommandPlanner",
            "request_adapters": "",
            "start_state_max_bounds_error": 0.1,
        },
        "ompl": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": """default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints""",
            "start_state_max_bounds_error": 0.1,
        },
    }

    ompl_planning_yaml = load_yaml(
        "moveit_resources_prbt_moveit_config", "config/ompl_planning.yaml"
    )
    planning_pipelines_config["ompl"].update(ompl_planning_yaml)

    move_group_server = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"publish_planning_scene_hz": 30.0},
            {"allow_trajectory_execution": True},
            {"publish_planning_scene": True},
            {"publish_state_updates": True},
            {"publish_transforms_updates": True},
            {"planning_pipelines": ["ompl", "pilz_industrial_motion_planner",]},
            planning_pipelines_config
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file, "--ros-args", "--log-level", "error"],
        parameters=[robot_description_kinematics],
    )

    # Add driver node (C++ executable)
    driver_node = Node(
        package="kuka_kr70_moveit_rsi",
        executable="kr70_moveitcpp_driver", 
        output="screen",
        parameters=[moveit_config.to_dict()],  # <-- this line loads all the robot/kinematics params into node
    )


    return [move_group_server, rviz, driver_node]

def generate_launch_description():
    launch_arguments = [
        DeclareLaunchArgument("robot_model", default_value="kr70_r2100"),
        DeclareLaunchArgument("robot_family_support", default_value="kuka_iontec_support"),
        DeclareLaunchArgument("namespace", default_value=""),
        DeclareLaunchArgument("x", default_value="0"),
        DeclareLaunchArgument("y", default_value="0"),
        DeclareLaunchArgument("z", default_value="0"),
        DeclareLaunchArgument("roll", default_value="0"),
        DeclareLaunchArgument("pitch", default_value="0"),
        DeclareLaunchArgument("yaw", default_value="0"),
    ]
    return LaunchDescription(launch_arguments + [OpaqueFunction(function=launch_setup)])
