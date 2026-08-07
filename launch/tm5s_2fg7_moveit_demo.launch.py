"""MoveIt demo launch using the generated TM5S + OnRobot 2FG7 URDF."""

from __future__ import annotations

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch, generate_rsp_launch


ARENA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_URDF = ARENA_DIR / "generated/tm5s_with_2fg7.urdf"
DEFAULT_RESULT = ARENA_DIR / "outputs/demo_seed7/result.json"
DEFAULT_RVIZ = ARENA_DIR / "config/tm5s_2fg7_moveit_pin_demo.rviz"


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def launch_setup(context, *args, **kwargs):
    urdf_path = LaunchConfiguration("urdf_path").perform(context)
    result_json = LaunchConfiguration("result_json").perform(context)
    publish_alignment = _truthy(LaunchConfiguration("publish_alignment").perform(context))

    moveit_config = (
        MoveItConfigsBuilder("tm5s", package_name="tm5s_moveit_config")
        .robot_description(file_path=urdf_path)
        .to_moveit_configs()
    )
    actions = []
    actions.append(
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="static_transform_publisher0",
            output="screen",
            arguments=["0", "0", "0", "0", "0", "0", "world", "base"],
        )
    )
    actions.extend(generate_rsp_launch(moveit_config).entities)
    actions.extend(generate_move_group_launch(moveit_config).entities)
    actions.append(
        Node(
            package="rviz2",
            executable="rviz2",
            output="log",
            respawn=False,
            arguments=["-d", LaunchConfiguration("rviz_config")],
            parameters=[
                moveit_config.planning_pipelines,
                moveit_config.robot_description_kinematics,
                moveit_config.joint_limits,
            ],
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        )
    )

    if publish_alignment and result_json and Path(result_json).exists():
        actions.extend(
            [
                ExecuteProcess(
                    cmd=[
                        "/usr/bin/python3",
                        str(ARENA_DIR / "scripts/publish_pin_scene.py"),
                        result_json,
                        "--cloud-ply",
                        str(Path(result_json).with_name("scene_cloud.ply")),
                        "--frame-id",
                        "base",
                    ],
                    output="screen",
                ),
                ExecuteProcess(
                    cmd=[
                        "/usr/bin/python3",
                        str(ARENA_DIR / "scripts/publish_alignment_frames.py"),
                        result_json,
                        "--end-effector-link",
                        "flange",
                        "--frame-id",
                        "base",
                    ],
                    output="screen",
                ),
            ]
        )
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("urdf_path", default_value=str(DEFAULT_URDF)),
            DeclareLaunchArgument("result_json", default_value=str(DEFAULT_RESULT)),
            DeclareLaunchArgument("rviz_config", default_value=str(DEFAULT_RVIZ)),
            DeclareLaunchArgument("publish_alignment", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            OpaqueFunction(function=launch_setup),
        ]
    )
