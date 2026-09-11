# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Marvin description-only RViz visualization launch.

Starts robot_state_publisher, joint_state_publisher (or gui), and RViz2.
No ros2_control, no hardware, no controllers.

Arm-only entry (`marvin.urdf.xacro`). Dual-Pika visualization lives in
`marvin_manipulation_rt_launch`.
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
import os
import sys
import xacro
import yaml


def _mount_value(config, arm, field):
    try:
        value = str(config[arm]["origin"][field])
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Missing {arm}.origin.{field} in arm mounts YAML") from exc
    if len(value.split()) != 3:
        raise RuntimeError(f"{arm}.origin.{field} must contain exactly three values")
    return value


def robot_state_publisher_spawner(context: LaunchContext):
    share = get_package_share_directory("marvin_description")

    use_joint_state_gui = LaunchConfiguration("use_joint_state_gui")
    joint_states_topic = LaunchConfiguration("joint_states_topic")
    mounts_file = context.perform_substitution(LaunchConfiguration("mounts_file"))
    if not mounts_file:
        mounts_file = os.path.join(share, "config", "arm_mounts.yaml")

    with open(mounts_file, "r", encoding="utf-8") as stream:
        mounts = yaml.safe_load(stream)

    left_base_xyz = _mount_value(mounts, "left_arm", "xyz")
    left_base_rpy = _mount_value(mounts, "left_arm", "rpy")
    right_base_xyz = _mount_value(mounts, "right_arm", "xyz")
    right_base_rpy = _mount_value(mounts, "right_arm", "rpy")

    base_xacro = str(PathJoinSubstitution([share, "urdf", "marvin.urdf.xacro"]).perform(context))

    mappings = {
        "ros2_control": "false",
        "use_fake_hardware": "true",
    }
    for name in ("connected_to", "xyz", "rpy"):
        value = context.perform_substitution(LaunchConfiguration(name))
        if value:
            mappings[name] = value
    if context.perform_substitution(LaunchConfiguration("mounts_file")):
        mappings["mounts_file"] = mounts_file

    robot_description = xacro.process_file(base_xacro, mappings=mappings).toprettyxml(
        indent="  "
    )

    return [
        LogInfo(
            msg=(
                f"Loaded arm mounts from {mounts_file}: "
                f"left xyz=[{left_base_xyz}] rpy=[{left_base_rpy}], "
                f"right xyz=[{right_base_xyz}] rpy=[{right_base_rpy}]"
            )
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
            remappings=[("joint_states", joint_states_topic)],
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            name="joint_state_publisher_gui",
            condition=IfCondition(use_joint_state_gui),
            parameters=[{"robot_description": robot_description}],
            remappings=[("joint_states", joint_states_topic)],
            # python_qt_binding expects CONDA_PREFIX, while pixi exposes the
            # environment through sys.prefix instead.
            additional_env={"CONDA_PREFIX": os.environ.get("CONDA_PREFIX", sys.prefix)},
        ),
        Node(
            package="joint_state_publisher",
            executable="joint_state_publisher",
            name="joint_state_publisher",
            condition=UnlessCondition(use_joint_state_gui),
            parameters=[{"robot_description": robot_description}],
            remappings=[("joint_states", joint_states_topic)],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("marvin_description")
    use_rviz = LaunchConfiguration("use_rviz")
    rviz_config = PathJoinSubstitution([share, "rviz", "visualize_marvin.rviz"])

    robot_state_publisher_spawner_opaque_function = OpaqueFunction(
        function=robot_state_publisher_spawner
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["--display-config", rviz_config],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("connected_to", default_value=""),
            DeclareLaunchArgument("xyz", default_value=""),
            DeclareLaunchArgument("rpy", default_value=""),
            DeclareLaunchArgument("mounts_file", default_value=""),
            DeclareLaunchArgument("use_joint_state_gui", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument(
                "joint_states_topic", default_value="/marvin_description/joint_states"
            ),
            robot_state_publisher_spawner_opaque_function,
            rviz,
        ]
    )
