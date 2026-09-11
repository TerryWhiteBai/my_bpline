import importlib.util
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from launch.actions import DeclareLaunchArgument

import yaml


SOURCE_ROOT = Path(__file__).resolve().parents[1]
XACRO = SOURCE_ROOT / "urdf" / "marvin.urdf.xacro"


def _expand_robot(*xacro_args, xacro_path=XACRO):
    result = subprocess.run(
        [
            "xacro",
            str(xacro_path),
            "ros2_control:=true",
            "use_fake_hardware:=true",
            *xacro_args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return ET.fromstring(result.stdout)


def _matmul(a, b):
    return [
        [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)
    ]


def _rpy_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _axis_matrix(axis, angle):
    norm = math.sqrt(sum(value * value for value in axis))
    x, y, z = (value / norm for value in axis)
    c, s, one_minus_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return [
        [
            c + x * x * one_minus_c,
            x * y * one_minus_c - z * s,
            x * z * one_minus_c + y * s,
        ],
        [
            y * x * one_minus_c + z * s,
            c + y * y * one_minus_c,
            y * z * one_minus_c - x * s,
        ],
        [
            z * x * one_minus_c - y * s,
            z * y * one_minus_c + x * s,
            c + z * z * one_minus_c,
        ],
    ]


def _transform(rotation, translation=(0.0, 0.0, 0.0)):
    return [
        [rotation[0][0], rotation[0][1], rotation[0][2], translation[0]],
        [rotation[1][0], rotation[1][1], rotation[1][2], translation[1]],
        [rotation[2][0], rotation[2][1], rotation[2][2], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _urdf_fk(root, base, tip, joints_deg):
    by_child = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        xyz = (
            [0.0, 0.0, 0.0]
            if origin is None
            else [float(v) for v in origin.get("xyz", "0 0 0").split()]
        )
        rpy = (
            [0.0, 0.0, 0.0]
            if origin is None
            else [float(v) for v in origin.get("rpy", "0 0 0").split()]
        )
        axis_tag = joint.find("axis")
        axis = (
            [0.0, 0.0, 1.0]
            if axis_tag is None
            else [float(v) for v in axis_tag.get("xyz").split()]
        )
        by_child[joint.find("child").get("link")] = (
            joint.find("parent").get("link"),
            joint.get("type"),
            xyz,
            rpy,
            axis,
        )

    chain = []
    current = tip
    while current != base:
        chain.append(by_child[current])
        current = by_child[current][0]

    result = _transform(_rpy_matrix(0.0, 0.0, 0.0))
    joint_index = 0
    for _, joint_type, xyz, rpy, axis in reversed(chain):
        result = _matmul(result, _transform(_rpy_matrix(*rpy), xyz))
        if joint_type == "revolute":
            result = _matmul(
                result,
                _transform(_axis_matrix(axis, math.radians(joints_deg[joint_index]))),
            )
            joint_index += 1
    return result


def _pose_error(actual, expected):
    position_error = math.sqrt(
        sum((actual[i][3] - expected[i][3]) ** 2 for i in range(3))
    )
    relative_trace = sum(
        expected[k][i] * actual[k][i] for i in range(3) for k in range(3)
    )
    rotation_error = math.acos(max(-1.0, min(1.0, (relative_trace - 1.0) / 2.0)))
    return position_error, rotation_error


def test_bimanual_arm_only_control_contract():
    root = _expand_robot()
    links = {link.get("name") for link in root.findall("link")}
    revolute = [
        joint for joint in root.findall("joint") if joint.get("type") == "revolute"
    ]
    controls = root.findall("ros2_control")

    assert len(revolute) == 14
    assert {"flange_L", "flange_R"} <= links
    assert {"left_pika_gripper_tcp", "right_pika_gripper_tcp"}.isdisjoint(links)
    assert {"left_gripper_base_link", "right_gripper_base_link"}.isdisjoint(links)
    assert {"camera_link", "left_tool", "right_tool"}.isdisjoint(links)
    assert len(controls) == 1
    assert len(controls[0].findall("joint")) == 14
    assert len(root.findall(".//command_interface[@name='position']")) == 14


def test_arm_mount_transforms_are_configurable():
    root = _expand_robot(
        "left_base_xyz:=0.1 0.2 0.3",
        "left_base_rpy:=0.01 0.02 0.03",
        "right_base_xyz:=-0.1 -0.2 -0.3",
        "right_base_rpy:=-0.01 -0.02 -0.03",
    )
    joints = {joint.get("name"): joint for joint in root.findall("joint")}

    assert joints["base_to_left_arm"].find("origin").get("xyz") == "0.1 0.2 0.3"
    assert joints["base_to_left_arm"].find("origin").get("rpy") == "0.01 0.02 0.03"
    assert joints["base_to_right_arm"].find("origin").get("xyz") == "-0.1 -0.2 -0.3"
    assert joints["base_to_right_arm"].find("origin").get("rpy") == "-0.01 -0.02 -0.03"


def test_real_hardware_safety_parameters_are_configurable():
    root = _expand_robot(
        "use_fake_hardware:=false",
        "robot_ip:=192.0.2.10",
        "stale_warn_ms:=25.0",
        "stale_error_ms:=125.0",
        "max_joint_velocity:=1.5",
    )
    hardware = root.find("ros2_control/hardware")
    params = {param.get("name"): param.text for param in hardware.findall("param")}

    assert hardware.find("plugin").text == (
        "marvin_hardware_interface/MarvinBimanualArmHardware"
    )
    assert params == {
        "use_fake_hardware": "false",
        "robot_ip": "192.0.2.10",
        "stale_warn_ms": "25.0",
        "stale_error_ms": "125.0",
        "max_joint_velocity": "1.5",
    }


def _xyz_tuple(text):
    return tuple(float(part) for part in str(text).split())


def test_stand_yaml_dimensions_are_expanded():
    stand = yaml.safe_load(
        (SOURCE_ROOT / "config" / "stand.yaml").read_text(encoding="utf-8")
    )
    root = _expand_robot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    column = next(
        link.find("collision/geometry/cylinder")
        for link in root.findall("link")
        if link.get("name") == "column_link"
    )

    assert float(column.get("radius")) == float(stand["column"]["radius"])
    assert float(column.get("length")) == float(stand["column"]["length"])
    assert _xyz_tuple(joints["base_to_column"].find("origin").get("xyz")) == _xyz_tuple(
        stand["column"]["origin"]["xyz"]
    )
    assert _xyz_tuple(
        joints["base_to_tracking_base"].find("origin").get("xyz")
    ) == _xyz_tuple(stand["tracking_base"]["origin"]["xyz"])


def test_yaml_limits_and_mirrored_inertials_are_expanded():
    root = _expand_robot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    links = {link.get("name"): link for link in root.findall("link")}

    expected_limits = {
        1: (-2.9671, 2.9671, 108.0),
        2: (-2.0944, 2.0944, 108.0),
        3: (-2.9671, 2.9671, 66.0),
        4: (-2.5307, 1.0472, 66.0),
        5: (-2.9671, 2.9671, 18.0),
        6: (-1.0472, 1.0472, 18.0),
        7: (-1.5708, 1.5708, 18.0),
    }
    for side in ("L", "R"):
        for number, expected in expected_limits.items():
            limit = joints[f"Joint{number}_{side}"].find("limit")
            actual = tuple(
                float(limit.get(key)) for key in ("lower", "upper", "effort")
            )
            assert actual == expected
            assert float(limit.get("velocity")) == 3.1416

    left_inertia = links["Link5_L"].find("inertial/inertia")
    right_inertia = links["Link5_R"].find("inertial/inertia")
    assert float(left_inertia.get("ixz")) == -float(right_inertia.get("ixz"))
    assert float(left_inertia.get("iyz")) == -float(right_inertia.get("iyz"))


def test_flange_fk_matches_vendor_m6_40_snapshots():
    root = _expand_robot()
    samples = [
        (
            [0.0] * 7,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.8705],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ),
        (
            [10.0, -20.0, 30.0, -40.0, 20.0, 15.0, -10.0],
            [
                [0.2834872271, -0.7945993370, 0.5368862876, 0.3798317353],
                [0.7779938499, 0.5178909255, 0.3556888511, 0.1963841623],
                [-0.5606786616, 0.3168609838, 0.7650088597, 0.6927411839],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ),
        (
            [-25.0, 35.0, -45.0, -30.0, 40.0, -20.0, 25.0],
            [
                [0.9409637083, 0.2995548076, -0.1576522022, -0.2741623466],
                [-0.3204777768, 0.6383677193, -0.6998433035, -0.0699988263],
                [-0.1090013494, 0.7090511774, 0.6966815152, 0.7705614834],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ),
    ]

    for base, tip in (("Base_L", "flange_L"), ("Base_R", "flange_R")):
        for joints, expected in samples:
            position_error, rotation_error = _pose_error(
                _urdf_fk(root, base, tip, joints), expected
            )
            assert position_error < 5e-6
            assert rotation_error < 3e-5


def test_visualize_launch_defers_model_poses_to_xacro():
    path = SOURCE_ROOT / "launch" / "visualize_marvin.launch.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    defaults = {}
    for entity in module.generate_launch_description().entities:
        if isinstance(entity, DeclareLaunchArgument):
            parts = entity.default_value or []
            defaults[entity.name] = "".join(
                getattr(part, "text", "") for part in parts
            )
    for name in ("connected_to", "xyz", "rpy", "mounts_file"):
        assert defaults[name] == ""
    source = path.read_text(encoding="utf-8")
    assert 'mappings["left_base_xyz"]' not in source
    assert 'mappings["right_base_xyz"]' not in source
