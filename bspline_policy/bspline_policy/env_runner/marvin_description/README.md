# marvin_description

ROS 2 description package for the Marvin CCS M6 bimanual arm embodiment.

This package contains the canonical arm-only URDF/xacro model, meshes, and
description-only launch files. It is intentionally hardware-free: no SDK
transport, no controller manager launch, no gripper hardware, and no real motion
commands live here.

## Scope

Current model:

- Marvin CCS M6 bimanual arm, controller type `1017`
- 14 revolute arm joints: `Joint1_L..Joint7_L` and `Joint1_R..Joint7_R`
- Native arm flange frames: `flange_L` and `flange_R`
- Stand/base frames: `base_link`, `column_link`, `tracking_base_link`
- Dual-Pika assembly lives in `marvin_manipulation_rt_launch`
  (`urdf/marvin_manipulation.urdf.xacro`), not here

Deferred modules:

- MoveIt configuration
- custom controllers from the higher-level control stack

## Package Layout

```text
marvin_description/
  config/
    arm_mounts.yaml
    stand.yaml
    inertials.yaml
    joint_limits.yaml
    kinematics.yaml
    curobo/
      marvin.yml              # arm-only; dual-Pika YAML in curobo_robot_models
  docs/
    CUROBO_ROBOT_MODEL.md
  launch/
    visualize_marvin.launch.py
  meshes/
    base/
    m6/
  scripts/
    generate_curobo_robot_model.sh
  rviz/
    visualize_marvin.rviz
  test/
    test_marvin_description.py
  urdf/
    marvin.urdf.xacro
    parts/
      marvin_stand.xacro
      marvin_left_arm.xacro
      marvin_right_arm.xacro
      marvin_bimanual_arm.xacro
      marvin_arm.ros2_control.xacro
```

## Xacro Entry Points

- `urdf/marvin.urdf.xacro` — arm-only (ends at `flange_L` / `flange_R`)
- Dual Pika: `marvin_manipulation_rt_launch/urdf/marvin_manipulation.urdf.xacro`

Shared stand/arm arguments:

```text
connected_to:=world
xyz:=0 0 0
rpy:=0 0 0
mounts_file:=$(find marvin_description)/config/arm_mounts.yaml
ros2_control:=true
use_fake_hardware:=true
hardware_plugin:=marvin_hardware_interface/MarvinBimanualArmHardware
robot_ip:=10.19.0.191
stale_warn_ms:=20.0
stale_error_ms:=100.0
max_joint_velocity:=6.2832
```

`config/arm_mounts.yaml` is the canonical calibration source for the fixed
`base_link -> Base_L` and `base_link -> Base_R` transforms. Translation values
are metres and RPY values are radians. The individual `left_base_*` and
`right_base_*` xacro arguments remain available for explicit one-off overrides.

`config/stand.yaml` is the canonical source for stand site geometry: column
radius/length and the fixed `base_link -> column_link` /
`base_link -> tracking_base_link` offsets.

The native M6 arm parameters are separated by responsibility:

- `config/kinematics.yaml`: joint and flange origins plus joint axes.
- `config/joint_limits.yaml`: position, velocity, and effort limits.
- `config/inertials.yaml`: CAD center of mass, mass, and inertia tensors,
  including the mirrored left/right Link5 and Link6 values.

The expected tree is:

```text
world
  -> base_link
      -> column_link
      -> tracking_base_link
      -> Base_L -> ... -> Link7_L -> flange_L
      -> Base_R -> ... -> Link7_R -> flange_R
```

`flange_L` / `flange_R` are native arm frames. Dual-Pika mounts and TCP frames
are owned by `marvin_manipulation_rt_launch` +
`pika_gripper_description`.

## ros2_control Block

`urdf/parts/marvin_arm.ros2_control.xacro` exposes only the 14 arm joints.

Command interfaces:

```text
position
```

The validated hardware command path is position-only.

State interfaces:

```text
position
velocity
effort
```

Hardware selection follows the fake/real switch pattern:

```text
use_fake_hardware:=true   -> mock_components/GenericSystem
use_fake_hardware:=false  -> hardware_plugin
```

The default is fake hardware. This package must not connect to the vendor SDK.
For real hardware, the description forwards the controller address and the
time-based feedback/write guards into the hardware plugin. Higher-level RT
bringup packages may override them without editing this package; the defaults
match the values already validated on the robot.

## Build

From a ROS 2 workspace containing this package:

```bash
colcon build --symlink-install --packages-select marvin_description --cmake-args -Wno-dev
source install/setup.bash
```

## Validate The URDF

```bash
xacro $(ros2 pkg prefix marvin_description)/share/marvin_description/urdf/marvin.urdf.xacro \
  ros2_control:=true \
  use_fake_hardware:=true \
  > /tmp/marvin.urdf

check_urdf /tmp/marvin.urdf
```

Expected (arm-only `marvin.urdf.xacro`):

- xacro expands successfully
- `check_urdf` succeeds
- one connected tree from `world`
- `flange_L` and `flange_R` exist
- 14 revolute arm joints
- no gripper joints

## Visualize

```bash
# Arm only
ros2 launch marvin_description visualize_marvin.launch.py

# Dual Pika (bringup package)
ros2 launch marvin_manipulation_rt_launch visualize_marvin_manipulation.launch.py
```

This launch starts `robot_state_publisher`, `joint_state_publisher` or
`joint_state_publisher_gui`, and RViz2. It does not start ros2_control or any
hardware driver.

## cuRobo collision config

Arm-only YAML (`config/curobo/marvin.yml`) is generated here. Dual-Pika
manipulation YAML lives in `curobo_robot_models`
(`config/marvin_manipulation.yml`). See this package's
**[docs/CUROBO_ROBOT_MODEL.md](docs/CUROBO_ROBOT_MODEL.md)** for arm-only
generation, and `curobo_robot_models` for manipulation.

Authority (arm-only): URDF/xacro + meshes + `config/*.yaml`. Derived artifact:
`config/curobo/marvin.yml`.

### Complete workflow (arm-only)

```bash
pixi shell
source install/setup.bash
cd src/embodiments/robots/marvin/marvin_description
bash scripts/generate_curobo_robot_model.sh
bash scripts/generate_curobo_robot_model.sh --visualize-only
```

For dual-Pika:

```bash
cd src/motion_planning/motion_planners/curobo_robot_models
bash scripts/generate_curobo_robot_model.sh --model marvin
```

Point adapters at `curobo_robot_models/config/marvin_manipulation.yml` for
manipulation planning.

## Real Hardware Notes

The real controller currently uses the Marvin CCS M6 4.0 SDK configuration.
Future SDK kinematics, IK, calibration, and real read/write gates should use
`ccs_m6_40.MvKDCfg`, not older 3.1/reference configuration files.

The native flange transform is aligned with `ccs_m6_40.MvKDCfg`, including its
terminal MDH segment `90, 0, 95, 90`. Offline FK comparison at zero and two
non-zero joint configurations agrees within `3e-6 m` and `2.3e-5 rad`.

`m_FB_Joint_SToq` is documented by the vendor only as feedback joint torque. Its
unit and scaling are not specified. The exposed effort state is therefore
provisional raw feedback and must not be used for force/torque control until a
known-load calibration confirms SI `N*m` semantics.
