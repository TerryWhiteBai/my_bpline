# cuRobo robot model generation

This description package is meant to run inside a **ROS 2 + nvidia-curobo**
environment (for Physical AI Runtime: the workspace Pixi shell). The script
[`scripts/generate_curobo_robot_model.sh`](../scripts/generate_curobo_robot_model.sh)
builds a cuRobo robot YAML (collision spheres + self-collision ignore matrix)
from the package URDF/xacro and meshes.

The script is **generic**: keep or copy it into any `xxx_description` package
that follows the usual layout (`package.xml`, `urdf/`, `meshes/`, `config/`).
Package-specific values are passed as flags; Marvin only injects defaults when
the package name is `marvin_description` and frames/clips were not provided.

## What it does

1. Expands the package xacro (`ros2_control:=false` by default) to a temporary URDF
2. Runs `python -m curobo.examples.getting_started.build_robot_model`
3. Writes `config/curobo/<name>.yml` under the **source** package tree
4. Rewrites machine-local absolute paths to portable relative paths

| Artifact | Role |
|----------|------|
| Collision spheres | Per-link MorphIt fit of mesh / collision geometry |
| Self-collision ignore matrix | Sampled link pairs that planning can skip |
| Tool frames | End-effector frames for IK / MPC |

Authority chain:

```text
urdf/ + meshes/ + config/*.yaml     (geometry authority)
            │
            ▼  generate_curobo_robot_model.sh
config/curobo/<name>.yml            (derived; version with geometry)
```

Regenerate after changing meshes, mount/stand calibration YAML, link collision
geometry, or joint limits that affect the ignore matrix.

XRDF (`--export-xrdf`) is optional and **not** required for Physical AI Runtime.
Adapters consume the cuRobo YAML.

## Prerequisites

1. Environment with ROS 2 and `nvidia-curobo` (Pixi shell in this workspace)
2. Description package on `AMENT_PREFIX_PATH` (`ros2 pkg prefix <name>` works)
3. Prefer symlink-install so share tracks source:

```bash
colcon build --symlink-install --packages-select <your_description_pkg>
source install/setup.bash
```

## Complete workflow

The script is a **dev-only tool** — run it from the **package source directory**,
not via `ros2 run`. It is not installed into the workspace.

```bash
# 0. Activate environment and source the workspace
pixi shell
source install/setup.bash

# 1. Enter the package source directory
cd src/embodiments/robots/marvin/marvin_description

# 2. Generate collision YAML (Marvin defaults auto-inject)
bash scripts/generate_curobo_robot_model.sh

# 3. Visualize the generated model (no refit, opens Viser on localhost:8080)
bash scripts/generate_curobo_robot_model.sh --visualize-only
# or with explicit path:
bash scripts/generate_curobo_robot_model.sh --visualize-only config/curobo/marvin.yml

# 4. After iterating, rebuild to deploy config/curobo/*.yml to install space
cd /path/to/workspace
pixi run build
source install/setup.bash
```

Marvin (this repo) defaults when flags are omitted:

- `--tool-frames flange_L flange_R`
- `--clip-link base_link z 0.0`
- `--seed 42` and `--compute-metrics`
- output `config/curobo/marvin.yml` (`marvin_description` → strip `_description`)

Other robots must pass tool frames (and clips if needed) explicitly:

```bash
bash scripts/generate_curobo_robot_model.sh \
  --tool-frames tool0 \
  --clip-link base_link z 0.0
```

## Useful options

| Option | Meaning |
|--------|---------|
| `--package-name NAME` | ROS package name (default: `<name>` from `package.xml`) |
| `--xacro REL_OR_ABS` | Entry xacro (default: first `urdf/*.urdf.xacro`) |
| `--output REL_OR_ABS` | Output YAML (default: `config/curobo/<name>.yml`) |
| `--asset-path DIR` | Override mesh root (parent of `<pkg>/` dirs). **Default:** temp dir with ament share symlinks for every `package://` in the expanded URDF |
| `--tool-frames F [...]` | End-effector / tool frame names |
| `--clip-link L A O` | Clip spheres on link `L` past axis `A` at offset `O` (repeatable) |
| `--xacro-arg KEY:=VAL` | Extra xacro mapping (repeatable) |
| `--visualize` | 生成同时打开 Viser（会重新拟合） |
| `--visualize-only [YML]` | **只可视化已有 YAML**，不重跑 MorphIt（默认 `config/curobo/<name>.yml`） |
| `--viz-port PORT` | Viser port (default: 8080) |
| `--sphere-density N` | More spheres per link (higher = denser) |
| `--seed N` | RNG seed (default 42) |
| `--num-collision-samples N` | Samples for ignore-matrix pruning |
| `--no-prune` | Skip ignore-matrix pruning (faster, less optimized) |
| `--export-xrdf` | Also write XRDF (Isaac / cuMotion) |
| `-- ...` | Extra args forwarded to `build_robot_model` |

```bash
# 生成时可视化
bash scripts/generate_curobo_robot_model.sh --visualize

# 已有 YAML 再看一眼（不 refit）
bash scripts/generate_curobo_robot_model.sh --visualize-only
# 或指定路径:
bash scripts/generate_curobo_robot_model.sh --visualize-only config/curobo/marvin.yml

# 浏览器打开 http://localhost:8080 ，Ctrl+C 结束
```

`--visualize-only` 会：

1. 展开当前 package 的 xacro 到临时 URDF
2. 写一份**临时** YAML，把 `asset_root_path` / `urdf_path` 改成**绝对路径**
3. 交给 cuRobo `RobotBuilder.from_config` + Viser

原因：cuRobo 的 `from_config` 对相对路径一律做
`join_path(get_assets_path(), urdf_path)`，会落到
`…/site-packages/curobo/content/assets/…`，而不是 YAML 旁的 description
树。绝对路径会被 `join_path` 原样保留。仓库里提交的相对路径 YAML **不会被改写**。

```bash
# Denser fit for a single custom robot
bash scripts/generate_curobo_robot_model.sh \
  --package-name my_robot_description \
  --tool-frames ee_link \
  --sphere-density 2.0
```

## Install space and relative paths (robustness)

`CMakeLists.txt` installs the whole `config/` tree:

```cmake
install(DIRECTORY config urdf DESTINATION share/${PROJECT_NAME})
```

So after build, the YAML also appears at:

```text
install/<pkg>/share/<pkg>/config/curobo/<name>.yml
```

Path rewrite after generation:

| Field | Value | Meaning |
|-------|--------|---------|
| `kinematics.asset_root_path` | `../../..` | Relative to **this YAML file**; parent of the folder named `<package_name>` |
| `kinematics.urdf_path` | `<package_name>/urdf/...` | Relative to `asset_root_path` |

Why `../../..` (not `../..`):

- YAML lives at `<pkg_root>/config/curobo/<name>.yml`
- `package://<pkg>/meshes/...` needs `asset_root/<pkg>/meshes/...`
- Therefore `asset_root` must be the **parent** of `<pkg_root>`

That parent is correct in **both** layouts:

```text
source:  .../robots/<robot>/<pkg>/config/curobo/file.yml
         ../../.. → .../robots/<robot>/
         meshes → .../robots/<robot>/<pkg>/meshes ✓

install: .../share/<pkg>/config/curobo/file.yml
         ../../.. → .../share/
         meshes → .../share/<pkg>/meshes ✓
```

Notes:

1. Prefer resolving the YAML via `get_package_share_directory(<pkg>)/config/curobo/...` at runtime so source vs install is transparent.
2. For dual-Pika planning, use `curobo_robot_models/config/marvin_manipulation.yml`
   (geometry from `marvin_manipulation_rt_launch`), not this package.
3. Collision spheres are **embedded** in the YAML; online planning typically does not re-read STL meshes when `load_collision_spheres` is enabled. Relative mesh paths still matter for rebuilds / tooling that reloads the URDF.
4. Do **not** commit machine-local absolute paths (`/home/...`, `/tmp/...`) in
   the tracked YAML. The generate script rewrites them to `../../..` style
   relatives for source/install layout documentation.
5. **Generate / `--visualize-only` mesh resolve:** the script expands the xacro,
   scans `package://<pkg>/...`, and builds a **temporary** asset root whose
   children are `$(ros2 pkg prefix <pkg>)/share/<pkg>` symlinks. Cross-package
   EE meshes (e.g. `pika_gripper_description` next to `marvin_description`)
   therefore resolve through the sourced workspace without a manual
   `--asset-path`. Pass `--asset-path` only to override that default.
6. **`--visualize-only` xacro:** if `--xacro` is omitted, the script prefers
   `kinematics.urdf_path` from the YAML. Arm-only configs here expand
   `marvin.urdf.xacro`; dual-Pika is generated under `curobo_robot_models`.
7. **cuRobo `RobotBuilder.from_config` caveat:** that API joins *relative*
   `urdf_path` / `asset_root_path` with `curobo/content/assets/`. Use
   `--visualize-only` (writes a temp absolute-path copy) or pass absolute paths
   when calling `from_config` yourself. Online planners that load the YAML
   through cuRobo's assets join need the same absolute resolve at load time.
8. Symlink-install is recommended so editing source `config/curobo/` and reading
   from share stay in sync without an extra copy step during iteration.

## Reusing the script in another description package

1. Copy `scripts/generate_curobo_robot_model.sh` into the other package
2. Ensure `install(DIRECTORY config ...)` exists in that package's CMakeLists
3. Run with explicit `--tool-frames` / `--clip-link` as needed
4. Commit the generated `config/curobo/*.yml` on the calibrated geometry branch
5. Optionally copy this doc as `docs/CUROBO_ROBOT_MODEL.md`

## Related

- [cuRobo Build Robot Model](https://nvlabs.github.io/curobo/latest/getting-started/build_robot_model.html)
- Package README section **cuRobo collision config**
