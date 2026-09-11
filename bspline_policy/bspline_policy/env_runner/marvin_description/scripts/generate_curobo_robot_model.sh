#!/usr/bin/env bash
# Generate a cuRobo robot collision YAML from a ROS 2 *_description package.
#
# Designed to be copied or kept inside any xxx_description repo. Defaults are
# inferred from this package (package.xml name, single top-level *.urdf.xacro).
#
# Requires: ROS 2 (xacro), nvidia-curobo, and the description package on
# AMENT_PREFIX_PATH. Prefer: colcon build --symlink-install
#
# Docs: docs/CUROBO_ROBOT_MODEL.md
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PACKAGE_NAME=""
XACRO_REL=""
OUTPUT_REL=""
ASSET_PATH=""
TOOL_FRAMES=()
CLIP_LINKS=()
XACRO_ARGS=(ros2_control:=false)
EXTRA_ARGS=(--compute-metrics --seed 42)
SEED_SET=0
VISUALIZE_ONLY=0
VISUALIZE_CONFIG=""
VIZ_PORT=8080

usage() {
  cat <<'EOF'
Generate cuRobo collision spheres + self-collision ignore matrix for a
ROS 2 description package.

Usage:
  bash scripts/generate_curobo_robot_model.sh [options] [-- <build_robot_model args>]
  bash scripts/generate_curobo_robot_model.sh --visualize-only [CONFIG.yml]

Options:
  --package-name NAME     ROS package name (default: <name> from package.xml)
  --xacro REL_OR_ABS      xacro entry under share/ or absolute path
                          (default: first urdf/*.urdf.xacro under the package)
  --output REL_OR_ABS     output YAML path
                          (default: config/curobo/<name>.yml, strips _description)
  --asset-path DIR        parent of <pkg>/ dirs for package:// mesh resolve
                          (default: temp root with ROS share symlinks for every
                          package:// referenced by the expanded URDF)
  --tool-frames F [F...]  ee / tool frame names (repeatable tokens until next --*)
  --clip-link L A O       clip spheres on link L at axis A offset O (repeatable)
  --xacro-arg KEY:=VAL    extra xacro mapping (repeatable; default ros2_control:=false)
  --visualize             visualize while generating (Viser UI)
  --visualize-only [YML]  visualize an existing YAML (no sphere refit; default: config/curobo/<name>.yml)
  --export-xrdf           also export XRDF (not required for this runtime)
  --no-prune              skip collision-matrix pruning
  --sphere-density N      MorphIt sphere density multiplier
  --viz-port PORT         Viser port (default: 8080)
  --seed N                RNG seed (default: 42)
  --num-collision-samples N
  -h, --help              show this help

Environment:
  Must be able to `import curobo`. Generate mode also needs
  `ros2 pkg prefix <package-name>`.

Examples:
  bash scripts/generate_curobo_robot_model.sh
  bash scripts/generate_curobo_robot_model.sh --visualize
  bash scripts/generate_curobo_robot_model.sh --visualize-only
  bash scripts/generate_curobo_robot_model.sh --visualize-only config/curobo/marvin.yml
EOF
}

read_package_name() {
  local xml="${PKG_ROOT}/package.xml"
  if [[ ! -f "${xml}" ]]; then
    echo "package.xml not found at ${PKG_ROOT}" >&2
    exit 1
  fi
  python3 - "${xml}" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
match = re.search(r"<name>\s*([^<]+?)\s*</name>", text)
if not match:
    raise SystemExit("could not parse <name> from package.xml")
print(match.group(1).strip())
PY
}

default_xacro_rel() {
  local found
  found="$(find -L "${PKG_ROOT}/urdf" -maxdepth 1 -type f -name '*.urdf.xacro' 2>/dev/null | sort | head -n 1 || true)"
  if [[ -z "${found}" ]]; then
    echo "No urdf/*.urdf.xacro found under ${PKG_ROOT}/urdf" >&2
    exit 1
  fi
  echo "urdf/$(basename "${found}")"
}

# Prefer the xacro recorded in an existing cuRobo YAML (urdf_path).
xacro_rel_from_config() {
  local cfg="$1"
  python3 - "${cfg}" "${PACKAGE_NAME}" <<'PY'
from pathlib import Path
import sys

import yaml

cfg = Path(sys.argv[1])
package_name = sys.argv[2]
data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
urdf_path = str((data.get("kinematics") or {}).get("urdf_path") or "").strip()
if not urdf_path:
    raise SystemExit(0)
# Absolute expanded URDF from a previous viz run is not reusable as --xacro.
if Path(urdf_path).is_absolute():
    raise SystemExit(0)
prefix = f"{package_name}/"
if urdf_path.startswith(prefix):
    urdf_path = urdf_path[len(prefix) :]
print(urdf_path)
PY
}

# cuRobo resolves package://NAME/... as <asset_root>/NAME/...
# Build a temp root whose children are ROS share dirs via ament.
build_ros_asset_root() {
  local urdf="$1"
  local dest="$2"
  local pkg prefix share
  local -a pkgs=()

  mkdir -p "${dest}"

  mapfile -t pkgs < <(
    python3 - "${urdf}" "${PACKAGE_NAME}" <<'PY'
import re
import sys
from pathlib import Path

urdf = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
primary = sys.argv[2]
pkgs = set(re.findall(r"package://([^/\"'\s>]+)", urdf))
pkgs.add(primary)
for name in sorted(pkgs):
    print(name)
PY
  )

  if [[ ${#pkgs[@]} -eq 0 ]]; then
    echo "No package:// references found in ${urdf}" >&2
    exit 1
  fi

  echo "Resolving mesh packages via ROS (ament):" >&2
  for pkg in "${pkgs[@]}"; do
    if ! prefix="$(ros2 pkg prefix "${pkg}" 2>/dev/null)"; then
      echo "  package://${pkg} → not on AMENT_PREFIX_PATH (source install/setup.bash)" >&2
      exit 1
    fi
    share="${prefix}/share/${pkg}"
    if [[ ! -d "${share}" ]]; then
      echo "  package://${pkg} → missing share dir: ${share}" >&2
      exit 1
    fi
    ln -sfn "${share}" "${dest}/${pkg}"
    echo "  package://${pkg} → ${share}" >&2
  done
  printf '%s\n' "${dest}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --package-name)
      PACKAGE_NAME="$2"; shift 2 ;;
    --xacro)
      XACRO_REL="$2"; shift 2 ;;
    --output)
      OUTPUT_REL="$2"; shift 2 ;;
    --asset-path)
      ASSET_PATH="$2"; shift 2 ;;
    --tool-frames)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        TOOL_FRAMES+=("$1"); shift
      done
      ;;
    --clip-link)
      CLIP_LINKS+=("$2" "$3" "$4"); shift 4 ;;
    --xacro-arg)
      XACRO_ARGS+=("$2"); shift 2 ;;
    --visualize|--export-xrdf|--no-prune|--compute-metrics)
      EXTRA_ARGS+=("$1"); shift ;;
    --visualize-only)
      VISUALIZE_ONLY=1
      shift
      if [[ $# -gt 0 && "$1" != --* ]]; then
        VISUALIZE_CONFIG="$1"
        shift
      fi
      ;;
    --sphere-density|--num-collision-samples)
      EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
    --viz-port)
      VIZ_PORT="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --seed)
      EXTRA_ARGS+=("$1" "$2"); SEED_SET=1; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1 (see --help)" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${PACKAGE_NAME}" ]]; then
  PACKAGE_NAME="$(read_package_name)"
fi
if [[ -z "${OUTPUT_REL}" ]]; then
  _base="${PACKAGE_NAME%_description}"
  OUTPUT_REL="config/curobo/${_base}.yml"
fi
if [[ "${OUTPUT_REL}" = /* ]]; then
  OUT_YML="${OUTPUT_REL}"
else
  OUT_YML="${PKG_ROOT}/${OUTPUT_REL}"
fi

rewrite_portable_paths() {
  local yml="$1"
  local package_name="$2"
  local xacro_rel="$3"
  python3 - "${yml}" "${package_name}" "${xacro_rel}" <<'PY'
from pathlib import Path
import sys

import yaml

out = Path(sys.argv[1])
package_name = sys.argv[2]
xacro_rel = sys.argv[3].lstrip("./")
data = yaml.safe_load(out.read_text(encoding="utf-8"))
kinematics = data.setdefault("kinematics", {})
kinematics["asset_root_path"] = "../../.."
if xacro_rel.startswith(f"{package_name}/"):
    kinematics["urdf_path"] = xacro_rel
else:
    kinematics["urdf_path"] = f"{package_name}/{xacro_rel}"
data.setdefault("description_package", {})
data["description_package"]["name"] = package_name
data["description_package"]["config_relpath"] = "config/curobo"
out.write_text(
    yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
    encoding="utf-8",
)
print(f"Rewrote portable paths in {out}")
PY
}

if [[ "${VISUALIZE_ONLY}" -eq 1 ]]; then
  if ! python3 -c "import curobo" >/dev/null 2>&1; then
    echo "curobo not importable. Activate the Pixi / cuRobo environment first." >&2
    exit 1
  fi
  if [[ -n "${VISUALIZE_CONFIG}" ]]; then
    if [[ "${VISUALIZE_CONFIG}" = /* ]]; then
      CFG="${VISUALIZE_CONFIG}"
    else
      CFG="${PKG_ROOT}/${VISUALIZE_CONFIG}"
    fi
  else
    CFG="${OUT_YML}"
  fi
  if [[ ! -f "${CFG}" ]]; then
    echo "Config not found: ${CFG}" >&2
    echo "Generate first, or pass an existing YAML path to --visualize-only." >&2
    exit 1
  fi
  if [[ -z "${XACRO_REL}" ]]; then
    XACRO_REL="$(xacro_rel_from_config "${CFG}" || true)"
  fi
  if [[ -z "${XACRO_REL}" ]]; then
    XACRO_REL="$(default_xacro_rel)"
  fi

  # RobotBuilder.from_config always does join_path(get_assets_path(), urdf_path).
  # Relative paths therefore resolve under curobo/content/assets/ and break.
  # Absolute paths are passed through; use a temp YAML + expanded URDF.
  if ! command -v xacro >/dev/null 2>&1; then
    echo "xacro not found (needed to expand URDF for visualization)." >&2
    exit 1
  fi
  if ! command -v ros2 >/dev/null 2>&1; then
    echo "ros2 not found. Activate a ROS 2 + cuRobo environment (e.g. Pixi shell)." >&2
    exit 1
  fi
  if ! ros2 pkg prefix "${PACKAGE_NAME}" >/dev/null 2>&1; then
    echo "${PACKAGE_NAME} is not on AMENT_PREFIX_PATH (needed to expand xacro)." >&2
    exit 1
  fi
  SHARE="$(ros2 pkg prefix "${PACKAGE_NAME}")/share/${PACKAGE_NAME}"
  if [[ "${XACRO_REL}" = /* ]]; then
    XACRO_IN="${XACRO_REL}"
  else
    XACRO_IN="${SHARE}/${XACRO_REL}"
  fi
  if [[ ! -f "${XACRO_IN}" ]]; then
    echo "xacro not found: ${XACRO_IN}" >&2
    exit 1
  fi

  TMP_DIR="${TMPDIR:-/tmp}/${PACKAGE_NAME}_curobo_viz_$$"
  TMP_URDF="${TMP_DIR}/robot.urdf"
  TMP_YML="${TMP_DIR}/robot_abs.yml"
  mkdir -p "${TMP_DIR}"
  trap 'rm -rf "${TMP_DIR}"' EXIT

  echo "Expanding ${XACRO_IN} for visualization"
  xacro "${XACRO_IN}" "${XACRO_ARGS[@]}" > "${TMP_URDF}"

  if [[ -z "${ASSET_PATH}" ]]; then
    ASSET_PATH="$(build_ros_asset_root "${TMP_URDF}" "${TMP_DIR}/assets")"
  fi

  python3 - "${CFG}" "${TMP_YML}" "${ASSET_PATH}" "${TMP_URDF}" <<'PY'
from pathlib import Path
import sys

import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
asset_root = str(Path(sys.argv[3]).resolve())
urdf_path = str(Path(sys.argv[4]).resolve())
data = yaml.safe_load(src.read_text(encoding="utf-8"))
kinematics = data.setdefault("kinematics", {})
kinematics["asset_root_path"] = asset_root
kinematics["urdf_path"] = urdf_path
dst.write_text(
    yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
    encoding="utf-8",
)
print(f"Prepared absolute-path temp config: {dst}")
PY

  echo "Visualizing existing config: ${CFG}"
  echo "Open http://localhost:${VIZ_PORT}  (Ctrl+C to stop)"
  # Do not overwrite the portable source YAML.
  python3 -m curobo.examples.getting_started.build_robot_model \
    --edit-config "${TMP_YML}" \
    --output "${TMP_DIR}/robot_out.yml" \
    --visualize \
    --viz-port "${VIZ_PORT}"
  exit 0
fi

if [[ -z "${XACRO_REL}" ]]; then
  XACRO_REL="$(default_xacro_rel)"
fi

# Marvin defaults when the contributor did not pass frames/clips.
if [[ ${#TOOL_FRAMES[@]} -eq 0 && "${PACKAGE_NAME}" == "marvin_description" ]]; then
  TOOL_FRAMES=(flange_L flange_R)
fi
if [[ ${#CLIP_LINKS[@]} -eq 0 && "${PACKAGE_NAME}" == "marvin_description" ]]; then
  CLIP_LINKS=(base_link z 0.0)
fi

OUT_DIR="$(dirname "${OUT_YML}")"

TMP_DIR="${TMPDIR:-/tmp}/${PACKAGE_NAME}_curobo_$$"
TMP_URDF="${TMP_DIR}/robot.urdf"

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ros2 not found. Activate a ROS 2 + cuRobo environment (e.g. Pixi shell)." >&2
  exit 1
fi
if ! command -v xacro >/dev/null 2>&1; then
  echo "xacro not found. Build/source the workspace first." >&2
  exit 1
fi
if ! python3 -c "import curobo" >/dev/null 2>&1; then
  echo "curobo not importable. Install/enable nvidia-curobo in this environment." >&2
  exit 1
fi
if ! ros2 pkg prefix "${PACKAGE_NAME}" >/dev/null 2>&1; then
  echo "${PACKAGE_NAME} is not on AMENT_PREFIX_PATH." >&2
  echo "Build and source it (prefer --symlink-install)." >&2
  exit 1
fi

SHARE="$(ros2 pkg prefix "${PACKAGE_NAME}")/share/${PACKAGE_NAME}"
if [[ "${XACRO_REL}" = /* ]]; then
  XACRO_IN="${XACRO_REL}"
else
  XACRO_IN="${SHARE}/${XACRO_REL}"
fi
if [[ ! -f "${XACRO_IN}" ]]; then
  echo "xacro not found: ${XACRO_IN}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}" "${TMP_DIR}"
trap 'rm -rf "${TMP_DIR}"' EXIT

BUILD_ARGS=()
if [[ ${#TOOL_FRAMES[@]} -gt 0 ]]; then
  BUILD_ARGS+=(--tool-frames "${TOOL_FRAMES[@]}")
fi
if [[ ${#CLIP_LINKS[@]} -gt 0 ]]; then
  i=0
  while [[ $i -lt ${#CLIP_LINKS[@]} ]]; do
    BUILD_ARGS+=(--clip-link "${CLIP_LINKS[$i]}" "${CLIP_LINKS[$((i + 1))]}" "${CLIP_LINKS[$((i + 2))]}")
    i=$((i + 3))
  done
fi

xacro "${XACRO_IN}" "${XACRO_ARGS[@]}" > "${TMP_URDF}"
if command -v check_urdf >/dev/null 2>&1; then
  check_urdf "${TMP_URDF}" >/dev/null
else
  echo "warning: check_urdf not found; skipping URDF check" >&2
fi

if [[ -z "${ASSET_PATH}" ]]; then
  ASSET_PATH="$(build_ros_asset_root "${TMP_URDF}" "${TMP_DIR}/assets")"
fi

echo "Package:  ${PACKAGE_NAME}"
echo "Xacro:    ${XACRO_IN}"
echo "Asset:    ${ASSET_PATH}"
echo "Output:   ${OUT_YML}"

python3 -m curobo.examples.getting_started.build_robot_model \
  --urdf "${TMP_URDF}" \
  --asset-path "${ASSET_PATH}" \
  --output "${OUT_YML}" \
  "${BUILD_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

rewrite_portable_paths "${OUT_YML}" "${PACKAGE_NAME}" "${XACRO_REL}"

echo "Wrote ${OUT_YML}"
echo "Regenerate after changing meshes, mount/stand YAML, or link collision geometry."
echo "Visualize without refitting: bash scripts/generate_curobo_robot_model.sh --visualize-only"
echo "See docs/CUROBO_ROBOT_MODEL.md for install-space path notes."
