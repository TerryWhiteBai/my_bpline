#!/usr/bin/env python3
"""
把双臂 (Bimanual) robomimic HDF5 里的 delta OSC_POSE action 转成绝对位姿 action。

注意：robosuite 的 Bimanual 机器人原始 action 顺序是 right-first：
    [right_arm_control, right_gripper_control, left_arm_control, left_gripper_control]

本脚本会先把左右臂拆开，再按训练需要的 left-first 顺序输出：
    [left_pos(3), left_rotvec(3), right_pos(3), right_rotvec(3), left_gripper(1), right_gripper(1)]

训练时 RobomimicReplayBSplineImageDataset 会根据 abs_action=True 和
rotation_rep='rotation_6d' 把这 14D 转成 20D dual_arm_ee_rot6d：
    [left_pos(3), left_rot6d(6), right_pos(3), right_rot6d(6), left_gripper(1), right_gripper(1)]

同时会把输出 HDF5 的 env_args.controller_configs.control_delta 改成 false。

用法（在 mimicgen / robomimic 环境里执行）：
    conda activate mimicgen
    python convert_bimanual_delta_to_abs.py -i bimanual_data/demo.hdf5 -o bimanual_data/demo_abs.hdf5
"""

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from robomimic.config import config_factory
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils


def convert_demo(env, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """对单个 demo 的双臂 delta action 序列逐帧转成 left-first 的绝对 action。"""
    T = len(states)
    left_pos = np.zeros((T, 3), dtype=actions.dtype)
    left_ori = np.zeros((T, 3), dtype=actions.dtype)
    right_pos = np.zeros((T, 3), dtype=actions.dtype)
    right_ori = np.zeros((T, 3), dtype=actions.dtype)
    left_gripper = np.zeros((T, 1), dtype=actions.dtype)
    right_gripper = np.zeros((T, 1), dtype=actions.dtype)

    env_config = getattr(env.base_env, "env_configuration", None)
    is_bimanual = env_config == "bimanual"

    if is_bimanual:
        robot = env.env.robots[0]
    else:
        # TwoArmEnv 的 two independent robots 配置：robots[0]=right, robots[1]=left
        right_robot = env.env.robots[0]
        left_robot = env.env.robots[1]

    for i in range(T):
        env.reset_to({"states": states[i]})

        if is_bimanual:
            # Bimanual: 直接传 14D action
            robot.control(actions[i], policy_step=True)
            right_controller = robot.controller["right"]
            left_controller = robot.controller["left"]
        else:
            # 两个独立机器人：把 14D action 拆成两个 7D action
            # 原始顺序 right-first: [right(7), left(7)]
            right_action = actions[i, :7]
            left_action = actions[i, 7:]
            right_robot.control(right_action, policy_step=True)
            left_robot.control(left_action, policy_step=True)
            right_controller = right_robot.controller
            left_controller = left_robot.controller

        right_pos[i] = right_controller.goal_pos
        right_ori[i] = Rotation.from_matrix(right_controller.goal_ori).as_rotvec()
        left_pos[i] = left_controller.goal_pos
        left_ori[i] = Rotation.from_matrix(left_controller.goal_ori).as_rotvec()

        # 原始 action: [right(7), left(7)] -> right_gripper=action[6], left_gripper=action[13]
        right_gripper[i] = actions[i, 6]
        left_gripper[i] = actions[i, 13]

    # 输出严格 left-first
    abs_actions = np.concatenate(
        [left_pos, left_ori, right_pos, right_ori, left_gripper, right_gripper],
        axis=-1,
    )
    return abs_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="bimanual_data/demo.hdf5", help="输入 HDF5")
    parser.add_argument(
        "-o", "--output", default="bimanual_data/demo_abs.hdf5", help="输出 HDF5"
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = config_factory(algo_name="bc")
    ObsUtils.initialize_obs_utils_with_config(config)

    env_meta = FileUtils.get_env_metadata_from_dataset(str(input_path))
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )

    env_config = getattr(env.base_env, "env_configuration", None)
    is_bimanual = env_config == "bimanual"
    if is_bimanual:
        robot = env.env.robots[0]
        print(f"Bimanual arms: {robot.arms}")
    else:
        print(f"Two-robot env config: {env_config}")

    shutil.copy(str(input_path), str(output_path))

    with h5py.File(str(output_path), "r+") as out_f:
        demos = out_f["data"]
        n_demos = len(demos)

        for i in range(n_demos):
            demo = demos[f"demo_{i}"]
            states = demo["states"][:]
            actions = demo["actions"][:]
            if actions.shape[-1] != 14:
                raise ValueError(
                    f"demo_{i} action 维度={actions.shape[-1]}，期望 14（双臂 Bimanual delta）"
                )
            abs_actions = convert_demo(env, states, actions)
            demo["actions"][:] = abs_actions
            print(f"demo_{i}: {actions.shape[0]} steps converted")

        env_args = json.loads(demos.attrs["env_args"])
        env_args["env_kwargs"]["controller_configs"]["control_delta"] = False
        demos.attrs["env_args"] = json.dumps(env_args)

    print(f"Done. Bimanual absolute-action dataset saved to {output_path}")


if __name__ == "__main__":
    main()
