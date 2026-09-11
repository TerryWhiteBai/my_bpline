"""
在 robomimic / robosuite 仿真中验证输出 B-spline 参数的模型。

主要复用 `PolicyLocalBSpline` 里的 B-spline 推理、时间对齐 (`_align_new_plan`)、
以及 B-spline 参数到绝对路径点的转换逻辑；再把解码后的绝对位姿转换成
robomimic 可接受的 action vector（OSC_POSE absolute）。

使用示例：

    from bspline_policy.env_runner.robomimic_bspline_runner import RobomimicBSplineRunner

    runner = RobomimicBSplineRunner(
        output_dir="outputs/bspline_square_eval",
        ckpt_path="path/to/checkpoint.ckpt",
        dataset_path="square_data/demo_abs.hdf5",
        n_train=10,
        n_test=10,
        max_steps=800,
        fps=20,
    )
    log_data = runner.run()  # runner 内部持有 policy，忽略外部传入的 policy

运行环境：
    conda activate mimicgen
    python run_bspline_sim.py
"""

import sys
import time
import copy
import math
import collections
import pathlib
import uuid
from pathlib import Path
try:
    import mimicgen
    import mimicgen_envs
except ImportError:
    mimicgen = None
    mimicgen_envs = None

# 让本文件在未安装 editable 包时也能直接 import 到 bspline_policy / diffusion_policy 内部源码
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "diffusion_policy" / "diffusion_policy") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "diffusion_policy" / "diffusion_policy"))
if str(_REPO_ROOT / "bspline_policy" / "bspline_policy") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "bspline_policy" / "bspline_policy"))

import numpy as np
import torch
import h5py
import tqdm
import wandb
import wandb.sdk.data_types.video as wv
from scipy.spatial.transform import Rotation

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from policy_local_bspline import PolicyLocalBSpline


def _rot6d_to_rotvec(rot6d):
    """把 6D rotation representation 转成 axis-angle (rotvec)。"""
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(6)
    a1 = rot6d[:3]
    a2 = rot6d[3:]
    b1 = a1 / max(np.linalg.norm(a1), 1e-12)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / max(np.linalg.norm(b2), 1e-12)
    b3 = np.cross(b1, b2)
    mat = np.stack((b1, b2, b3), axis=-2)
    return Rotation.from_matrix(mat).as_rotvec()


class RobomimicBSplineRunner(BaseImageRunner):
    """
    用 B-spline 输出模型在 robomimic 仿真环境里做 rollout。

    注意：
      - 环境默认使用绝对 OSC_POSE（control_delta=False），因为 B-spline 输出的是绝对位姿。
      - 每次 rollout 都会新建一个 `PolicyLocalBSpline` 实例，reset 后复用其 B-spline
        推理、chunk 对齐（`_align_new_plan`）以及 `_sample_action` 逻辑。
      - 为了和 `PolicyLocalBSpline` 内部自己维护 obs history 一致，外面的
        `MultiStepWrapper` 的 `n_obs_steps=1`；obs 的时序堆叠由 policy 自己完成。
    """

    def __init__(
        self,
        output_dir,
        ckpt_path,
        dataset_path,
        n_train=10,
        n_train_vis=3,
        train_start_idx=0,
        n_test=22,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=800,
        fps=20,
        crf=22,
        render_obs_key=None,
        action_format=None,
        rotation_output="axis_angle",
        device="cuda",
        # B-spline policy 相关参数
        speed_up_times=1.0,
        predict_before_end=0.06,
        origin_time_scale=10.0,
        use_action_derivatives=False,
        disable_time_align=False,
        time_align_error_threshold=0.1,
        time_align_larger_t=1.0,
        restart_on_time_align_error=False,
        consider_gripper_during_align=False,
        gripper_slowdown_enabled=False,
        gripper_slowdown_threshold=0.08,
        gripper_slowdown_steps=7,
        normalize_knots_zero=True,
        obs_stride=1,
        tqdm_interval_sec=5.0,
    ):
        super().__init__(output_dir)

        self.ckpt_path = ckpt_path
        self.dataset_path = dataset_path
        self.max_steps = max_steps
        self.fps = fps
        self.crf = crf
        self.render_obs_key = render_obs_key
        self.tqdm_interval_sec = tqdm_interval_sec
        self.device = device

        self.n_train = n_train
        self.n_train_vis = n_train_vis
        self.train_start_idx = train_start_idx
        self.n_test = n_test
        self.n_test_vis = n_test_vis
        self.test_start_seed = test_start_seed

        # 加载 B-spline 策略（会启动后台推理线程）
        self.policy = PolicyLocalBSpline(
            ckpt_path=ckpt_path,
            device=device,
            speed_up_times=speed_up_times,
            predict_before_end=predict_before_end,
            origin_time_scale=origin_time_scale,
            use_action_derivatives=use_action_derivatives,
            disable_time_align=disable_time_align,
            time_align_error_threshold=time_align_error_threshold,
            time_align_larger_t=time_align_larger_t,
            restart_on_time_align_error=restart_on_time_align_error,
            consider_gripper_during_align=consider_gripper_during_align,
            gripper_slowdown_enabled=gripper_slowdown_enabled,
            gripper_slowdown_threshold=gripper_slowdown_threshold,
            gripper_slowdown_steps=gripper_slowdown_steps,
            normalize_knots_zero=normalize_knots_zero,
            obs_stride=obs_stride,
        )

        # 根据模型 cfg 确定 observation / action 元信息
        self.shape_meta = copy.deepcopy(self.policy.model.cfg.shape_meta)
        # 若用户未指定渲染相机，根据 shape_meta 自动选一个存在的 rgb key
        if self.render_obs_key is None:
            rgb_keys = [
                k for k, v in self.shape_meta.get("obs", {}).items()
                if v.get("type") == "rgb"
            ]
            self.render_obs_key = rgb_keys[0] if rgb_keys else "agentview_image"
        self.n_obs_steps = self.policy.n_obs_steps
        self.action_meta = self.policy.action_meta
        self.action_format = action_format or self.action_meta.get("action_format")
        self.rotation_output = rotation_output

        # 如果用户没显式指定 action_format，根据 action_dim 给一个合理的默认（对应 square / dual arm 等）
        if self.action_format is None or self.action_format == "bspline":
            action_dim = int(self.shape_meta["action"]["shape"][0])
            if action_dim == 10:
                self.action_format = "single_yam_rot6d"
            elif action_dim == 20:
                self.action_format = "dual_arm_ee_rot6d"
            elif action_dim == 14:
                self.action_format = "dual_arm_ee_pose6d"
            elif action_dim == 23:
                self.action_format = "real_bimanual_base_rot6d"
            else:
                raise ValueError(f"无法自动推断 action_format，action_dim={action_dim}")

        # 覆盖 action_meta，让 decode_action_vector 按我们想要的格式解码
        self.action_meta["action_format"] = self.action_format
        self.action_meta["rotation_output"] = self.rotation_output
        self.policy.action_meta = self.action_meta
        self.policy.action_format = self.action_format

        # 计算输出给 robomimic 的 action 维度
        self.env_action_dim = self._infer_env_action_dim(self.action_format)
        # 修正 wrapper action_space 用的 shape_meta（不影响 policy，只影响 env wrapper）
        self.env_shape_meta = copy.deepcopy(self.shape_meta)
        self.env_shape_meta["action"]["shape"] = [self.env_action_dim]

        # 加载环境元信息，强制绝对控制
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
        # 图像 runner 不需要 object state 作为 low-dim obs
        env_meta["env_kwargs"]["use_object_obs"] = False
        self.env_meta = env_meta

        self.steps_per_render = max(20 // fps, 1)

    # ------------------------------------------------------------------ #
    # 公共接口
    # ------------------------------------------------------------------ #
    def run(self, policy=None):
        """执行仿真验证。外部传入的 policy 会被忽略，runner 使用构造时加载的 B-spline 策略。"""
        del policy

        output_dir = pathlib.Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        media_dir = output_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)

        # 收集所有初始条件
        inits = []
        with h5py.File(self.dataset_path, "r") as f:
            for i in range(self.n_train):
                train_idx = self.train_start_idx + i
                init_state = f[f"data/demo_{train_idx}/states"][0]
                init_model = f[f"data/demo_{train_idx}"].attrs["model_file"]
                inits.append(("train/", train_idx, init_state, init_model, i < self.n_train_vis))
        for i in range(self.n_test):
            seed = self.test_start_seed + i
            inits.append(("test/", seed, None, None, i < self.n_test_vis))

        max_rewards = collections.defaultdict(list)
        log_data = {}

        for prefix, seed, init_state, init_model, enable_render in inits:
            video_path = None
            if enable_render:
                video_path = media_dir / (uuid.uuid4().hex + ".mp4")

            reward = self._run_one_rollout(
                prefix=prefix,
                seed=seed,
                init_state=init_state,
                init_model=init_model,
                video_path=video_path,
            )
            max_reward = float(np.max(reward)) if len(reward) > 0 else 0.0
            max_rewards[prefix].append(max_reward)
            log_data[f"{prefix}sim_max_reward_{seed}"] = max_reward

            if video_path is not None and video_path.exists():
                log_data[f"{prefix}sim_video_{seed}"] = wandb.Video(str(video_path))

        for prefix, vals in max_rewards.items():
            log_data[f"{prefix}mean_score"] = float(np.mean(vals))

        self.policy.wait_for_pending_inference(timeout=10.0)
        self.policy.print_inference_summary()
        return log_data

    # ------------------------------------------------------------------ #
    # 单次 rollout
    # ------------------------------------------------------------------ #
    def _run_one_rollout(self, prefix, seed, init_state, init_model, video_path):
        env = self._create_env()
        video_wrapper = env.env  # MultiStepWrapper -> VideoRecordingWrapper
        image_wrapper = video_wrapper.env  # VideoRecordingWrapper -> RobomimicImageWrapper

        # 设置初始状态 / seed
        if init_state is not None:
            image_wrapper.init_state = init_state
            image_wrapper.init_model = init_model
            video_wrapper.file_path = None
        else:
            image_wrapper.init_state = None
            image_wrapper.init_model = None
            env.seed(seed)

        if video_path is not None:
            video_wrapper.file_path = str(video_path)

        obs = env.reset()
        self.policy.reset()

        pbar = tqdm.tqdm(
            total=self.max_steps,
            desc=f"Eval {prefix}{seed}",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        )

        done = False
        step_count = 0
        while not done and step_count < self.max_steps:
            obs_for_policy = self._obs_to_policy(obs)

            # 喂入新观测并请求/复用 B-spline plan
            action_dict = self.policy.step(obs_for_policy)
            if action_dict is None:
                # 第一次 plan 或新 chunk 还没准备好，稍等后 poll
                action_dict = self._wait_for_action(timeout=0.2)
            if action_dict is None:
                # 仍然无可用动作，保持当前末端位姿，避免 delta/绝对控制下零向量导致乱动
                action_vec = self._hold_action(obs)
            else:
                action_vec = self._convert_action_dict(action_dict)

            # 绝对位姿的 pos/rotvec 不应 clip 到 [-1,1]，否则会破坏目标朝向；
            # 只对 gripper 维度做限幅。
            if self.env_action_dim == 14:
                # 双臂：[right(7), left(7)]，两个 gripper 分别限幅
                action_vec = np.concatenate([
                    action_vec[:6],
                    np.clip(action_vec[6:7], -1.0, 1.0),
                    action_vec[7:13],
                    np.clip(action_vec[13:14], -1.0, 1.0),
                ])
            else:
                action_vec = np.concatenate([
                    action_vec[:6],
                    np.clip(action_vec[6:7], -1.0, 1.0),
                ])
            # MultiStepWrapper 期望 (n_action_steps, action_dim)，这里 n_action_steps=1
            action_batch = action_vec[np.newaxis, :]

            obs, reward, done, info = env.step(action_batch)
            done = bool(np.all(done))
            step_count += 1
            pbar.update(1)

        pbar.close()

        if hasattr(env, "call"):
            rewards = env.call("get_attr", "reward")
        else:
            rewards = [env.reward]
        return rewards

    def _wait_for_action(self, timeout=0.2):
        """等待后台推理线程给出新的 B-spline 动作；超时返回 None。"""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            action_dict = self.policy.poll_action()
            if action_dict is not None:
                return action_dict
            if self.policy.waiting_for_first_plan():
                time.sleep(0.001)
            else:
                # predictor 已存在但当前时刻可能已超出 plan 范围，再 poll 一次
                action_dict = self.policy.poll_action()
                if action_dict is not None:
                    return action_dict
                time.sleep(0.001)
        return None

    # ------------------------------------------------------------------ #
    # 环境构造
    # ------------------------------------------------------------------ #
    def _create_env(self):
        modality_mapping = collections.defaultdict(list)
        for key, attr in self.env_shape_meta["obs"].items():
            modality_mapping[attr.get("type", "low_dim")].append(key)
        ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

        # B-spline 策略始终需要图像观测作为输入，图像渲染不能和是否录制视频绑定，
        # 否则非可视化 rollout（enable_render=False）会拿不到 agentview_image 等键。
        robomimic_env = EnvUtils.create_env_from_metadata(
            env_meta=self.env_meta,
            render=False,
            render_offscreen=True,
            use_image_obs=True,
        )
        # 避免 hard reset 造成大量内存/渲染开销
        robomimic_env.env.hard_reset = False

        return MultiStepWrapper(
            VideoRecordingWrapper(
                RobomimicImageWrapper(
                    env=robomimic_env,
                    shape_meta=self.env_shape_meta,
                    init_state=None,
                    render_obs_key=self.render_obs_key,
                ),
                video_recoder=VideoRecorder.create_h264(
                    fps=self.fps,
                    codec="h264",
                    input_pix_fmt="rgb24",
                    crf=self.crf,
                    thread_type="FRAME",
                    thread_count=1,
                ),
                file_path=None,
                steps_per_render=self.steps_per_render,
            ),
            n_obs_steps=1,  # policy 自己维护 history
            n_action_steps=1,
            max_episode_steps=self.max_steps,
        )

    # ------------------------------------------------------------------ #
    # 观测/动作格式转换
    # ------------------------------------------------------------------ #
    def _obs_to_policy(self, obs):
        """
        MultiStepWrapper 出来的 obs 带一个 batch 维度 (1, ...)，
        且图像是 float32 CHW；policy 内部期望 uint8 HWC 单帧。
        """
        out = {}
        for key, value in obs.items():
            if not isinstance(value, np.ndarray):
                out[key] = value
                continue
            # 去掉 wrapper 的 batch 维
            v = value[0] if value.shape[0] == 1 else value
            meta = self.shape_meta["obs"].get(key, {})
            if meta.get("type") == "rgb":
                if v.dtype != np.uint8:
                    # CHW float [0,1] -> HWC uint8
                    v = np.transpose(v, (1, 2, 0))
                    v = (v * 255.0).clip(0, 255).astype(np.uint8)
            out[key] = v
        return out

    def _infer_env_action_dim(self, action_format):
        if action_format in (
            "single_yam_rot6d",
            "single_left_arm_rot6d",
            "single_right_arm_rot6d",
        ):
            return 7
        if action_format in ("dual_arm_ee_rot6d", "dual_arm_ee_rot6d_next", "dual_arm_ee_pose6d"):
            return 14
        if action_format == "real_bimanual_base_rot6d":
            return 17
        raise ValueError(f"不支持的 action_format={action_format!r}")

    def _convert_action_dict(self, action_dict):
        """
        把 `decode_action_vector` 返回的 dict 转成 robomimic 可执行的 flat action。
        输出是 OSC_POSE absolute：[pos(3), rotvec(3), gripper(1)]，
        双臂则是 [left(7), right(7)]。
        """
        fmt = self.action_format

        def _single_arm(arm_dict):
            pos = np.asarray(arm_dict["arm_pos"], dtype=np.float64)
            quat = np.asarray(arm_dict["arm_quat"], dtype=np.float64)  # xyzw
            gripper = np.asarray(arm_dict["gripper_pos"], dtype=np.float64).reshape(-1)
            rotvec = Rotation.from_quat(quat).as_rotvec()
            return np.concatenate([pos, rotvec, gripper[:1]])

        if fmt == "single_yam_rot6d":
            return _single_arm(action_dict).astype(np.float32)

        if fmt == "single_left_arm_rot6d":
            return _single_arm(action_dict["arm_left"]).astype(np.float32)

        if fmt == "single_right_arm_rot6d":
            return _single_arm(action_dict["arm_right"]).astype(np.float32)

        if fmt in ("dual_arm_ee_rot6d", "dual_arm_ee_rot6d_next"):
            left = self._dual_arm_ee_to_7d(action_dict["left_arm"])
            right = self._dual_arm_ee_to_7d(action_dict["right_arm"])
            # robosuite 双臂环境（single-arm-opposed / bimanual）期望 right-first
            return np.concatenate([right, left]).astype(np.float32)

        if fmt == "dual_arm_ee_pose6d":
            left = self._dual_arm_pose6d_to_7d(action_dict["left_arm"])
            right = self._dual_arm_pose6d_to_7d(action_dict["right_arm"])
            return np.concatenate([right, left]).astype(np.float32)

        if fmt == "real_bimanual_base_rot6d":
            base = np.asarray(action_dict["base_velocity"], dtype=np.float64).reshape(-1)[:3]
            left = _single_arm(action_dict["arm_left"])
            right = _single_arm(action_dict["arm_right"])
            return np.concatenate([base, left, right]).astype(np.float32)

        raise NotImplementedError(f"action_format={fmt!r} 的仿真转换未实现")

    def _dual_arm_ee_to_7d(self, arm_dict):
        """dual_arm_ee_rot6d: pose_6d = [pos(3), rot(3)]，rot 由 rotation_output 决定。"""
        pose6d = np.asarray(arm_dict["pose_6d"], dtype=np.float64)
        pos = pose6d[:3]
        rot = pose6d[3:]
        rot_output = self.rotation_output
        if rot_output == "axis_angle":
            rotvec = rot
        elif rot_output == "euler_xyz":
            rotvec = Rotation.from_euler("xyz", rot).as_rotvec()
        else:
            raise ValueError(f"不支持的 rotation_output={rot_output!r}")
        gripper = np.asarray([arm_dict["gripper_pos"]], dtype=np.float64)
        return np.concatenate([pos, rotvec, gripper])

    def _hold_action(self, obs):
        """当 B-spline plan 尚未生成或已耗尽时，保持当前末端位姿。"""
        if self.env_action_dim == 14:
            # 双臂保持姿态：[right(7), left(7)]；robot0=right, robot1=left
            def _arm_hold(pos_key, quat_key):
                pos = np.asarray(obs[pos_key][0], dtype=np.float64)
                quat = np.asarray(obs[quat_key][0], dtype=np.float64)
                rotvec = Rotation.from_quat(quat).as_rotvec()
                gripper = np.zeros(1, dtype=np.float64)
                return np.concatenate([pos, rotvec, gripper])

            if "robot0_eef_pos" in obs and "robot1_eef_pos" in obs:
                right = _arm_hold("robot0_eef_pos", "robot0_eef_quat")
                left = _arm_hold("robot1_eef_pos", "robot1_eef_quat")
                return np.concatenate([right, left]).astype(np.float32)

        if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
            pos = np.asarray(obs["robot0_eef_pos"][0], dtype=np.float64)
            quat = np.asarray(obs["robot0_eef_quat"][0], dtype=np.float64)  # xyzw
            rotvec = Rotation.from_quat(quat).as_rotvec()
            gripper = np.zeros(1, dtype=np.float64)
            return np.concatenate([pos, rotvec, gripper]).astype(np.float32)
        return np.zeros(self.env_action_dim, dtype=np.float32)

    def _dual_arm_pose6d_to_7d(self, arm_dict):
        """dual_arm_ee_pose6d: pose_6d = [pos(3), euler_xyz(3)]。"""
        pose6d = np.asarray(arm_dict["pose_6d"], dtype=np.float64)
        pos = pose6d[:3]
        rotvec = Rotation.from_euler("xyz", pose6d[3:]).as_rotvec()
        gripper = np.asarray([arm_dict["gripper_pos"]], dtype=np.float64)
        return np.concatenate([pos, rotvec, gripper])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n_train", type=int, default=10)
    parser.add_argument("--n_test", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="cuda")

    # B-spline 控制与时间对齐参数（最常用的几个）
    parser.add_argument("--speed_up_times", type=float, default=1.0,
                        help="B-spline 执行倍速（相对真实时间）")
    parser.add_argument("--predict_before_end", type=float, default=0.06,
                        help="在当前 plan 结束前多久开始预测下一个 plan")
    parser.add_argument("--time_align_error_threshold", type=float, default=0.1,
                        help="time-align 误差阈值，超过会打印警告")
    parser.add_argument("--time_align_larger_t", type=float, default=1.0,
                        help="time-align 搜索范围比例（1.0 表示全区间搜索）")
    parser.add_argument("--restart_on_time_align_error", action="store_true",
                        help="对齐误差过大时直接从新 plan 的 t=0 开始")
    args = parser.parse_args()

    runner = RobomimicBSplineRunner(
        output_dir=args.output_dir,
        ckpt_path=args.ckpt,
        dataset_path=args.dataset,
        n_train=args.n_train,
        n_test=args.n_test,
        max_steps=args.max_steps,
        fps=args.fps,
        device=args.device,
        speed_up_times=args.speed_up_times,
        predict_before_end=args.predict_before_end,
        time_align_error_threshold=args.time_align_error_threshold,
        time_align_larger_t=args.time_align_larger_t,
        restart_on_time_align_error=args.restart_on_time_align_error,
    )
    log_data = runner.run()
    print(log_data)
