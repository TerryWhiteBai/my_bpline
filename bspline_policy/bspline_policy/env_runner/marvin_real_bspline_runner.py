"""
在真机 Marvin 双臂平台上运行 B-spline 策略的 env_runner。

硬件抽象：
  - 两条 Marvin M6 手臂，通过 ROS 2 的 TaskSpaceKinematicPositionController 接收
    Cartesian pose reference（/execution/{left,right}_arm/pose_reference）。
  - 两个 Pika 夹爪，通过 ForwardCommandController 接收 joint reference
    （/execution/{left,right}_gripper/joint_reference）。
  - 两个腕部 RealSense D405（/left_pika_d405、/right_pika_d405）和一个头部
    RealSense D435（/head_d435）提供 RGB 图像。
  - 末端位姿通过 TF（left_pika_gripper_tcp / right_pika_gripper_tcp）获取。

本文件假设运行在已 source ROS 2 workspace 的环境中（rclpy / tf2_ros / moveit_msgs
可用）。所有 ROS 相关的 import 都集中在 MarvinRealEnv 内部，避免在无 ROS 环境 import
本模块时直接失败。
"""
from __future__ import annotations

import copy
import pathlib
import threading
import time
import uuid
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

# 让本文件在未安装 editable 包时也能直接 import 到 bspline_policy / diffusion_policy 内部源码
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "diffusion_policy" / "diffusion_policy") not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_REPO_ROOT / "diffusion_policy" / "diffusion_policy"))
if str(_REPO_ROOT / "bspline_policy" / "bspline_policy") not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_REPO_ROOT / "bspline_policy" / "bspline_policy"))

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.real_world.video_recorder import VideoRecorder
from policy_local_bspline import PolicyLocalBSpline


# --------------------------------------------------------------------------- #
# 默认真机配置（与 marvin_description / execution_manager.yaml 一致）
# --------------------------------------------------------------------------- #
MARVIN_DEFAULT_IMAGE_TOPICS = {
    "head_image": "/head_d435/color/image_raw",
    "left_wrist_image": "/left_pika_d405/color/image_raw",
    "right_wrist_image": "/right_pika_d405/color/image_raw",
}

MARVIN_LEFT_ARM = {
    "side": "left",
    "pos_key": "arm_pos_l",
    "quat_key": "arm_quat_l",
    "gripper_key": "gripper_pos_l",
    "base_frame": "Base_L",
    "tcp_frame": "left_pika_gripper_tcp",
    "pose_topic": "/execution/left_arm/pose_reference",
    "gripper_topic": "/execution/left_gripper/joint_reference",
    "gripper_joint_name": "left_gripper_left_joint",
}

MARVIN_RIGHT_ARM = {
    "side": "right",
    "pos_key": "arm_pos_r",
    "quat_key": "arm_quat_r",
    "gripper_key": "gripper_pos_r",
    "base_frame": "Base_R",
    "tcp_frame": "right_pika_gripper_tcp",
    "pose_topic": "/execution/right_arm/pose_reference",
    "gripper_topic": "/execution/right_gripper/joint_reference",
    "gripper_joint_name": "right_gripper_left_joint",
}

GRIPPER_LIMITS = (0.0, 0.045)  # m, from joint_limits.yaml


# --------------------------------------------------------------------------- #
# ROS 2 真机环境封装
# --------------------------------------------------------------------------- #
class MarvinRealEnv:
    """
    与 Marvin 双臂真机交互的最小 env 封装。

    主要接口与 gym-like env 保持一致：
      - reset()：清空内部状态，等待观测就绪。
      - get_obs()：返回 policy 期望的观测 dict（图像 uint8 HWC + low_dim）。
      - step(action_dict)：发送双臂 pose reference 与 gripper joint reference。
      - render()：返回一帧用于视频录制的 RGB 图像。
      - close()：关闭 ROS 2 节点与相机订阅。
    """

    def __init__(
        self,
        *,
        node_name: str = "marvin_real_env",
        control_hz: float = 20.0,
        joint_state_topic: str = "/joint_states",
        image_topic_map: dict[str, str] | None = None,
        proprio_config: list[dict[str, Any]] | None = None,
        gripper_limits: tuple[float, float] = GRIPPER_LIMITS,
        gripper_invert: bool = False,
        max_pos_step_m: float | None = None,
        max_rot_step_rad: float | None = None,
        use_cameras: bool = True,
        wait_for_data_timeout: float = 10.0,
    ):
        # ROS 2 只在真正连接真机时才 import
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, JointState
        from tf2_ros.buffer import Buffer
        from tf2_ros.transform_listener import TransformListener

        self.control_hz = float(control_hz)
        self.control_period = 1.0 / self.control_hz
        self.gripper_limits = tuple(gripper_limits)
        self.gripper_invert = bool(gripper_invert)
        self.max_pos_step_m = max_pos_step_m
        self.max_rot_step_rad = max_rot_step_rad
        self.wait_for_data_timeout = float(wait_for_data_timeout)

        self.image_topic_map = dict(image_topic_map or MARVIN_DEFAULT_IMAGE_TOPICS)
        self.proprio_config = list(proprio_config or [MARVIN_LEFT_ARM, MARVIN_RIGHT_ARM])

        if not rclpy.ok():
            rclpy.init()
        self.node: Node = rclpy.create_node(node_name)

        # --- joint states ---
        self._joint_state_lock = threading.Lock()
        self._joint_positions: dict[str, float] = {}
        self._joint_state_time = 0.0
        self._joint_sub = self.node.create_subscription(
            JointState,
            joint_state_topic,
            self._on_joint_state,
            qos_profile_sensor_data,
        )

        # --- cameras ---
        self._image_lock = threading.Lock()
        self._latest_images: dict[str, np.ndarray] = {}
        self._image_subs: list[Any] = []
        self.use_cameras = bool(use_cameras)
        if self.use_cameras:
            for obs_key, topic in self.image_topic_map.items():
                self._image_subs.append(
                    self.node.create_subscription(
                        Image,
                        topic,
                        lambda msg, k=obs_key: self._on_image(msg, k),
                        qos_profile_sensor_data,
                    )
                )

        # --- TF ---
        self._tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=2.0))
        self._tf_listener = TransformListener(self._tf_buffer, self.node)

        # --- publishers ---
        from moveit_msgs.msg import CartesianTrajectory
        from trajectory_msgs.msg import JointTrajectory

        self._pose_pubs: dict[str, Any] = {}
        self._gripper_pubs: dict[str, Any] = {}
        for cfg in self.proprio_config:
            self._pose_pubs[cfg["side"]] = self.node.create_publisher(
                CartesianTrajectory, cfg["pose_topic"], 1
            )
            self._gripper_pubs[cfg["side"]] = self.node.create_publisher(
                JointTrajectory, cfg["gripper_topic"], 1
            )

        # --- 后台 spin ---
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        self._last_cmd_pose: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # ROS callbacks
    # ------------------------------------------------------------------ #
    def _on_joint_state(self, msg) -> None:
        positions = {name: float(pos) for name, pos in zip(msg.name, msg.position)}
        with self._joint_state_lock:
            self._joint_positions.update(positions)
            self._joint_state_time = time.monotonic()

    def _on_image(self, msg, obs_key: str) -> None:
        img = self._image_msg_to_numpy(msg)
        if img is None:
            return
        with self._image_lock:
            self._latest_images[obs_key] = img

    @staticmethod
    def _image_msg_to_numpy(msg) -> np.ndarray | None:
        if msg.encoding.lower() not in {"rgb8", "bgr8"}:
            return None
        data = np.frombuffer(msg.data, dtype=np.uint8)
        rows = data.reshape(msg.height, msg.step)
        img = rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        if msg.encoding.lower() == "bgr8":
            img = img[..., ::-1]
        return img.copy()

    # ------------------------------------------------------------------ #
    # 公共接口
    # ------------------------------------------------------------------ #
    def reset(self) -> dict[str, Any]:
        """等待真机观测就绪；真机复位请在场外完成，这里不主动移动机械臂。"""
        print("[MarvinRealEnv] reset: waiting for observations...")
        self._wait_for_joint_state()
        self._last_cmd_pose.clear()
        print("[MarvinRealEnv] reset: ready")
        return self.get_obs()

    def get_obs(self) -> dict[str, Any]:
        obs: dict[str, Any] = {}

        # 图像
        with self._image_lock:
            for obs_key in self.image_topic_map:
                obs[obs_key] = self._latest_images.get(obs_key)

        # 本体感受：TCP 位姿 + gripper 开合
        for cfg in self.proprio_config:
            pos, quat_xyzw = self._get_tcp_pose(cfg["base_frame"], cfg["tcp_frame"])
            gripper_joint_pos = self._get_joint_position(cfg["gripper_joint_name"])
            gripper_norm = self._normalize_gripper(gripper_joint_pos)

            obs[cfg["pos_key"]] = pos.astype(np.float32)
            obs[cfg["quat_key"]] = quat_xyzw.astype(np.float32)
            obs[cfg["gripper_key"]] = np.array([gripper_norm], dtype=np.float32)

        return obs

    def step(self, action_dict: dict[str, Any] | None = None) -> None:
        """
        action_dict 格式与 policy_local_utils.decode_action_vector 对
        real_bimanual_base_rot6d 的输出一致：
          {
            "base_velocity": [vx, vy, vomega],   # 当前固定底盘，忽略
            "arm_left":  {"arm_pos": [...], "arm_quat": [x,y,z,w], "gripper_pos": [v]},
            "arm_right": {"arm_pos": [...], "arm_quat": [x,y,z,w], "gripper_pos": [v]},
          }
        传入 None 表示保持当前位姿。
        """
        for cfg in self.proprio_config:
            side = cfg["side"]
            if action_dict is None:
                arm_action = None
            elif side == "left":
                arm_action = action_dict.get("arm_left")
            elif side == "right":
                arm_action = action_dict.get("arm_right")
            else:
                arm_action = None

            # 获取当前位姿作为 hold 基准或安全裁剪参考
            cur_pos, cur_quat = self._get_tcp_pose(cfg["base_frame"], cfg["tcp_frame"])

            if arm_action is None:
                target_pos, target_quat = cur_pos, cur_quat
                gripper_norm = self._normalize_gripper(
                    self._get_joint_position(cfg["gripper_joint_name"])
                )
            else:
                target_pos = np.asarray(arm_action["arm_pos"], dtype=np.float64)
                target_quat = np.asarray(arm_action["arm_quat"], dtype=np.float64)
                gripper_norm = float(np.asarray(arm_action["gripper_pos"]).reshape(-1)[0])

                # 可选的每步安全限幅
                target_pos, target_quat = self._apply_step_safety(
                    cur_pos, cur_quat, target_pos, target_quat
                )

            self._send_pose_command(cfg, target_pos, target_quat)
            self._send_gripper_command(cfg, gripper_norm)

    def render(self, obs_key: str | None = None) -> np.ndarray | None:
        """返回一帧 RGB 图像用于视频录制；默认取 head_image。"""
        with self._image_lock:
            if obs_key is None:
                if "head_image" in self._latest_images:
                    return self._latest_images["head_image"].copy()
                if self._latest_images:
                    return next(iter(self._latest_images.values())).copy()
                return None
            img = self._latest_images.get(obs_key)
            return img.copy() if img is not None else None

    def close(self) -> None:
        print("[MarvinRealEnv] closing...")
        import rclpy
        self._executor.shutdown()
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("[MarvinRealEnv] closed")

    # ------------------------------------------------------------------ #
    # 底层 helpers
    # ------------------------------------------------------------------ #
    def _wait_for_joint_state(self) -> None:
        deadline = time.monotonic() + self.wait_for_data_timeout
        while time.monotonic() < deadline:
            with self._joint_state_lock:
                if self._joint_positions and (
                    time.monotonic() - self._joint_state_time < 1.0
                ):
                    return
            time.sleep(0.05)
        raise TimeoutError("Timed out waiting for /joint_states")

    def _get_joint_position(self, joint_name: str) -> float:
        with self._joint_state_lock:
            return self._joint_positions.get(joint_name, 0.0)

    def _normalize_gripper(self, joint_pos: float) -> float:
        lower, upper = self.gripper_limits
        if upper <= lower:
            return 0.0
        norm = float(np.clip((joint_pos - lower) / (upper - lower), 0.0, 1.0))
        return 1.0 - norm if self.gripper_invert else norm

    def _denormalize_gripper(self, norm: float) -> float:
        lower, upper = self.gripper_limits
        norm = float(np.clip(norm, 0.0, 1.0))
        if self.gripper_invert:
            norm = 1.0 - norm
        return lower + norm * (upper - lower)

    def _get_tcp_pose(self, base_frame: str, tcp_frame: str) -> tuple[np.ndarray, np.ndarray]:
        import rclpy
        now = rclpy.time.Time()
        timeout = rclpy.duration.Duration(seconds=0.2)
        try:
            t = self._tf_buffer.lookup_transform(base_frame, tcp_frame, now, timeout)
            pos = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
            quat = np.array([
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w,
            ])
            return pos, quat
        except Exception as exc:
            print(f"[MarvinRealEnv] TF lookup {tcp_frame}->{base_frame} failed: {exc}")
            return np.zeros(3, dtype=np.float64), np.array([0.0, 0.0, 0.0, 1.0])

    def _apply_step_safety(
        self,
        cur_pos: np.ndarray,
        cur_quat: np.ndarray,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.max_pos_step_m is not None:
            delta = target_pos - cur_pos
            dist = np.linalg.norm(delta)
            if dist > self.max_pos_step_m:
                target_pos = cur_pos + delta * (self.max_pos_step_m / dist)
        if self.max_rot_step_rad is not None:
            # 当前姿态 -> 目标姿态的最短路径球面插值
            r_cur = Rotation.from_quat(cur_quat)
            r_tgt = Rotation.from_quat(target_quat)
            rel = r_cur.inv() * r_tgt
            rotvec = rel.as_rotvec()
            angle = np.linalg.norm(rotvec)
            if angle > self.max_rot_step_rad:
                frac = self.max_rot_step_rad / angle
                r_safe = r_cur * Rotation.from_rotvec(rotvec * frac)
                target_quat = r_safe.as_quat()
        return target_pos, target_quat

    def _send_pose_command(self, cfg: dict[str, Any], pos: np.ndarray, quat_xyzw: np.ndarray) -> None:
        from moveit_msgs.msg import CartesianTrajectory, CartesianTrajectoryPoint
        from builtin_interfaces.msg import Duration

        msg = CartesianTrajectory()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = cfg["base_frame"]
        msg.tracked_frame = cfg["tcp_frame"]

        point = CartesianTrajectoryPoint()
        point.point.pose.position.x = float(pos[0])
        point.point.pose.position.y = float(pos[1])
        point.point.pose.position.z = float(pos[2])
        # 确保 w 非负，保持四元数手性一致
        if quat_xyzw[3] < 0.0:
            quat_xyzw = -quat_xyzw
        point.point.pose.orientation.x = float(quat_xyzw[0])
        point.point.pose.orientation.y = float(quat_xyzw[1])
        point.point.pose.orientation.z = float(quat_xyzw[2])
        point.point.pose.orientation.w = float(quat_xyzw[3])

        dt = self.control_period
        point.time_from_start = Duration(sec=int(dt), nanosec=int((dt % 1.0) * 1e9))
        msg.points.append(point)

        self._pose_pubs[cfg["side"]].publish(msg)
        self._last_cmd_pose[cfg["side"]] = {"pos": pos.copy(), "quat": quat_xyzw.copy()}

    def _send_gripper_command(self, cfg: dict[str, Any], gripper_norm: float) -> None:
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from builtin_interfaces.msg import Duration

        joint_pos = self._denormalize_gripper(gripper_norm)
        msg = JointTrajectory()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.joint_names = [cfg["gripper_joint_name"]]
        point = JointTrajectoryPoint()
        point.positions = [float(joint_pos)]
        dt = self.control_period
        point.time_from_start = Duration(sec=int(dt), nanosec=int((dt % 1.0) * 1e9))
        msg.points.append(point)
        self._gripper_pubs[cfg["side"]].publish(msg)


# --------------------------------------------------------------------------- #
# B-spline 真机 runner
# --------------------------------------------------------------------------- #
class MarvinRealBSplineRunner(BaseImageRunner):
    """
    仿照 RobomimicBSplineRunner 的接口，在 Marvin 真机上运行 B-spline 策略。

    与仿真 runner 的主要区别：
      - 不需要 dataset_path / env_meta，直接构造 MarvinRealEnv。
      - 控制频率、相机话题、TF frame、夹爪限幅均可配置。
      - 视频录制使用头部（或指定）相机画面。
    """

    def __init__(
        self,
        output_dir: str,
        ckpt_path: str,
        *,
        n_rollouts: int = 1,
        max_steps: int = 800,
        control_hz: float = 20.0,
        fps: int = 20,
        crf: int = 22,
        device: str = "cuda",
        render_obs_key: str | None = "head_image",
        joint_state_topic: str = "/joint_states",
        image_topic_map: dict[str, str] | None = None,
        proprio_config: list[dict[str, Any]] | None = None,
        gripper_limits: tuple[float, float] = GRIPPER_LIMITS,
        gripper_invert: bool = False,
        max_pos_step_m: float | None = None,
        max_rot_step_rad: float | None = None,
        use_cameras: bool = True,
        # PolicyLocalBSpline 参数
        speed_up_times: float = 1.0,
        predict_before_end: float = 0.06,
        origin_time_scale: float = 10.0,
        use_action_derivatives: bool = False,
        disable_time_align: bool = False,
        time_align_error_threshold: float = 0.1,
        time_align_larger_t: float = 1.0,
        restart_on_time_align_error: bool = False,
        consider_gripper_during_align: bool = False,
        gripper_slowdown_enabled: bool = False,
        gripper_slowdown_threshold: float = 0.08,
        gripper_slowdown_steps: int = 7,
        normalize_knots_zero: bool = True,
        obs_stride: int = 1,
        tqdm_interval_sec: float = 5.0,
    ):
        super().__init__(output_dir)

        self.max_steps = int(max_steps)
        self.control_hz = float(control_hz)
        self.control_period = 1.0 / self.control_hz
        self.fps = int(fps)
        self.crf = int(crf)
        self.render_obs_key = render_obs_key
        self.tqdm_interval_sec = tqdm_interval_sec
        self.n_rollouts = int(n_rollouts)

        self.env = MarvinRealEnv(
            node_name="marvin_real_env",
            control_hz=control_hz,
            joint_state_topic=joint_state_topic,
            image_topic_map=image_topic_map,
            proprio_config=proprio_config,
            gripper_limits=gripper_limits,
            gripper_invert=gripper_invert,
            max_pos_step_m=max_pos_step_m,
            max_rot_step_rad=max_rot_step_rad,
            use_cameras=use_cameras,
        )

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

        self.shape_meta = copy.deepcopy(self.policy.model.cfg.shape_meta)
        if self.render_obs_key is None:
            rgb_keys = [
                k for k, v in self.shape_meta.get("obs", {}).items()
                if v.get("type") == "rgb"
            ]
            self.render_obs_key = rgb_keys[0] if rgb_keys else None

        self.video_recorder = VideoRecorder.create_h264(
            fps=self.fps,
            codec="h264",
            input_pix_fmt="rgb24",
            crf=self.crf,
            thread_type="FRAME",
            thread_count=1,
        )

    def run(self, policy=None):
        """执行真机 rollout。外部传入的 policy 会被忽略，runner 使用构造时加载的策略。"""
        del policy

        import tqdm
        import wandb

        output_dir = pathlib.Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        media_dir = output_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)

        log_data = {}

        for rollout_idx in range(self.n_rollouts):
            print(f"\n[MarvinRealBSplineRunner] rollout {rollout_idx + 1}/{self.n_rollouts}")
            video_path = media_dir / (uuid.uuid4().hex + ".mp4")

            self.env.reset()
            self.policy.reset()
            self.video_recorder.start(video_path)

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Real {rollout_idx}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            for step_count in range(self.max_steps):
                loop_start = time.perf_counter()

                obs = self.env.get_obs()
                obs = self._ensure_obs_keys(obs)
                action_dict = self.policy.step(obs)
                if action_dict is None:
                    action_dict = self.policy.poll_action()
                # 保持当前位姿（不发送零动作）
                self.env.step(action_dict)

                frame = self.env.render(self.render_obs_key)
                if frame is not None:
                    self.video_recorder.write_frame(frame)

                # 维持控制频率
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0.0, self.control_period - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

                pbar.update(1)

            pbar.close()
            self.video_recorder.stop()

            if video_path.exists():
                log_data[f"real_video_{rollout_idx}"] = wandb.Video(str(video_path))
            print(f"[MarvinRealBSplineRunner] saved video: {video_path}")

        self.policy.wait_for_pending_inference(timeout=10.0)
        self.policy.print_inference_summary()
        return log_data

    def _ensure_obs_keys(self, obs: dict[str, Any]) -> dict[str, Any]:
        """补全 shape_meta 中存在但观测里缺失的低维 key（如 base_velocity）。"""
        for key, meta in self.shape_meta.get("obs", {}).items():
            if key in obs:
                continue
            shape = tuple(meta.get("shape", []))
            if meta.get("type") == "rgb":
                obs[key] = np.zeros(shape, dtype=np.uint8)
            else:
                print(f"[MarvinRealBSplineRunner] warning: filling missing obs key {key!r} with zeros")
                obs[key] = np.zeros(shape, dtype=np.float32)
        return obs

    def close(self):
        self.video_recorder.stop()
        self.env.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run B-spline policy on real Marvin bimanual robot")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n_rollouts", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--control_hz", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gripper_invert", action="store_true")
    parser.add_argument("--max_pos_step_m", type=float, default=None)
    parser.add_argument("--max_rot_step_rad", type=float, default=None)
    parser.add_argument("--speed_up_times", type=float, default=1.0)
    parser.add_argument("--predict_before_end", type=float, default=0.06)

    args = parser.parse_args()

    runner = MarvinRealBSplineRunner(
        output_dir=args.output_dir,
        ckpt_path=args.ckpt,
        n_rollouts=args.n_rollouts,
        max_steps=args.max_steps,
        control_hz=args.control_hz,
        fps=args.fps,
        device=args.device,
        gripper_invert=args.gripper_invert,
        max_pos_step_m=args.max_pos_step_m,
        max_rot_step_rad=args.max_rot_step_rad,
        speed_up_times=args.speed_up_times,
        predict_before_end=args.predict_before_end,
    )
    try:
        log_data = runner.run()
        print(log_data)
    finally:
        runner.close()
