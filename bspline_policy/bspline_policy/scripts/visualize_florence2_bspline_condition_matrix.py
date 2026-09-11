#!/usr/bin/env python3
"""Mimic ``visualize_bspline_condition_matrix.py`` for the Florence-2 encoder.

For a contiguous sequence of dataset samples the script:

  1. loads a Florence-2-backed policy checkpoint (optional) or the raw
     Florence2VisionEncoder,
  2. extracts the latest RGB frame per camera,
  3. computes each camera's contribution to the global condition vector,
  4. plots an image strip + condition-matrix heatmap for every camera,
  5. saves the B-spline action parameter matrix as an additional heatmap,
  6. saves the full concatenated global condition matrix as a reference,
  7. (optional) visualizes intermediate Florence-2 vision-tower feature maps,
     upsampled to the cropped image size and overlaid on the input image for
     direct spatial comparison.

The resulting PNGs and an ``.npz`` with raw tensors are written to
``--output_dir``.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from omegaconf import OmegaConf


# --------------------------------------------------------------------------- #
# Path setup
# --------------------------------------------------------------------------- #
_ROOT = Path(__file__).resolve().parents[3]
for _p in (
    _ROOT,
    _ROOT / "bspline_policy",
    _ROOT / "diffusion_policy",
    _ROOT / "robomimic",
):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

import hydra
from diffusion_policy.policy.florence2 import Florence2VisionEncoder


# --------------------------------------------------------------------------- #
# OmegaConf resolvers (same as the robomimic visualizer)
# --------------------------------------------------------------------------- #
try:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
except Exception:
    pass

try:
    OmegaConf.register_new_resolver("now", lambda *_args: "00000000", replace=True)
except Exception:
    pass


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def rgb_tensor_to_uint8(img: torch.Tensor) -> np.ndarray:
    """Convert an RGB tensor in (C,H,W) to a uint8 numpy array in (H,W,C)."""
    img = img.detach().cpu().numpy()
    if img.dtype == np.uint8:
        pass
    else:
        img = (img * 255.0).clip(0, 255)
        img = img.astype(np.uint8)
    if img.shape[0] in (1, 3):
        img = np.moveaxis(img, 0, -1)
    return img


def maybe_normalize_per_dim(matrix: np.ndarray, normalize: bool) -> np.ndarray:
    if not normalize or matrix.size == 0:
        return matrix
    min_v = matrix.min(axis=0, keepdims=True)
    max_v = matrix.max(axis=0, keepdims=True)
    rng = max_v - min_v
    rng[rng == 0] = 1.0
    return (matrix - min_v) / rng


def infer_spatial_grid(num_tokens: int) -> Tuple[int, int]:
    side = int(math.isqrt(num_tokens))
    if side * side == num_tokens:
        return side, side
    best = (1, num_tokens)
    best_ratio = float(num_tokens)
    for h in range(1, side + 1):
        if num_tokens % h == 0:
            w = num_tokens // h
            ratio = max(h, w) / min(h, w)
            if ratio < best_ratio:
                best_ratio = ratio
                best = (h, w)
    return best


# --------------------------------------------------------------------------- #
# Plotting functions (matching visualize_bspline_condition_matrix.py style)
# --------------------------------------------------------------------------- #
def plot_camera_condition(
    output_path: Path,
    camera_name: str,
    frames: List[np.ndarray],
    cond_matrix: np.ndarray,
    cmap: str = "viridis",
    normalize: bool = False,
):
    """Save one PNG: image strip on top, condition heatmap below."""
    n_frames = len(frames)
    if n_frames == 0:
        return

    strip = np.concatenate(frames, axis=1)
    plot_matrix = maybe_normalize_per_dim(cond_matrix, normalize)

    fig, (ax_img, ax_cond) = plt.subplots(
        2, 1,
        figsize=(max(8, n_frames * 1.4), 6),
        gridspec_kw={"height_ratios": [1, 1.2]},
    )

    ax_img.imshow(strip)
    ax_img.set_title(f"{camera_name} consecutive frames", fontsize=12)
    ax_img.axis("off")

    im = ax_cond.imshow(
        plot_matrix.T,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
    )
    ax_cond.set_xlabel("frame index")
    ax_cond.set_ylabel("feature dimension")
    title = f"{camera_name} condition matrix"
    if normalize:
        title += " (per-dim normalized)"
    ax_cond.set_title(title, fontsize=12)
    ax_cond.set_xticks(np.arange(n_frames))
    fig.colorbar(im, ax=ax_cond)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_global_condition(
    output_path: Path,
    global_matrix: np.ndarray,
    key_boundaries: List[int],
    key_names: List[str],
    normalize: bool = False,
):
    """Save a heatmap of the full concatenated global condition matrix."""
    n_frames = global_matrix.shape[0]
    plot_matrix = maybe_normalize_per_dim(global_matrix, normalize)

    fig, ax = plt.subplots(figsize=(max(8, n_frames * 0.5), 5))
    im = ax.imshow(
        plot_matrix.T,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_xlabel("frame index")
    ax.set_ylabel("concatenated feature dimension")
    title = "global condition matrix"
    if normalize:
        title += " (per-dim normalized)"
    ax.set_title(title, fontsize=12)
    ax.set_xticks(np.arange(n_frames))

    for b in key_boundaries[1:-1]:
        ax.axhline(y=b - 0.5, color="white", linewidth=0.8, alpha=0.6)

    mid_points = [
        (key_boundaries[i] + key_boundaries[i + 1]) / 2
        for i in range(len(key_names))
    ]
    for name, y in zip(key_names, mid_points):
        ax.text(
            n_frames + 0.5, y, name,
            color="black", fontsize=8, va="center",
        )

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_action_condition(
    output_path: Path,
    action_matrices: np.ndarray,
):
    """Save a heatmap of the B-spline parameter matrix across frames."""
    n_frames, n_steps, n_channels = action_matrices.shape
    fig, axes = plt.subplots(
        n_channels,
        1,
        figsize=(max(8, n_frames * 0.6), 1.8 * n_channels),
        squeeze=False,
    )

    for c in range(n_channels):
        ax = axes[c, 0]
        mat = action_matrices[:, :, c]  # (n_frames, n_steps)
        vmax = np.abs(mat).max()
        vmin = -vmax if np.any(mat < 0) else 0.0
        im = ax.imshow(mat.T, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
        if c == 0:
            ax.set_title("knot vector", fontsize=10)
        else:
            ax.set_title(f"control point dim {c}", fontsize=10)
        ax.set_ylabel("param step")
        ax.set_xticks(np.arange(n_frames))
        fig.colorbar(im, ax=ax)

    axes[-1, 0].set_xlabel("frame index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def blend_activation_heatmap(
    image: np.ndarray,
    activation: np.ndarray,
    cmap: str = "jet",
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a single-channel activation map onto an RGB image."""
    H, W = image.shape[:2]
    act_t = torch.from_numpy(activation.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    act_up = F.interpolate(act_t, size=(H, W), mode="bilinear", align_corners=False)
    act_up = act_up.squeeze(0).squeeze(0).numpy()

    mn, mx = act_up.min(), act_up.max()
    if mx > mn:
        act_up = (act_up - mn) / (mx - mn)
    else:
        act_up = np.zeros_like(act_up)

    colored = (plt.get_cmap(cmap)(act_up)[:, :, :3] * 255).astype(np.uint8)
    blended = image.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha
    return blended.clip(0, 255).astype(np.uint8)


def plot_intermediate_overlays(
    output_path: Path,
    camera_name: str,
    layer_name: str,
    images: List[np.ndarray],
    features: np.ndarray,
    max_channels: int = 8,
    alpha: float = 0.5,
    cmap: str = "jet",
):
    """Save aligned vision-tower feature-map overlays for direct image comparison.

    ``images`` is a list of length T of uint8 arrays (H, W, 3).
    ``features`` has shape (T, C, h, w).  Each feature channel is upsampled
    to (H, W) and overlaid on the corresponding image.  The first column
    shows the original cropped image; remaining columns show overlays.
    """
    n_frames = len(images)
    n_channels = min(features.shape[1], max_channels)
    target_size = images[0].shape[:2]  # (H, W)

    n_cols = 1 + n_channels
    fig, axes = plt.subplots(
        n_frames,
        n_cols,
        figsize=(n_cols * 2.5, n_frames * 2.5),
        squeeze=False,
    )

    for t in range(n_frames):
        img = images[t]
        axes[t, 0].imshow(img)
        axes[t, 0].axis("off")
        axes[t, 0].set_title(f"frame {t}", fontsize=9)

        feat = torch.from_numpy(features[t][:n_channels]).unsqueeze(0).float()
        feat_up = F.interpolate(
            feat,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )[0].numpy()  # (n_channels, H, W)

        for c in range(n_channels):
            overlay = blend_activation_heatmap(
                img, feat_up[c], cmap=cmap, alpha=alpha
            )
            axes[t, c + 1].imshow(overlay)
            axes[t, c + 1].axis("off")
            axes[t, c + 1].set_title(f"ch {c}", fontsize=8)

    fig.suptitle(
        f"{camera_name} / {layer_name}  upsampled feature overlays",
        fontsize=12,
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Florence-2 intermediate feature extraction
# --------------------------------------------------------------------------- #
def make_hook_storage() -> Tuple[List[torch.Tensor], callable]:
    storage: List[torch.Tensor] = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            storage.append(output[0].detach())
        else:
            storage.append(output.detach())

    return storage, hook


def extract_intermediate_features(
    encoder: Florence2VisionEncoder,
    pixel_values: torch.Tensor,
    stage_indices: List[int],
) -> Dict[str, torch.Tensor]:
    """Extract selected stage outputs from the Florence-2 vision tower.

    Returns a dict mapping ``stage{i}`` to a tensor of shape (B, C, H, W).
    """
    vision_tower = encoder.model.vision_tower
    handles = []
    storages = []

    try:
        for idx, block in enumerate(vision_tower.blocks):
            storage, hook = make_hook_storage()
            handles.append(block.register_forward_hook(hook))
            storages.append(storage)

        _ = encoder._encode_image(pixel_values)

        features = {}
        for idx, storage in enumerate(storages):
            if idx not in stage_indices or len(storage) == 0:
                continue
            feat = storage[0]
            if feat.dim() == 3:
                B, L, C = feat.shape
                side = int(math.isqrt(L))
                if side * side == L:
                    feat = feat.transpose(1, 2).reshape(B, C, side, side)
                else:
                    h, w = infer_spatial_grid(L)
                    feat = feat.transpose(1, 2).reshape(B, C, h, w)
            features[f"stage{idx}"] = feat
        return features
    finally:
        for h in handles:
            h.remove()


# --------------------------------------------------------------------------- #
# Policy / encoder loading
# --------------------------------------------------------------------------- #
def is_bspline_config(cfg: OmegaConf) -> bool:
    """Heuristic: dataset or policy mentions B-spline."""
    ds_target = cfg.task.dataset.get("_target_", "")
    if "bspline" in ds_target.lower():
        return True
    if cfg.task.dataset.get("bspline_degree") is not None:
        return True
    if cfg.policy.get("bspline_degree") is not None:
        return True
    return False


def load_policy_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    use_ema: bool = False,
):
    """Load a policy (or its EMA copy) from a workspace checkpoint."""
    import dill

    payload = torch.load(
        str(checkpoint_path),
        pickle_module=dill,
        map_location="cpu",
    )
    cfg = payload["cfg"]

    if is_bspline_config(cfg):
        try:
            OmegaConf.set_struct(cfg.policy, False)
        except Exception:
            pass
        cfg.policy._target_ = (
            "bspline_policy.policy.diffusion_unet_bspline_image_policy"
            ".DiffusionUnetBSplineImagePolicy"
        )
        if "bspline_degree" not in cfg.policy:
            cfg.policy.bspline_degree = cfg.task.dataset.get("bspline_degree", 3)

    model = hydra.utils.instantiate(cfg.policy)
    model.load_state_dict(payload["state_dicts"]["model"])
    model.to(device)
    model.eval()

    target_model = model
    if use_ema and "ema_model" in payload["state_dicts"]:
        ema_model = hydra.utils.instantiate(cfg.policy)
        ema_model.load_state_dict(payload["state_dicts"]["ema_model"])
        ema_model.to(device)
        ema_model.eval()
        target_model = ema_model

    return target_model, cfg


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Mimic visualize_bspline_condition_matrix.py for Florence-2."
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=Path("my_dataset/transport/transport_abs.hdf5"),
        help="Path to the Robomimic HDF5 dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/florence2_bspline_condition_matrix"),
        help="Directory to write PNGs and raw data.",
    )
    parser.add_argument("--start_idx", type=int, default=0, help="First dataset index to visualize.")
    parser.add_argument("--n_frames", type=int, default=16, help="Number of consecutive frames to visualize.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional policy checkpoint. If omitted, only the raw Florence-2 encoder is used.",
    )
    parser.add_argument("--use_ema", action="store_true", help="Use EMA weights from the checkpoint.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Florence-2-base",
        help="HuggingFace model id for Florence-2.",
    )
    parser.add_argument("--output_dim", type=int, default=512, help="Per-camera projection dimension.")
    parser.add_argument("--image_size", type=int, default=768, help="Florence-2 input resolution.")
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap for condition matrices.")
    parser.add_argument(
        "--normalize_heatmap",
        action="store_true",
        help="Normalize condition matrices per dimension before plotting.",
    )
    parser.add_argument(
        "--intermediate",
        action="store_true",
        help="Also visualize intermediate Florence-2 vision-tower feature overlays.",
    )
    parser.add_argument(
        "--intermediate_stages",
        type=str,
        default="0,1,2,3",
        help="Comma-separated vision-tower stage indices to visualize.",
    )
    parser.add_argument("--intermediate_channels", type=int, default=8, help="Max channels per intermediate overlay.")
    parser.add_argument("--intermediate_alpha", type=float, default=0.5, help="Opacity for intermediate overlays.")
    parser.add_argument("--intermediate_size", type=int, default=76, help="Center-crop size for intermediate overlays.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    stage_indices = [int(x.strip()) for x in args.intermediate_stages.split(",") if x.strip()]

    # Bimanual transport shape metadata.
    shape_meta = {
        "obs": {
            "shouldercamera0_image": {"shape": [3, 84, 84], "type": "rgb"},
            "robot1_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
            "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
            "robot1_eef_pos": {"shape": [3]},
            "robot1_eef_quat": {"shape": [4]},
            "robot1_gripper_qpos": {"shape": [2]},
            "robot0_eef_pos": {"shape": [3]},
            "robot0_eef_quat": {"shape": [4]},
            "robot0_gripper_qpos": {"shape": [2]},
        },
        "action": {"shape": [20]},
    }
    rgb_keys = sorted(k for k, v in shape_meta["obs"].items() if v.get("type") == "rgb")
    lowdim_keys = sorted(k for k, v in shape_meta["obs"].items() if v.get("type") != "rgb")

    dataset_cfg = OmegaConf.create({
        "_target_": "bspline_policy.dataset.robomimic_replay_bspline_image_dataset.RobomimicReplayBSplineImageDataset",
        "shape_meta": shape_meta,
        "dataset_path": str(args.dataset_path),
        "horizon": 1,
        "pad_before": 0,
        "pad_after": 0,
        "n_obs_steps": 1,
        "abs_action": True,
        "rotation_rep": "rotation_6d",
        "chunk_size": 10,
        "bspline_degree": 3,
        "max_error": 0.002,
        "stride": 1,
        "relative_knots": False,
        "use_cache": True,
        "cache_suffix": "bimanual_data_cache",
        "cache_decoded_replay": False,
        "cache_preprocessed_samples": False,
        "seed": 42,
        "val_ratio": 0.0,
    })
    dataset = hydra.utils.instantiate(dataset_cfg)
    if args.start_idx + args.n_frames > len(dataset):
        args.n_frames = len(dataset) - args.start_idx
        print(f"Clipped n_frames to {args.n_frames} (dataset length={len(dataset)})")

    # Load model: policy checkpoint or raw encoder.
    encoder: Florence2VisionEncoder
    if args.checkpoint is not None:
        print(f"Loading policy checkpoint: {args.checkpoint}")
        policy, cfg = load_policy_from_checkpoint(args.checkpoint, device, args.use_ema)
        encoder = policy.obs_encoder
        print(f"Policy loaded. Encoder total dim={encoder.output_shape()[0]}")
    else:
        print(f"Loading raw Florence-2 encoder: {args.model_name}")
        encoder = Florence2VisionEncoder(
            shape_meta=shape_meta,
            model_name=args.model_name,
            output_dim=args.output_dim,
            freeze_backbone=True,
            input_range=(0.0, 1.0),
            image_size=(args.image_size, args.image_size),
            dtype=torch.float32,
            device=device,
        )
        encoder = encoder.to(device)
        encoder.eval()
        print(f"Encoder loaded. per-camera dim={args.output_dim}, total dim={encoder.output_shape()[0]}")

    frames_per_camera: Dict[str, List[np.ndarray]] = {k: [] for k in rgb_keys}
    cond_per_camera: Dict[str, List[np.ndarray]] = {k: [] for k in rgb_keys}
    cond_per_lowdim: Dict[str, List[np.ndarray]] = {k: [] for k in lowdim_keys}
    global_vectors: List[np.ndarray] = []
    action_matrices: List[np.ndarray] = []

    # Intermediate inputs: (offset, cropped_image, pixel_values)
    intermediate_inputs: List[Tuple[int, Dict[str, np.ndarray], Dict[str, torch.Tensor]]] = []

    for offset in range(args.n_frames):
        idx = args.start_idx + offset
        sample = dataset[idx]
        obs = sample["obs"]

        obs_last = {k: v[-1:].to(device) for k, v in obs.items()}
        for cam in rgb_keys:
            frames_per_camera[cam].append(rgb_tensor_to_uint8(obs[cam][-1]))

        with torch.no_grad():
            global_vec = encoder(obs_last).detach().cpu().numpy()[0]
        global_vectors.append(global_vec)

        ptr = 0
        for cam in rgb_keys:
            cam_dim = encoder.output_dim if hasattr(encoder, "output_dim") else args.output_dim
            cond_per_camera[cam].append(global_vec[ptr:ptr + cam_dim])
            ptr += cam_dim
        for key in lowdim_keys:
            dim = shape_meta["obs"][key]["shape"][0]
            cond_per_lowdim[key].append(global_vec[ptr:ptr + dim])
            ptr += dim

        action_matrices.append(sample["action"].detach().cpu().numpy())

        if args.intermediate:
            cropped_images = {}
            pixel_values_dict = {}
            features_dict = {}
            for cam in rgb_keys:
                img = rgb_tensor_to_uint8(obs[cam][-1])
                H, W = img.shape[:2]
                crop_size = args.intermediate_size
                top = (H - crop_size) // 2
                left = (W - crop_size) // 2
                cropped = img[top:top + crop_size, left:left + crop_size]
                cropped_images[cam] = cropped
                with torch.no_grad():
                    pixel_values = encoder._preprocess(obs_last[cam])
                    features = extract_intermediate_features(
                        encoder, pixel_values, stage_indices
                    )
                pixel_values_dict[cam] = pixel_values
                features_dict[cam] = features
            intermediate_inputs.append((offset, cropped_images, features_dict))

    # ----------------------------------------------------------------------- #
    # Plot per-camera condition matrices.
    # ----------------------------------------------------------------------- #
    saved_paths = []
    for cam in rgb_keys:
        cond_matrix = np.stack(cond_per_camera[cam], axis=0)
        out_path = args.output_dir / f"condition_matrix_{cam}.png"
        plot_camera_condition(
            output_path=out_path,
            camera_name=cam,
            frames=frames_per_camera[cam],
            cond_matrix=cond_matrix,
            cmap=args.cmap,
            normalize=args.normalize_heatmap,
        )
        saved_paths.append(out_path)
        print(f"Saved: {out_path}")

    # ----------------------------------------------------------------------- #
    # Plot the full concatenated global condition matrix.
    # ----------------------------------------------------------------------- #
    global_matrix = np.stack(global_vectors, axis=0)
    cam_dim = encoder.output_dim if hasattr(encoder, "output_dim") else args.output_dim
    key_dims = [cam_dim] * len(rgb_keys) + [
        shape_meta["obs"][k]["shape"][0] for k in lowdim_keys
    ]
    key_names = rgb_keys + lowdim_keys
    key_boundaries = [0] + list(np.cumsum(key_dims))
    global_out = args.output_dir / "condition_matrix_global.png"
    plot_global_condition(
        output_path=global_out,
        global_matrix=global_matrix,
        key_boundaries=key_boundaries,
        key_names=key_names,
        normalize=args.normalize_heatmap,
    )
    saved_paths.append(global_out)
    print(f"Saved: {global_out}")

    # ----------------------------------------------------------------------- #
    # Plot B-spline action parameter matrix.
    # ----------------------------------------------------------------------- #
    action_matrix = np.stack(action_matrices, axis=0)  # (T, n_steps, n_channels)
    action_out = args.output_dir / "condition_matrix_bspline_action.png"
    plot_action_condition(action_out, action_matrix)
    saved_paths.append(action_out)
    print(f"Saved: {action_out}")

    # ----------------------------------------------------------------------- #
    # Intermediate Florence-2 vision-tower feature overlays.
    # ----------------------------------------------------------------------- #
    if args.intermediate:
        intermediate_dir = args.output_dir / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        # Aggregate intermediate features across frames.
        intermediate_features: Dict[str, Dict[str, List[torch.Tensor]]] = {
            cam: {f"stage{s}": [] for s in stage_indices}
            for cam in rgb_keys
        }
        intermediate_images: Dict[str, List[np.ndarray]] = {cam: [] for cam in rgb_keys}

        for offset, cropped_images, features_dict in intermediate_inputs:
            for cam in rgb_keys:
                intermediate_images[cam].append(cropped_images[cam])
                for layer_name, feat in features_dict[cam].items():
                    intermediate_features[cam][layer_name].append(feat)

        for cam in rgb_keys:
            for layer_name in sorted(intermediate_features[cam].keys()):
                feat_list = intermediate_features[cam][layer_name]
                if not feat_list:
                    continue
                feat_array = torch.cat(feat_list, dim=0).cpu().numpy()
                out_path = intermediate_dir / f"{cam}_{layer_name}.png"
                plot_intermediate_overlays(
                    output_path=out_path,
                    camera_name=cam,
                    layer_name=layer_name,
                    images=intermediate_images[cam],
                    features=feat_array,
                    max_channels=args.intermediate_channels,
                    alpha=args.intermediate_alpha,
                    cmap="jet",
                )
                saved_paths.append(out_path)
                print(f"Saved: {out_path}")

    # Save raw matrices.
    npz_path = args.output_dir / "condition_matrices.npz"
    save_dict = {
        "bspline_action": action_matrix,
        "global_condition": global_matrix,
    }
    for cam in rgb_keys:
        save_dict[f"obs_{cam}"] = np.stack(cond_per_camera[cam], axis=0)
        save_dict[f"frames_{cam}"] = np.stack(frames_per_camera[cam], axis=0)
    for key in lowdim_keys:
        save_dict[f"lowdim_{key}"] = np.stack(cond_per_lowdim[key], axis=0)
    np.savez_compressed(npz_path, **save_dict)
    print(f"Saved raw data: {npz_path}")

    print("\nVisualization complete.")
    for p in saved_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
