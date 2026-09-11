#!/usr/bin/env python3
"""Visualize the per-camera condition matrices produced by Florence2VisionEncoder.

This script mimics ``visualize_bspline_condition_matrix.py``: it takes the same
Robomimic B-spline image dataset as input, replaces the ResNet18 observation
encoder with the pre-trained Florence-2 vision tower, and outputs per-camera
condition-matrix heatmaps plus a global condition-matrix heatmap.

No trained policy checkpoint is required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

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
from diffusion_policy.common.pytorch_util import dict_apply


# --------------------------------------------------------------------------- #
# Plotting helpers (identical style to visualize_bspline_condition_matrix.py)
# --------------------------------------------------------------------------- #
def rgb_tensor_to_uint8(img: torch.Tensor) -> np.ndarray:
    img = img.detach().cpu().numpy()
    if img.dtype != np.uint8:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Visualize Florence2VisionEncoder condition matrices "
                    "(heatmap style, no checkpoint needed)."
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        type=Path,
        help="Path to the robomimic HDF5 dataset.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory where the visualization PNGs will be saved.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Florence-2-base",
        help="HuggingFace model id for Florence-2.",
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        default=512,
        help="Per-camera output dimension of Florence2VisionEncoder.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=768,
        help="Image size the Florence-2 vision tower expects.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="First dataset sample index to visualize.",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=32,
        help="Number of consecutive samples/frames to visualize.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run the encoder on.",
    )
    parser.add_argument(
        "--camera_keys",
        type=str,
        default=None,
        help="Comma-separated list of camera keys. If omitted, all obs keys "
             "with type 'rgb' in shape_meta are used.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap for the condition heatmaps.",
    )
    parser.add_argument(
        "--normalize_heatmap",
        action="store_true",
        help="Min-max normalize each feature dimension before plotting.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Shape meta for the bimanual/transport dataset used previously.
    shape_meta = {
        "obs": {
            "shouldercamera0_image": {"shape": [3, 128, 128], "type": "rgb"},
            "robot1_eye_in_hand_image": {"shape": [3, 128, 128], "type": "rgb"},
            "robot0_eye_in_hand_image": {"shape": [3, 128, 128], "type": "rgb"},
            "robot1_eef_pos": {"shape": [3]},
            "robot1_eef_quat": {"shape": [4]},
            "robot1_gripper_qpos": {"shape": [2]},
            "robot0_eef_pos": {"shape": [3]},
            "robot0_eef_quat": {"shape": [4]},
            "robot0_gripper_qpos": {"shape": [2]},
        },
        "action": {"shape": [20]},
    }

    if args.camera_keys is not None:
        rgb_keys = [k.strip() for k in args.camera_keys.split(",") if k.strip()]
    else:
        rgb_keys = [
            k for k, attr in shape_meta["obs"].items()
            if attr.get("type", "low_dim") == "rgb"
        ]
    lowdim_keys = sorted([
        k for k, attr in shape_meta["obs"].items()
        if attr.get("type", "low_dim") == "low_dim"
    ])
    print(f"Cameras: {rgb_keys}")
    print(f"Low-dim keys: {lowdim_keys}")

    # Build the same B-spline image dataset as the robomimic visualizer.
    print("Building dataset...")
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

    # Load Florence-2 vision encoder.
    print(f"Loading Florence-2 vision encoder: {args.model_name}")
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

    # ----------------------------------------------------------------------- #
    # Iterate over consecutive samples.
    # ----------------------------------------------------------------------- #
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
            cond_per_camera[cam].append(global_vec[ptr:ptr + args.output_dim])
            ptr += args.output_dim
        for key in lowdim_keys:
            dim = shape_meta["obs"][key]["shape"][0]
            cond_per_lowdim[key].append(global_vec[ptr:ptr + dim])
            ptr += dim

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
    key_dims = [args.output_dim] * len(rgb_keys) + [
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

    # Save raw matrices.
    npz_path = args.output_dir / "condition_matrices.npz"
    save_dict = {"global_condition": global_matrix}
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
