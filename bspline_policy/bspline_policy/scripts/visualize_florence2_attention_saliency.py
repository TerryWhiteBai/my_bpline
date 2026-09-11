#!/usr/bin/env python3
"""Visualize where the Florence-2 encoder focuses on each input image.

For every selected video frame and camera, this script computes a pixel-level
saliency map by back-propagating the per-camera condition vector through the
frozen Florence-2 vision tower.  Bright regions in the overlay indicate image
locations that most influence the condition vector passed to the diffusion
policy.

The saliency is computed w.r.t. the original (e.g. 84×84) input image, so it
can be overlaid directly on the original frame without extra cropping.
"""

from __future__ import annotations

import argparse
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
# Helpers
# --------------------------------------------------------------------------- #
def rgb_tensor_to_uint8(img: torch.Tensor) -> np.ndarray:
    """Convert an RGB tensor in (C,H,W) to a uint8 numpy array in (H,W,C)."""
    img = img.detach().cpu().numpy()
    if img.dtype != np.uint8:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    if img.shape[0] in (1, 3):
        img = np.moveaxis(img, 0, -1)
    return img


def compute_condition_saliency(
    encoder: Florence2VisionEncoder,
    img: torch.Tensor,
    target_mode: str = "norm",
    target_dim: int = -1,
) -> torch.Tensor:
    """Compute per-pixel saliency for a single camera frame.

    Args:
        encoder: Florence2VisionEncoder instance.
        img: (1, 3, H, W) image tensor with gradient support.
        target_mode: ``norm`` (L2 norm of condition vector), ``mean``,
            or ``dim`` (a single output dimension).
        target_dim: output dimension index when ``target_mode == "dim"``.

    Returns:
        (H, W) saliency tensor on the original image space.
    """
    img = img.detach().clone().requires_grad_(True)

    pixel_values = encoder._preprocess(img)
    # Bypass the no_grad wrapper inside encoder._encode_image so gradients
    # flow back to the input image.
    tokens = encoder.model._encode_image(pixel_values)
    feat = tokens.mean(dim=1)
    condition = encoder.projection(feat)

    if target_mode == "norm":
        target = condition.norm(dim=-1).sum()
    elif target_mode == "mean":
        target = condition.sum()
    elif target_mode == "dim":
        target = condition[0, target_dim]
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    target.backward()

    # (1, 3, H, W) -> (H, W)
    saliency = img.grad.abs().mean(dim=1)[0]
    return saliency.detach()


def overlay_saliency(
    image: np.ndarray,
    saliency: np.ndarray,
    cmap: str = "jet",
    alpha: float = 0.5,
    normalize: bool = True,
) -> np.ndarray:
    """Overlay a single-channel saliency map on an RGB image.

    Args:
        image: (H, W, 3) uint8 RGB image.
        saliency: (H, W) float saliency map.
        cmap: matplotlib colormap.
        alpha: overlay opacity.
        normalize: whether to min-max normalize the saliency.

    Returns:
        (H, W, 3) uint8 overlaid image.
    """
    saliency = saliency.astype(np.float32)
    if normalize:
        mn, mx = saliency.min(), saliency.max()
        if mx > mn:
            saliency = (saliency - mn) / (mx - mn)
        else:
            saliency = np.zeros_like(saliency)

    colored = (plt.get_cmap(cmap)(saliency)[:, :, :3] * 255).astype(np.uint8)
    blended = image.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha
    return blended.clip(0, 255).astype(np.uint8)


def plot_saliency_strip(
    output_path: Path,
    camera_name: str,
    frames: List[np.ndarray],
    saliencies: List[np.ndarray],
    cmap: str = "jet",
    alpha: float = 0.5,
    n_cols: int = 8,
):
    """Save a grid of saliency overlays for one camera."""
    n_frames = len(frames)
    if n_frames == 0:
        return

    overlays = [overlay_saliency(f, s, cmap=cmap, alpha=alpha) for f, s in zip(frames, saliencies)]

    n_rows = math.ceil(n_frames / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.8))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx < n_frames:
            ax.imshow(overlays[idx])
            ax.set_title(f"t={idx}", fontsize=8)

    fig.suptitle(
        f"{camera_name} – Florence-2 condition saliency ({cmap}, α={alpha})",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_saliency_side_by_side(
    output_path: Path,
    camera_name: str,
    frames: List[np.ndarray],
    saliencies: List[np.ndarray],
    cmap: str = "jet",
    alpha: float = 0.5,
    n_cols: int = 8,
):
    """Save original / saliency / overlay triplets."""
    n_frames = len(frames)
    if n_frames == 0:
        return

    overlays = [overlay_saliency(f, s, cmap=cmap, alpha=alpha) for f, s in zip(frames, saliencies)]

    n_rows = math.ceil(n_frames / n_cols)
    fig, axes = plt.subplots(n_rows * 3, n_cols, figsize=(n_cols * 1.4, n_rows * 4.2))
    if n_rows * 3 == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()

    for idx in range(n_frames):
        col = idx % n_cols
        row = idx // n_cols
        base = row * 3 * n_cols + col

        for ax, img, title in [
            (axes[base], frames[idx], "original"),
            (axes[base + n_cols], saliencies[idx], "saliency"),
            (axes[base + 2 * n_cols], overlays[idx], "overlay"),
        ]:
            ax.axis("off")
            ax.set_title(f"t={idx} {title}", fontsize=7)
            if title == "saliency":
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(img)

    for ax in axes[n_frames * 3:]:
        ax.axis("off")

    fig.suptitle(f"{camera_name} – original / saliency / overlay", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Visualize Florence-2 condition saliency on original frames."
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
        default=Path("outputs/florence2_attention_saliency"),
        help="Directory to write PNGs and raw data.",
    )
    parser.add_argument("--start_idx", type=int, default=0, help="First dataset index to visualize.")
    parser.add_argument("--n_frames", type=int, default=16, help="Number of consecutive frames to visualize.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Florence-2-base",
        help="HuggingFace model id for Florence-2.",
    )
    parser.add_argument("--output_dim", type=int, default=512, help="Per-camera projection dimension.")
    parser.add_argument("--image_size", type=int, default=768, help="Florence-2 input resolution.")
    parser.add_argument("--cmap", type=str, default="jet", help="Matplotlib colormap for the saliency.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Saliency overlay opacity.")
    parser.add_argument(
        "--target_mode",
        type=str,
        default="norm",
        choices=["norm", "mean", "dim"],
        help="What to backprop from the condition vector.",
    )
    parser.add_argument("--target_dim", type=int, default=0, help="Condition dimension index when target_mode=dim.")
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Normalize each saliency map independently before overlay.",
    )
    parser.add_argument(
        "--side_by_side",
        action="store_true",
        help="Also save original/saliency/overlay triplets.",
    )
    parser.add_argument("--n_cols", type=int, default=8, help="Number of columns in the output grid.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

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

    saved_paths = []
    for cam in rgb_keys:
        frames: List[np.ndarray] = []
        saliencies: List[np.ndarray] = []

        for offset in range(args.n_frames):
            idx = args.start_idx + offset
            sample = dataset[idx]
            img_tensor = sample["obs"][cam][-1]  # (3, 84, 84)
            frames.append(rgb_tensor_to_uint8(img_tensor))

            img_batch = img_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                pass
            saliency = compute_condition_saliency(
                encoder,
                img_batch,
                target_mode=args.target_mode,
                target_dim=args.target_dim,
            )
            saliencies.append(saliency.cpu().numpy())
            print(f"  {cam} frame {offset + 1}/{args.n_frames}")

        out_path = args.output_dir / f"saliency_overlay_{cam}.png"
        plot_saliency_strip(
            output_path=out_path,
            camera_name=cam,
            frames=frames,
            saliencies=saliencies,
            cmap=args.cmap,
            alpha=args.alpha,
            n_cols=args.n_cols,
        )
        saved_paths.append(out_path)
        print(f"Saved: {out_path}")

        if args.side_by_side:
            side_path = args.output_dir / f"saliency_overlay_{cam}_sidebyside.png"
            plot_saliency_side_by_side(
                output_path=side_path,
                camera_name=cam,
                frames=frames,
                saliencies=saliencies,
                cmap=args.cmap,
                alpha=args.alpha,
                n_cols=args.n_cols,
            )
            saved_paths.append(side_path)
            print(f"Saved: {side_path}")

    npz_path = args.output_dir / "florence2_saliency_maps.npz"
    save_dict = {}
    for cam in rgb_keys:
        save_dict[f"frames_{cam}"] = np.stack(frames, axis=0)
        save_dict[f"saliency_{cam}"] = np.stack(saliencies, axis=0)
    np.savez_compressed(npz_path, **save_dict)
    print(f"Saved raw data: {npz_path}")

    print("\nVisualization complete.")
    for p in saved_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
