#!/usr/bin/env python3
"""Visualize Florence-2 condition vectors as spatial heatmaps overlaid on original images.

This script mimics ``visualize_bspline_condition_matrix.py`` but focuses on
*spatial* condition visualization rather than per-frame vector plots.  For a
contiguous sequence of dataset samples it:

  1. loads the pre-trained Florence-2 vision tower (no policy checkpoint needed),
  2. extracts the visual-token embeddings for each camera frame,
  3. projects the tokens through the same linear projection used by the policy,
  4. converts the per-token condition magnitudes into a coarse spatial heatmap,
  5. upsamples the heatmap to the original image size and overlays it on the
     original RGB frame,
  6. saves a per-camera strip of overlaid frames for a subset of video frames.

The heatmap therefore shows *which image regions most strongly drive the
Florence-2 condition vector* passed to the diffusion policy.
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


def infer_spatial_grid(num_tokens: int) -> Tuple[int, int]:
    """Infer a (H, W) grid shape that multiplies to ``num_tokens``.

    For Florence-2 with a 768 px input the vision tower down-samples by 32,
    yielding a 24x24 token grid (576 tokens).  This routine first checks for a
    perfect square and otherwise falls back to the most square integer factor
    pair.
    """
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


def tokens_to_heatmap(
    tokens: torch.Tensor,
    projection: torch.nn.Linear,
    mode: str = "projected_norm",
    target_condition: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert per-image visual tokens into a 2-D spatial heatmap.

    Args:
        tokens: visual tokens of shape (B, L, H).
        projection: the final linear projection from hidden_dim to output_dim.
        mode: how to aggregate per-token projected vectors:
            - ``projected_norm``: L2 norm of the projected token (default).
            - ``dot_condition``: dot product with the final condition vector.
            - ``token_norm``: L2 norm of the raw token embedding.
        target_condition: (B, output_dim) condition vector; required when
            ``mode == "dot_condition"``.

    Returns:
        A 2-D heatmap tensor of shape (B, H_grid, W_grid).
    """
    if mode == "token_norm":
        scores = tokens.norm(dim=-1)  # (B, L)
    elif mode == "projected_norm":
        proj = projection(tokens)  # (B, L, output_dim)
        scores = proj.norm(dim=-1)  # (B, L)
    elif mode == "dot_condition":
        if target_condition is None:
            raise ValueError("target_condition is required for dot_condition mode")
        proj = projection(tokens)  # (B, L, output_dim)
        # cosine-like similarity weighted by vector magnitude
        scores = (proj * target_condition.unsqueeze(1)).sum(dim=-1)  # (B, L)
        scores = scores.abs()
    else:
        raise ValueError(f"Unknown heatmap mode: {mode}")

    B, L = scores.shape
    h, w = infer_spatial_grid(L)
    heatmap = scores.reshape(B, h, w)
    return heatmap


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    cmap: str = "jet",
    alpha: float = 0.5,
    normalize: bool = True,
) -> np.ndarray:
    """Overlay a single-channel heatmap on an RGB image.

    Args:
        image: (H, W, 3) uint8 RGB image.
        heatmap: (h, w) float heatmap, will be resized to (H, W).
        cmap: matplotlib colormap name.
        alpha: blending weight for the heatmap.
        normalize: whether to min-max normalize the heatmap before coloring.

    Returns:
        (H, W, 3) uint8 overlaid image.
    """
    H, W = image.shape[:2]

    # Resize heatmap with bilinear interpolation using PyTorch (no cv2 dependency).
    heatmap_t = torch.from_numpy(heatmap.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized_t = F.interpolate(heatmap_t, size=(H, W), mode="bilinear", align_corners=False)
    resized = resized_t.squeeze(0).squeeze(0).numpy()

    if normalize:
        mn, mx = resized.min(), resized.max()
        if mx > mn:
            resized = (resized - mn) / (mx - mn)
        else:
            resized = np.zeros_like(resized)

    colored = (plt.get_cmap(cmap)(resized)[:, :, :3] * 255).astype(np.uint8)
    blended = (image.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha)
    return blended.clip(0, 255).astype(np.uint8)


def plot_camera_overlay_strip(
    output_path: Path,
    camera_name: str,
    frames: List[np.ndarray],
    heatmaps: List[np.ndarray],
    cmap: str = "jet",
    alpha: float = 0.5,
    n_cols: int = 8,
):
    """Save a grid of overlaid frames for one camera.

    Args:
        output_path: where to save the PNG.
        camera_name: title text.
        frames: list of (H, W, 3) uint8 images.
        heatmaps: list of (h, w) float heatmaps, one per frame.
        cmap: heatmap colormap.
        alpha: overlay opacity.
        n_cols: number of columns in the frame grid.
    """
    n_frames = len(frames)
    if n_frames == 0:
        return

    overlays = [overlay_heatmap_on_image(f, h, cmap=cmap, alpha=alpha) for f, h in zip(frames, heatmaps)]

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

    fig.suptitle(f"{camera_name} – Florence-2 condition heatmap overlay", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_side_by_side(
    output_path: Path,
    camera_name: str,
    frames: List[np.ndarray],
    heatmaps: List[np.ndarray],
    cmap: str = "jet",
    alpha: float = 0.5,
    n_cols: int = 8,
):
    """Save original / heatmap / overlay triplets for each selected frame."""
    n_frames = len(frames)
    if n_frames == 0:
        return

    overlays = [overlay_heatmap_on_image(f, h, cmap=cmap, alpha=alpha) for f, h in zip(frames, heatmaps)]

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
            (axes[base + n_cols], heatmaps[idx], "heatmap"),
            (axes[base + 2 * n_cols], overlays[idx], "overlay"),
        ]:
            ax.axis("off")
            ax.set_title(f"t={idx} {title}", fontsize=7)
            if title == "heatmap":
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(img)

    for ax in axes[n_frames * 3:]:
        ax.axis("off")

    fig.suptitle(f"{camera_name} – original / heatmap / overlay", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Overlay Florence-2 condition heatmaps on original video frames."
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
        default=Path("outputs/florence2_condition_heatmap_overlay"),
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
    parser.add_argument("--cmap", type=str, default="jet", help="Matplotlib colormap for the heatmap.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap overlay opacity.")
    parser.add_argument(
        "--heatmap_mode",
        type=str,
        default="projected_norm",
        choices=["projected_norm", "dot_condition", "token_norm"],
        help="How to derive per-token scores.",
    )
    parser.add_argument("--n_cols", type=int, default=8, help="Number of columns in the output grid.")
    parser.add_argument(
        "--side_by_side",
        action="store_true",
        help="Also save original/heatmap/overlay triplets.",
    )
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--normalize_heatmap",
        action="store_true",
        help="Normalize each heatmap independently before overlay.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Bimanual transport task shape metadata (matches the cache files in
    # my_dataset/transport).
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

    # Build the same dataset the training config uses.
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
        input_range=(0.0, 1.0),  # replay images are already in [0, 1]
        image_size=(args.image_size, args.image_size),
        dtype=torch.float32,
        device=device,
    )
    encoder = encoder.to(device)
    encoder.eval()
    print(f"Encoder loaded. per-camera dim={args.output_dim}, total dim={encoder.output_shape()[0]}")

    frames_per_camera: Dict[str, List[np.ndarray]] = {k: [] for k in rgb_keys}
    heatmaps_per_camera: Dict[str, List[np.ndarray]] = {k: [] for k in rgb_keys}

    for offset in range(args.n_frames):
        idx = args.start_idx + offset
        sample = dataset[idx]
        obs = sample["obs"]

        # Latest observed frame per camera, on device.
        obs_last = {k: v[-1:].to(device) for k, v in obs.items()}

        with torch.no_grad():
            for cam in rgb_keys:
                img = obs_last[cam]
                # Use the encoder's preprocessing and image encoding helpers.
                pixel_values = encoder._preprocess(img)
                tokens = encoder._encode_image(pixel_values)  # (1, L, H)

                # Per-camera condition vector before concatenation.
                cam_condition = encoder.projection(tokens.mean(dim=1))  # (1, output_dim)

                heatmap = tokens_to_heatmap(
                    tokens,
                    encoder.projection,
                    mode=args.heatmap_mode,
                    target_condition=cam_condition if args.heatmap_mode == "dot_condition" else None,
                )
                heatmap_np = heatmap[0].detach().cpu().numpy()
                if args.normalize_heatmap:
                    mn, mx = heatmap_np.min(), heatmap_np.max()
                    if mx > mn:
                        heatmap_np = (heatmap_np - mn) / (mx - mn)

                frames_per_camera[cam].append(rgb_tensor_to_uint8(obs[cam][-1]))
                heatmaps_per_camera[cam].append(heatmap_np)

        print(f"Processed frame {offset + 1}/{args.n_frames}")

    saved_paths = []
    for cam in rgb_keys:
        out_path = args.output_dir / f"heatmap_overlay_{cam}.png"
        plot_camera_overlay_strip(
            output_path=out_path,
            camera_name=cam,
            frames=frames_per_camera[cam],
            heatmaps=heatmaps_per_camera[cam],
            cmap=args.cmap,
            alpha=args.alpha,
            n_cols=args.n_cols,
        )
        saved_paths.append(out_path)
        print(f"Saved: {out_path}")

        if args.side_by_side:
            side_path = args.output_dir / f"heatmap_overlay_{cam}_sidebyside.png"
            plot_side_by_side(
                output_path=side_path,
                camera_name=cam,
                frames=frames_per_camera[cam],
                heatmaps=heatmaps_per_camera[cam],
                cmap=args.cmap,
                alpha=args.alpha,
                n_cols=args.n_cols,
            )
            saved_paths.append(side_path)
            print(f"Saved: {side_path}")

    # Save raw frames and heatmaps for downstream use.
    npz_path = args.output_dir / "florence2_condition_heatmaps.npz"
    save_dict: Dict[str, np.ndarray] = {}
    for cam in rgb_keys:
        save_dict[f"frames_{cam}"] = np.stack(frames_per_camera[cam], axis=0)
        save_dict[f"heatmap_{cam}"] = np.stack(heatmaps_per_camera[cam], axis=0)
    np.savez_compressed(npz_path, **save_dict)
    print(f"Saved raw data: {npz_path}")

    print("\nVisualization complete.")
    for p in saved_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
