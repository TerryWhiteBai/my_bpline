#!/usr/bin/env python3
"""Visualize intermediate spatial features from Florence-2's vision tower.

For each selected video frame the script:

  1. loads the pre-trained Florence-2 vision tower (no policy checkpoint needed),
  2. extracts the output of each DaViT stage as a spatial feature map,
  3. picks one or more channels per stage (by index or by top activation),
  4. bilinearly upsamples each channel to 76×76,
  5. overlays it with the ``jet`` colormap at 50 % opacity on the cropped
     original RGB frame,
  6. arranges the results in a grid: rows = frames, columns = layer/channel
     combinations.

This matches the style of the robomimic ResNet intermediate-feature
visualizer, but uses Florence-2's hierarchical vision transformer features.
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


def center_crop_hw(img: np.ndarray, crop_size: int = 76) -> np.ndarray:
    """Center-crop a (H, W, C) image to ``crop_size x crop_size``."""
    H, W = img.shape[:2]
    top = (H - crop_size) // 2
    left = (W - crop_size) // 2
    return img[top:top + crop_size, left:left + crop_size]


def overlay_feature_on_image(
    image: np.ndarray,
    feature: np.ndarray,
    cmap: str = "jet",
    alpha: float = 0.5,
    normalize: bool = True,
    target_size: int = 76,
) -> np.ndarray:
    """Overlay a single-channel feature map on a cropped RGB image.

    Args:
        image: (H, W, 3) uint8 RGB image, already cropped to target_size.
        feature: (h, w) float feature map.
        cmap: matplotlib colormap name.
        alpha: blending weight for the feature map.
        normalize: whether to min-max normalize the feature before coloring.
        target_size: side length to resize the feature to.

    Returns:
        (target_size, target_size, 3) uint8 overlaid image.
    """
    feature_t = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized_t = F.interpolate(
        feature_t, size=(target_size, target_size), mode="bilinear", align_corners=False
    )
    resized = resized_t.squeeze(0).squeeze(0).numpy()

    if normalize:
        mn, mx = resized.min(), resized.max()
        if mx > mn:
            resized = (resized - mn) / (mx - mn)
        else:
            resized = np.zeros_like(resized)

    colored = (plt.get_cmap(cmap)(resized)[:, :, :3] * 255).astype(np.uint8)
    blended = image.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha
    return blended.clip(0, 255).astype(np.uint8)


def make_hook_storage() -> Tuple[List[torch.Tensor], callable]:
    """Return a list and a forward hook that appends the module output to it."""
    storage: List[torch.Tensor] = []

    def hook(module, input, output):
        # output may be a tuple (tensor, size) for DaViT blocks.
        if isinstance(output, tuple):
            storage.append(output[0].detach())
        else:
            storage.append(output.detach())

    return storage, hook


def collect_stage_features(
    encoder: Florence2VisionEncoder,
    pixel_values: torch.Tensor,
) -> List[Tuple[str, torch.Tensor]]:
    """Run one image through the Florence-2 vision tower and return stage outputs.

    Returns a list of (layer_name, feature_map) where feature_map has shape
    (1, C, H, W).
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

        features = []
        for idx, storage in enumerate(storages):
            if len(storage) == 0:
                continue
            feat = storage[0]  # (1, H*W, C) or (1, C, H, W)
            if feat.dim() == 3:
                B, L, C = feat.shape
                side = int(math.isqrt(L))
                if side * side == L:
                    feat = feat.transpose(1, 2).reshape(B, C, side, side)
                else:
                    # Fallback: keep as (B, C, L) and let interpolation treat L as 1-D.
                    feat = feat.transpose(1, 2).unsqueeze(-1)
            features.append((f"stage{idx}_C{C}", feat))
        return features
    finally:
        for h in handles:
            h.remove()


def select_channels(
    feature: torch.Tensor,
    channel_mode: str,
    n_channels: int,
    frame_idx: int = 0,
) -> List[Tuple[int, torch.Tensor]]:
    """Select channels from a feature map.

    Args:
        feature: (1, C, H, W) tensor.
        channel_mode: ``first``, ``top_var``, or ``random``.
        n_channels: number of channels to pick.
        frame_idx: used as a random seed when channel_mode == ``random``.

    Returns:
        List of (channel_index, channel_map) tuples.
    """
    C = feature.shape[1]
    n_channels = min(n_channels, C)

    if channel_mode == "first":
        indices = list(range(n_channels))
    elif channel_mode == "top_var":
        # Pick channels with highest spatial variance across the current frame.
        flat = feature[0].reshape(C, -1)
        var = flat.var(dim=1)
        indices = var.argsort(descending=True)[:n_channels].tolist()
    elif channel_mode == "random":
        g = torch.Generator().manual_seed(42 + frame_idx)
        perm = torch.randperm(C, generator=g)
        indices = perm[:n_channels].tolist()
    else:
        raise ValueError(f"Unknown channel_mode: {channel_mode}")

    return [(i, feature[0, i]) for i in indices]


def plot_feature_overlay_grid(
    output_path: Path,
    camera_name: str,
    frames: List[np.ndarray],
    layer_column_features: List[Tuple[str, List[torch.Tensor]]],
    cmap: str = "jet",
    alpha: float = 0.5,
    target_size: int = 76,
):
    """Save a grid where rows are frames and columns are layer/channel combos.

    Args:
        output_path: destination PNG.
        camera_name: title text.
        frames: list of (H, W, 3) uint8 images.
        layer_column_features: list of (column_title, list_of_feature_maps_per_frame).
        cmap: matplotlib colormap.
        alpha: overlay opacity.
        target_size: crop/resize size for overlay.
    """
    n_frames = len(frames)
    n_cols = len(layer_column_features)
    if n_frames == 0 or n_cols == 0:
        return

    fig, axes = plt.subplots(n_frames, n_cols, figsize=(n_cols * 1.6, n_frames * 1.8))
    if n_frames == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_frames == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for col_idx, (col_title, feat_maps) in enumerate(layer_column_features):
        axes[0, col_idx].set_title(col_title, fontsize=8)
        for row_idx in range(n_frames):
            ax = axes[row_idx, col_idx]
            img_cropped = center_crop_hw(frames[row_idx], crop_size=target_size)
            overlay = overlay_feature_on_image(
                img_cropped, feat_maps[row_idx].cpu().numpy(),
                cmap=cmap, alpha=alpha, target_size=target_size,
            )
            ax.imshow(overlay)
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(f"t={row_idx}", fontsize=8)

    fig.suptitle(
        f"{camera_name} – Florence-2 intermediate feature overlays ({target_size}×{target_size})",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Overlay Florence-2 intermediate vision features on original frames."
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
        default=Path("outputs/florence2_intermediate_feature_overlay"),
        help="Directory to write PNGs and raw data.",
    )
    parser.add_argument("--start_idx", type=int, default=0, help="First dataset index to visualize.")
    parser.add_argument("--n_frames", type=int, default=8, help="Number of consecutive frames to visualize.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Florence-2-base",
        help="HuggingFace model id for Florence-2.",
    )
    parser.add_argument("--output_dim", type=int, default=512, help="Per-camera projection dimension.")
    parser.add_argument("--image_size", type=int, default=768, help="Florence-2 input resolution.")
    parser.add_argument("--cmap", type=str, default="jet", help="Matplotlib colormap for the overlay.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Feature map overlay opacity.")
    parser.add_argument(
        "--target_size",
        type=int,
        default=76,
        help="Size to which feature maps are upsampled and images are center-cropped.",
    )
    parser.add_argument(
        "--channel_mode",
        type=str,
        default="first",
        choices=["first", "top_var", "random"],
        help="How to select which channel(s) to visualize per layer.",
    )
    parser.add_argument("--n_channels", type=int, default=1, help="Number of channels to visualize per layer.")
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated stage indices to visualize, e.g. ``0,2,3``. Default: all stages.",
    )
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    selected_stages = None
    if args.stages is not None:
        selected_stages = [int(x.strip()) for x in args.stages.split(",") if x.strip()]

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
        stage_features_per_frame: Dict[int, List[torch.Tensor]] = {}

        for offset in range(args.n_frames):
            idx = args.start_idx + offset
            sample = dataset[idx]
            obs = sample["obs"]
            img_tensor = obs[cam][-1]  # (3, 84, 84)
            frames.append(rgb_tensor_to_uint8(img_tensor))

            obs_last = {cam: img_tensor.unsqueeze(0).to(device)}
            # Provide dummy low-dim obs so the encoder's forward is happy if it
            # expects them; we bypass it for feature extraction anyway.
            for key in shape_meta["obs"]:
                if key not in obs_last:
                    shape = tuple(shape_meta["obs"][key]["shape"])
                    obs_last[key] = torch.zeros((1,) + shape, dtype=torch.float32, device=device)

            with torch.no_grad():
                pixel_values = encoder._preprocess(obs_last[cam])
                stage_features = collect_stage_features(encoder, pixel_values)

            for stage_idx, (name, feat) in enumerate(stage_features):
                stage_features_per_frame.setdefault(stage_idx, []).append(feat)

        # Decide which stages to visualize.
        all_stage_indices = sorted(stage_features_per_frame.keys())
        if selected_stages is not None:
            stage_indices = [s for s in selected_stages if s in all_stage_indices]
        else:
            stage_indices = all_stage_indices

        # Build column list: each column is one (stage, channel) combination.
        layer_column_features: List[Tuple[str, List[torch.Tensor]]] = []
        for stage_idx in stage_indices:
            feat_per_frame = stage_features_per_frame[stage_idx]
            C = feat_per_frame[0].shape[1]
            # Pick channel indices once (based on the first frame) so columns
            # are consistent across frames.
            selected = select_channels(
                feat_per_frame[0], args.channel_mode, args.n_channels, frame_idx=0
            )
            channel_indices = [i for i, _ in selected]

            for ch_idx in channel_indices:
                col_feats = [feat_per_frame[offset][0, ch_idx] for offset in range(args.n_frames)]
                layer_column_features.append((f"stage{stage_idx}\nch{ch_idx}/{C}", col_feats))

        out_path = args.output_dir / f"intermediate_feature_overlay_{cam}.png"
        plot_feature_overlay_grid(
            output_path=out_path,
            camera_name=cam,
            frames=frames,
            layer_column_features=layer_column_features,
            cmap=args.cmap,
            alpha=args.alpha,
            target_size=args.target_size,
        )
        saved_paths.append(out_path)
        print(f"Saved: {out_path}")

    print("\nVisualization complete.")
    for p in saved_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
