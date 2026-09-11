#!/usr/bin/env python3
"""Visualize Florence-2 spatial attention maps overlaid on original images.

This script extracts window-attention weights from each DaViT stage of the
Florence-2 vision tower, aggregates them into per-token importance maps, and
overlays the upsampled attention heatmap on the original RGB frames.

The aggregation is performed per stage and then averaged across stages, giving
a single "where does the model look" map for each input frame.
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
    img = img.detach().cpu().numpy()
    if img.dtype != np.uint8:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    if img.shape[0] in (1, 3):
        img = np.moveaxis(img, 0, -1)
    return img


def make_attention_hook(storage: List[torch.Tensor]) -> callable:
    """Return a hook that recomputes and stores the windowed attention matrix."""
    def hook(module, input, output):
        x = input[0]  # (B, L, C)
        size = input[1] if len(input) > 1 else output[1]
        H, W = size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # Reproduce the exact window partition from WindowAttention.forward.
        x = x.view(B, H, W, C)
        pad_r = (module.window_size - W % module.window_size) % module.window_size
        pad_b = (module.window_size - H % module.window_size) % module.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        ws = module.window_size
        x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)

        B_, N, C = x.shape
        qkv = module.qkv(x).reshape(B_, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * module.scale
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        storage.append(attn.detach())
    return hook


def attention_to_spatial_map(
    attn: torch.Tensor,
    batch_size: int,
    H: int,
    W: int,
    window_size: int,
    mode: str = "key",
) -> torch.Tensor:
    """Convert windowed attention (B*windows, heads, ws^2, ws^2) to a spatial map.

    Args:
        attn: (B * n_h * n_w, num_heads, ws*ws, ws*ws)
        batch_size: B
        H, W: spatial size before padding
        window_size: window size
        mode: ``key`` -> sum over queries (which keys are attended to);
              ``query`` -> sum over keys (which queries attend outward)

    Returns:
        (B, H, W) attention map.
    """
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    Hp = H + pad_b
    Wp = W + pad_r

    n_h = Hp // window_size
    n_w = Wp // window_size
    ws = window_size

    # Average over heads, then aggregate over queries or keys.
    attn = attn.mean(dim=1)  # (B*n_h*n_w, ws^2, ws^2)
    if mode == "key":
        scores = attn.sum(dim=1)  # (B*n_h*n_w, ws^2)  which keys are attended to
    elif mode == "query":
        scores = attn.sum(dim=2)  # (B*n_h*n_w, ws^2)  which queries attend outward
    else:
        raise ValueError(f"Unknown attention mode: {mode}")

    # Reshape to windows and reverse partition.
    scores = scores.reshape(batch_size, n_h, n_w, ws, ws)
    scores = scores.permute(0, 1, 3, 2, 4).contiguous()
    scores = scores.reshape(batch_size, Hp, Wp)

    if pad_r > 0 or pad_b > 0:
        scores = scores[:, :H, :W]
    return scores


def extract_attention_maps(
    encoder: Florence2VisionEncoder,
    pixel_values: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Extract spatial attention maps from every WindowAttention layer.

    Returns a dict mapping ``stage{i}_attn`` to a (B, H, W) tensor.
    """
    vision_tower = encoder.model.vision_tower
    window_attn_modules = []
    storages = []
    handles = []

    # Walk through SpatialBlock instances and find their WindowAttention submodules.
    for stage_idx, block in enumerate(vision_tower.blocks):
        for name, module in block.named_modules():
            if "window_attn" in name and hasattr(module, "qkv"):
                storage: List[torch.Tensor] = []
                handle = module.register_forward_hook(make_attention_hook(storage))
                window_attn_modules.append((stage_idx, module))
                storages.append(storage)
                handles.append(handle)

    # Also hook block outputs to get spatial shapes.
    block_storages = []
    block_handles = []
    for idx, block in enumerate(vision_tower.blocks):
        storage, hook = make_block_output_hook()
        block_handles.append(block.register_forward_hook(hook))
        block_storages.append(storage)

    try:
        _ = encoder.model._encode_image(pixel_values)

        # Get spatial shapes from block outputs.
        spatial_shapes = []
        for storage in block_storages:
            feat = storage[0]
            B, L, C = feat.shape
            side = int(math.isqrt(L))
            if side * side == L:
                spatial_shapes.append((side, side))
            else:
                spatial_shapes.append(infer_spatial_grid(L))

        # Build attention maps.
        attention_maps = {}
        for (stage_idx, module), storage in zip(window_attn_modules, storages):
            if len(storage) == 0:
                continue
            attn = storage[0]
            H, W = spatial_shapes[stage_idx]
            window_size = module.window_size
            amap = attention_to_spatial_map(attn, pixel_values.shape[0], H, W, window_size, mode="key")
            attention_maps[f"stage{stage_idx}_attn"] = amap

        return attention_maps
    finally:
        for h in handles + block_handles:
            h.remove()


def make_block_output_hook() -> Tuple[List[torch.Tensor], callable]:
    storage: List[torch.Tensor] = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            storage.append(output[0].detach())
        else:
            storage.append(output.detach())

    return storage, hook


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


def aggregate_attention_maps(
    attention_maps: Dict[str, torch.Tensor],
    target_size: Tuple[int, int],
    mode: str = "mean",
) -> torch.Tensor:
    """Upsample all attention maps to target_size and aggregate.

    Args:
        attention_maps: dict of (B, H, W) tensors.
        target_size: (H, W)
        mode: ``mean`` or ``max``.

    Returns:
        (B, target_H, target_W) aggregated attention map.
    """
    upsampled = []
    for amap in attention_maps.values():
        amap_t = amap.unsqueeze(1).float()
        up = F.interpolate(amap_t, size=target_size, mode="bilinear", align_corners=False)
        upsampled.append(up.squeeze(1))

    stacked = torch.stack(upsampled, dim=0)
    if mode == "mean":
        return stacked.mean(dim=0)
    elif mode == "max":
        return stacked.max(dim=0)[0]
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")


def overlay_attention(
    image: np.ndarray,
    attention: np.ndarray,
    cmap: str = "jet",
    alpha: float = 0.5,
    normalize: bool = True,
) -> np.ndarray:
    """Overlay a single-channel attention map on an RGB image."""
    H, W = image.shape[:2]
    attn_t = torch.from_numpy(attention.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized_t = F.interpolate(attn_t, size=(H, W), mode="bilinear", align_corners=False)
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


def plot_attention_strip(
    output_path: Path,
    camera_name: str,
    frames: List[np.ndarray],
    attentions: List[np.ndarray],
    cmap: str = "jet",
    alpha: float = 0.5,
    n_cols: int = 8,
):
    n_frames = len(frames)
    if n_frames == 0:
        return

    overlays = [overlay_attention(f, a, cmap=cmap, alpha=alpha) for f, a in zip(frames, attentions)]

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
        f"{camera_name} – Florence-2 attention rollout ({cmap}, α={alpha})",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attention_side_by_side(
    output_path: Path,
    camera_name: str,
    frames: List[np.ndarray],
    attentions: List[np.ndarray],
    cmap: str = "jet",
    alpha: float = 0.5,
    n_cols: int = 8,
):
    n_frames = len(frames)
    if n_frames == 0:
        return

    overlays = [overlay_attention(f, a, cmap=cmap, alpha=alpha) for f, a in zip(frames, attentions)]

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
            (axes[base + n_cols], attentions[idx], "attention"),
            (axes[base + 2 * n_cols], overlays[idx], "overlay"),
        ]:
            ax.axis("off")
            ax.set_title(f"t={idx} {title}", fontsize=7)
            if title == "attention":
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(img)

    for ax in axes[n_frames * 3:]:
        ax.axis("off")

    fig.suptitle(f"{camera_name} – original / attention / overlay", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Visualize Florence-2 window-attention maps on original frames."
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
        default=Path("outputs/florence2_attention_rollout"),
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
    parser.add_argument("--cmap", type=str, default="jet", help="Colormap for overlay.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay opacity.")
    parser.add_argument(
        "--aggregate",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="How to aggregate attention maps across DaViT stages.",
    )
    parser.add_argument(
        "--attention_mode",
        type=str,
        default="key",
        choices=["key", "query"],
        help="``key`` = which tokens are attended to; ``query`` = which tokens attend outward.",
    )
    parser.add_argument(
        "--side_by_side",
        action="store_true",
        help="Also save original/attention/overlay triplets.",
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
        attentions: List[np.ndarray] = []

        for offset in range(args.n_frames):
            idx = args.start_idx + offset
            sample = dataset[idx]
            img_tensor = sample["obs"][cam][-1]
            frames.append(rgb_tensor_to_uint8(img_tensor))

            img_batch = img_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                pixel_values = encoder._preprocess(img_batch)
                attention_maps = extract_attention_maps(encoder, pixel_values)
                H, W = img_tensor.shape[-2:]
                aggregated = aggregate_attention_maps(
                    attention_maps, (H, W), mode=args.aggregate
                )
            attentions.append(aggregated[0].cpu().numpy())
            print(f"  {cam} frame {offset + 1}/{args.n_frames}")

        out_path = args.output_dir / f"attention_rollout_{cam}.png"
        plot_attention_strip(
            output_path=out_path,
            camera_name=cam,
            frames=frames,
            attentions=attentions,
            cmap=args.cmap,
            alpha=args.alpha,
            n_cols=args.n_cols,
        )
        saved_paths.append(out_path)
        print(f"Saved: {out_path}")

        if args.side_by_side:
            side_path = args.output_dir / f"attention_rollout_{cam}_sidebyside.png"
            plot_attention_side_by_side(
                output_path=side_path,
                camera_name=cam,
                frames=frames,
                attentions=attentions,
                cmap=args.cmap,
                alpha=args.alpha,
                n_cols=args.n_cols,
            )
            saved_paths.append(side_path)
            print(f"Saved: {side_path}")

    npz_path = args.output_dir / "florence2_attention_maps.npz"
    save_dict = {}
    for cam in rgb_keys:
        save_dict[f"frames_{cam}"] = np.stack(frames, axis=0)
        save_dict[f"attention_{cam}"] = np.stack(attentions, axis=0)
    np.savez_compressed(npz_path, **save_dict)
    print(f"Saved raw data: {npz_path}")

    print("\nVisualization complete.")
    for p in saved_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
