#!/usr/bin/env python3
"""Visualize the per-camera observation condition features used by the
robomimic-backed diffusion policy after B-spline action encoding.

For a contiguous sequence of dataset samples the script:
  1. extracts the latest RGB frame per camera,
  2. computes each camera's contribution to the policy's global condition
     vector by re-running the robomimic ObservationEncoder per modality,
  3. plots an image strip + a condition-matrix heatmap for every camera,
  4. saves the B-spline action parameter matrix as an additional heatmap,
  5. saves the full concatenated global condition matrix as a reference,
  6. (optional) visualizes intermediate CNN feature maps for selected frames,
     upsampled to the cropped image size and overlaid on the input image for
     direct spatial comparison.

The resulting PNGs and an ``.npz`` with raw tensors are written to
``--output_dir``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# matplotlib uses a non-interactive backend so this can run headless.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from omegaconf import OmegaConf


# --------------------------------------------------------------------------- #
# Path setup so the script can be run directly from the repo root.
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
from diffusion_policy.common.pytorch_util import dict_apply


# --------------------------------------------------------------------------- #
# OmegaConf resolvers that are present in the training config but may not be
# registered in a standalone script.
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
        # training samples are usually float32 in [0, 1]
        img = (img * 255.0).clip(0, 255)
        img = img.astype(np.uint8)
    if img.shape[0] in (1, 3):
        img = np.moveaxis(img, 0, -1)
    return img


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
    """Load a policy (or its EMA copy) from a workspace checkpoint.

    Handles both the plain DiffusionUnetHybridImagePolicy checkpoints and the
    B-spline adapter checkpoints, even when the saved ``_target_`` was not set
    to the adapter class.
    """
    import dill

    payload = torch.load(
        str(checkpoint_path),
        pickle_module=dill,
        map_location="cpu",
    )
    cfg = payload["cfg"]

    # If this is a B-spline run, make sure we instantiate the adapter class
    # that expands the action dimension by one for the knot column.
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


def extract_per_key_features(
    obs_encoder: nn.Module,
    obs_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Run the robomimic ObservationEncoder key-by-key and return flat features.

    This mirrors ``ObservationEncoder.forward`` but keeps each modality's
    features separate so that per-camera condition matrices can be visualized.
    """
    features: Dict[str, torch.Tensor] = {}
    # ``obs_shapes`` is an OrderedDict; its order is the concatenation order.
    for key in obs_encoder.obs_shapes.keys():
        x = obs_dict[key]
        if obs_encoder.obs_randomizers[key] is not None:
            x = obs_encoder.obs_randomizers[key].forward_in(x)
        if obs_encoder.obs_nets[key] is not None:
            x = obs_encoder.obs_nets[key](x)
            if obs_encoder.activation is not None:
                x = obs_encoder.activation(x)
        if obs_encoder.obs_randomizers[key] is not None:
            x = obs_encoder.obs_randomizers[key].forward_out(x)
        # flatten to [B, D]
        x = x.reshape(x.shape[0], -1)
        features[key] = x
    return features


def compute_condition_features(
    policy,
    obs_dict_last: Dict[str, torch.Tensor],
) -> Dict[str, np.ndarray]:
    """Normalize ``obs_dict_last`` and return per-key flat features (numpy)."""
    nobs = policy.normalizer.normalize(obs_dict_last)
    with torch.no_grad():
        features = extract_per_key_features(policy.obs_encoder, nobs)
    return {k: v.detach().cpu().numpy() for k, v in features.items()}


def maybe_normalize_per_dim(matrix: np.ndarray, normalize: bool) -> np.ndarray:
    """Min-max normalize each feature dimension across time if requested."""
    if not normalize or matrix.size == 0:
        return matrix
    min_v = matrix.min(axis=0, keepdims=True)
    max_v = matrix.max(axis=0, keepdims=True)
    rng = max_v - min_v
    rng[rng == 0] = 1.0
    return (matrix - min_v) / rng


def extract_intermediate_features(
    obs_net: nn.Module,
    x: torch.Tensor,
    layer_names: List[str],
) -> Dict[str, torch.Tensor]:
    """Capture intermediate feature maps from a camera observation network.

    ``obs_net`` is expected to be a ``VisualCore`` with a
    ``backbone.nets`` Sequential (ResNet18Conv).  ``layer_names`` are
    standard ResNet stage names: conv1, bn1, relu, maxpool, layer1, layer2,
    layer3, layer4.
    """
    features: Dict[str, torch.Tensor] = {}
    name_to_idx = {
        "conv1": 0,
        "bn1": 1,
        "relu": 2,
        "maxpool": 3,
        "layer1": 4,
        "layer2": 5,
        "layer3": 6,
        "layer4": 7,
    }
    backbone = obs_net.backbone.nets
    handles = []

    def hook(name):
        def fn(module, inp, out):
            features[name] = out.detach()
        return fn

    for name in layer_names:
        idx = name_to_idx[name]
        handles.append(backbone[idx].register_forward_hook(hook(name)))

    try:
        with torch.no_grad():
            _ = obs_net(x)
    finally:
        for h in handles:
            h.remove()

    return features


def blend_activation_heatmap(
    image_uint8: np.ndarray,
    activation: np.ndarray,
    cmap: str = "jet",
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a single-channel activation map on an RGB image.

    ``image_uint8`` has shape (H, W, 3).  ``activation`` has shape (H, W).
    The activation is min-max normalized and colored with ``cmap`` before
    alpha-blending with the image.
    """
    image_uint8 = np.asarray(image_uint8)
    activation = np.asarray(activation)
    amin, amax = activation.min(), activation.max()
    norm = np.zeros_like(activation, dtype=np.float32)
    if amax > amin:
        norm = (activation - amin) / (amax - amin)

    heatmap = plt.get_cmap(cmap)(norm)[:, :, :3]  # (H, W, 3) float in [0, 1]
    img_float = image_uint8.astype(np.float32) / 255.0
    blended = img_float * (1.0 - alpha) + heatmap * alpha
    blended = (blended * 255.0).clip(0, 255).astype(np.uint8)
    return blended


# --------------------------------------------------------------------------- #
# Plotting
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

    # All frames are assumed to share the same spatial size.
    strip = np.concatenate(frames, axis=1)  # (H, W*n_frames, 3)
    plot_matrix = maybe_normalize_per_dim(cond_matrix, normalize)

    fig, (ax_img, ax_cond) = plt.subplots(
        2,
        1,
        figsize=(max(8, n_frames * 1.4), 6),
        gridspec_kw={"height_ratios": [1, 1.2]},
    )

    ax_img.imshow(strip)
    ax_img.set_title(f"{camera_name} consecutive frames", fontsize=12)
    ax_img.axis("off")

    # cond_matrix shape: (n_frames, feature_dim)
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
    """Save a heatmap of the full concatenated global condition vector."""
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

    # Horizontal separators showing where each modality starts/ends.
    for b in key_boundaries[1:-1]:
        ax.axhline(y=b - 0.5, color="white", linewidth=0.8, alpha=0.6)

    # Legend-like annotations for modality boundaries.
    mid_points = [(key_boundaries[i] + key_boundaries[i + 1]) / 2
                  for i in range(len(key_names))]
    for name, y in zip(key_names, mid_points):
        ax.text(
            n_frames + 0.5,
            y,
            name,
            color="black",
            fontsize=8,
            va="center",
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


def plot_intermediate_overlays(
    output_path: Path,
    camera_name: str,
    layer_name: str,
    images: List[np.ndarray],
    features: np.ndarray,
    max_channels: int = 8,
    alpha: float = 0.5,
):
    """Save aligned CNN feature-map overlays for direct image comparison.

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
                img, feat_up[c], cmap="jet", alpha=alpha
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
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Visualize per-camera condition features of a B-spline "
                    "diffusion policy."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to a TrainDiffusionUnetHybridWorkspace checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory where the visualization PNGs will be saved.",
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
        help="Device to run the policy on.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use the EMA model instead of the regular model.",
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
        help="Matplotlib colormap for the camera condition heatmaps.",
    )
    parser.add_argument(
        "--normalize_heatmap",
        action="store_true",
        help="Min-max normalize each feature dimension before plotting.",
    )
    parser.add_argument(
        "--intermediate_frame_indices",
        type=str,
        default="0,8,16,24",
        help="Comma-separated frame indices (relative to the sampled window) "
             "for which intermediate CNN feature maps are drawn. "
             "Set to empty to disable intermediate visualizations.",
    )
    parser.add_argument(
        "--intermediate_layers",
        type=str,
        default="layer1,layer2,layer3,layer4",
        help="Comma-separated ResNet stage names to visualize: "
             "conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4.",
    )
    parser.add_argument(
        "--intermediate_channels",
        type=int,
        default=8,
        help="Number of channels to show per intermediate layer.",
    )
    parser.add_argument(
        "--intermediate_alpha",
        type=float,
        default=0.5,
        help="Alpha blending weight for the activation heatmap overlay.",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Parse intermediate-layer settings.
    do_intermediate = bool(args.intermediate_frame_indices.strip())
    if do_intermediate:
        intermediate_indices = [
            int(x.strip())
            for x in args.intermediate_frame_indices.split(",")
            if x.strip() != ""
        ]
        intermediate_layers = [
            x.strip()
            for x in args.intermediate_layers.split(",")
            if x.strip() != ""
        ]
    else:
        intermediate_indices = []
        intermediate_layers = []

    # Load policy and config.
    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_policy_from_checkpoint(
        args.checkpoint, device=device, use_ema=args.use_ema
    )
    print(
        f"Model loaded (EMA={args.use_ema}). "
        f"obs_as_global_cond={model.obs_as_global_cond}"
    )

    # Build dataset from config.
    print("Building dataset...")
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    if args.start_idx + args.n_frames > len(dataset):
        args.n_frames = len(dataset) - args.start_idx
        print(f"Clipped n_frames to {args.n_frames} (dataset length={len(dataset)})")

    shape_meta = cfg.task.shape_meta
    if args.camera_keys is not None:
        rgb_keys = [k.strip() for k in args.camera_keys.split(",") if k.strip()]
    else:
        rgb_keys = [
            k for k, attr in shape_meta["obs"].items()
            if attr.get("type", "low_dim") == "rgb"
        ]
    print(f"Cameras to visualize: {rgb_keys}")

    # Validate intermediate frame indices.
    intermediate_indices = [i for i in intermediate_indices if 0 <= i < args.n_frames]
    if do_intermediate and not intermediate_indices:
        print("Warning: no valid intermediate frame indices; skipping intermediate visualizations.")
        do_intermediate = False
    if do_intermediate:
        print(f"Intermediate feature maps for frames: {intermediate_indices}")
        print(f"Intermediate layers: {intermediate_layers}")

    # Containers.
    frames_per_camera: Dict[str, List[np.ndarray]] = {k: [] for k in rgb_keys}
    cond_per_camera: Dict[str, List[np.ndarray]] = {k: [] for k in rgb_keys}
    cond_per_key: Dict[str, List[np.ndarray]] = {}
    action_matrices: List[np.ndarray] = []
    all_features_order: List[str] = []
    intermediate_inputs: List[Tuple[int, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = []

    # ----------------------------------------------------------------------- #
    # Iterate over consecutive dataset samples.
    # ----------------------------------------------------------------------- #
    for offset in range(args.n_frames):
        idx = args.start_idx + offset
        sample = dataset[idx]
        obs = sample["obs"]  # dict of tensors with time dim To first
        action = sample["action"]  # (n_action_steps, n_action_channels)
        action_matrices.append(action.detach().cpu().numpy())

        # Use the latest observed frame for the feature extraction.
        obs_last_raw = {k: v[-1:].to(device) for k, v in obs.items()}
        nobs_last = model.normalizer.normalize(obs_last_raw)

        # Per-camera frames for the strip (always use the latest frame).
        for cam in rgb_keys:
            frames_per_camera[cam].append(rgb_tensor_to_uint8(obs[cam][-1]))

        # Per-key condition features.
        with torch.no_grad():
            feats = extract_per_key_features(model.obs_encoder, nobs_last)
        feats = {k: v.detach().cpu().numpy() for k, v in feats.items()}
        if offset == 0:
            all_features_order = list(feats.keys())
            for k in all_features_order:
                cond_per_key[k] = []
        for k, v in feats.items():
            # v has shape (1, D); squeeze the batch dimension.
            cond_per_key[k].append(v[0])
        for cam in rgb_keys:
            cond_per_camera[cam].append(feats[cam][0])

        # Keep raw + normalized inputs for intermediate feature visualization.
        if do_intermediate and offset in intermediate_indices:
            intermediate_inputs.append((offset, obs_last_raw, nobs_last))

    # ----------------------------------------------------------------------- #
    # Plot per-camera condition matrices.
    # ----------------------------------------------------------------------- #
    saved_paths = []
    for cam in rgb_keys:
        cond_matrix = np.stack(cond_per_camera[cam], axis=0)  # (T, D)
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
    global_features = [np.stack(cond_per_key[k], axis=0) for k in all_features_order]
    global_matrix = np.concatenate(global_features, axis=1)  # (T, total_D)
    key_dims = [f.shape[1] for f in global_features]
    key_boundaries = [0] + list(np.cumsum(key_dims))
    global_out = args.output_dir / "condition_matrix_global.png"
    plot_global_condition(
        output_path=global_out,
        global_matrix=global_matrix,
        key_boundaries=key_boundaries,
        key_names=all_features_order,
        normalize=args.normalize_heatmap,
    )
    saved_paths.append(global_out)
    print(f"Saved: {global_out}")

    # ----------------------------------------------------------------------- #
    # Plot B-spline action parameter matrix across frames.
    # ----------------------------------------------------------------------- #
    action_matrix = np.stack(action_matrices, axis=0)  # (T, n_steps, n_channels)
    action_out = args.output_dir / "condition_matrix_bspline_action.png"
    plot_action_condition(action_out, action_matrix)
    saved_paths.append(action_out)
    print(f"Saved: {action_out}")

    # ----------------------------------------------------------------------- #
    # Intermediate CNN feature maps (optional).
    # ----------------------------------------------------------------------- #
    if do_intermediate:
        intermediate_dir = args.output_dir / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        # intermediate_features[cam][layer] = list of (offset, raw_cropped_img, feat)
        intermediate_features: Dict[str, Dict[str, List[Tuple[int, np.ndarray, torch.Tensor]]]] = {
            cam: {layer: [] for layer in intermediate_layers}
            for cam in rgb_keys
        }

        for offset, obs_last_raw, nobs_last in intermediate_inputs:
            for cam in rgb_keys:
                raw_cropped = obs_last_raw[cam]
                x = nobs_last[cam]
                randomizer = model.obs_encoder.obs_randomizers[cam]
                if randomizer is not None:
                    raw_cropped = randomizer.forward_in(raw_cropped)
                    x = randomizer.forward_in(x)

                feats_inter = extract_intermediate_features(
                    model.obs_encoder.obs_nets[cam],
                    x,
                    intermediate_layers,
                )
                raw_img = rgb_tensor_to_uint8(raw_cropped[0])
                for layer, feat in feats_inter.items():
                    intermediate_features[cam][layer].append((offset, raw_img, feat))

        for cam in rgb_keys:
            for layer in intermediate_layers:
                entries = intermediate_features[cam][layer]
                if not entries:
                    continue
                # Sort by offset and concatenate frames.
                entries = sorted(entries, key=lambda x: x[0])
                images = [e[1] for e in entries]
                feat_array = np.stack(
                    [e[2][0].cpu().numpy() for e in entries], axis=0
                )  # (T, C, h, w)
                out_path = intermediate_dir / f"{cam}_{layer}.png"
                plot_intermediate_overlays(
                    output_path=out_path,
                    camera_name=cam,
                    layer_name=layer,
                    images=images,
                    features=feat_array,
                    max_channels=args.intermediate_channels,
                    alpha=args.intermediate_alpha,
                )
                saved_paths.append(out_path)
                print(f"Saved: {out_path}")

    # Save raw matrices as npz for downstream analysis.
    npz_path = args.output_dir / "condition_matrices.npz"
    save_dict = {"bspline_action": action_matrix}
    for cam in rgb_keys:
        save_dict[f"obs_{cam}"] = np.stack(cond_per_camera[cam], axis=0)
        save_dict[f"frames_{cam}"] = np.stack(frames_per_camera[cam], axis=0)
    save_dict["global_condition"] = global_matrix
    save_dict["global_key_order"] = np.array(all_features_order, dtype=object)
    np.savez_compressed(npz_path, **save_dict)
    print(f"Saved raw data: {npz_path}")

    print("\nVisualization complete.")
    for p in saved_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
