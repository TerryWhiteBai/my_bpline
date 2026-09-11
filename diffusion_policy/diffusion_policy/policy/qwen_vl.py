from typing import Dict, List, Optional, Tuple, Union
import importlib
import importlib.util
import sys
import types

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForVision2Seq


# Qwen2-VL's modeling file imports flash_attn. On machines without nvcc we
# create a dummy module so the model file can load; the vision tower itself
# does not actually use flash attention.
def _ensure_flash_attn_dummy() -> None:
    if "flash_attn" in sys.modules or importlib.util.find_spec("flash_attn") is not None:
        return
    spec = importlib.util.spec_from_loader("flash_attn", loader=None)
    flash_mod = types.ModuleType("flash_attn")
    flash_mod.__spec__ = spec
    flash_mod.__version__ = "0.0.0"
    sys.modules["flash_attn"] = flash_mod

    bert_padding = types.ModuleType("flash_attn.bert_padding")
    bert_padding.index_first_axis = lambda *args, **kwargs: None
    bert_padding.pad_input = lambda *args, **kwargs: None
    bert_padding.unpad_input = lambda *args, **kwargs: None
    sys.modules["flash_attn.bert_padding"] = bert_padding


_ensure_flash_attn_dummy()


try:
    from transformers import dynamic_module_utils as _dmu

    _orig_check_imports = _dmu.check_imports

    def _check_imports_skip_flash(filename):
        imports = _dmu.get_imports(filename)
        missing = []
        for imp in imports:
            if imp == "flash_attn":
                continue
            try:
                importlib.import_module(imp)
            except ImportError:
                missing.append(imp)
        if missing:
            raise ImportError(
                "This modeling file requires the following packages that were not found in your environment: "
                f"{', '.join(missing)}. Run `pip install {' '.join(missing)}`"
            )
        return _dmu.get_relative_imports(filename)

    _dmu.check_imports = _check_imports_skip_flash
except Exception:
    pass



class QwenVLVisionEncoder(nn.Module):
    """
    Vision-only observation encoder backed by the Qwen2-VL vision tower.

    Takes a dictionary of observations (RGB images + low-dimensional states)
    and returns a single flat feature vector. The language/generation pathway
    of Qwen2-VL is **not** used; only the visual tokens from ``model.visual``
    are kept.
    """

    def __init__(
        self,
        shape_meta: dict,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        output_dim: int = 512,
        freeze_backbone: bool = True,
        input_range: Tuple[float, float] = (-1.0, 1.0),
        min_pixels: int = 56 * 56,
        max_pixels: int = 14 * 14 * 4 * 1280,
        micro_batch_size: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.output_dim = output_dim
        self.freeze_backbone = freeze_backbone
        self.input_range = input_range
        self.micro_batch_size = micro_batch_size

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.replace("torch.", ""))
        self.model_dtype = dtype
        self.model_device = device or torch.device("cuda:0")

        # parse rgb / low-dim keys
        self.rgb_keys: List[str] = []
        self.low_dim_keys: List[str] = []
        for key, attr in shape_meta["obs"].items():
            if attr.get("type", "low_dim") == "rgb":
                self.rgb_keys.append(key)
            else:
                self.low_dim_keys.append(key)

        # load image processor and model
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.image_processor.min_pixels = min_pixels
        self.image_processor.max_pixels = max_pixels

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=self.model_dtype,
        ).to(self.model_device)

        self.hidden_dim = self.model.config.hidden_size
        self.spatial_merge_size = getattr(
            self.model.config.vision_config, "spatial_merge_size", 1
        )
        self.projection = nn.Linear(self.hidden_dim, output_dim).to(self.model_device)

        if freeze_backbone:
            for param in self.model.visual.parameters():
                param.requires_grad = False
            self.model.visual.eval()

    def _preprocess(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a tensor batch of images into Qwen2-VL pixel_values/grid_thw."""
        low, high = self.input_range
        img = (img - low) / (high - low)
        img = torch.clamp(img, 0.0, 1.0)

        images = []
        for i in range(img.shape[0]):
            arr = (img[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            images.append(Image.fromarray(arr).convert("RGB"))

        out = self.image_processor(images, return_tensors="pt")
        pixel_values = out["pixel_values"].to(device=self.model_device, dtype=self.model_dtype)
        grid_thw = out["image_grid_thw"].to(device=self.model_device)
        return pixel_values, grid_thw

    def _split_and_pool(
        self, features: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Split concatenated visual tokens back to per-image vectors.

        Qwen2-VL merges spatial patches by ``spatial_merge_size`` inside the
        vision tower, so the number of output tokens per image is smaller than
        ``grid_thw.prod(dim=1)``.
        """
        tokens_per_image = (
            grid_thw[:, 0]
            * (grid_thw[:, 1] // self.spatial_merge_size)
            * (grid_thw[:, 2] // self.spatial_merge_size)
        )
        feats = []
        start = 0
        for n in tokens_per_image:
            n = int(n.item())
            feats.append(features[start : start + n].mean(dim=0))
            start += n
        return torch.stack(feats, dim=0)

    def _encode_image_chunk(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Single-chunk vision-tower encoding."""
        if self.freeze_backbone:
            with torch.no_grad():
                image_features = self.model.visual(pixel_values, grid_thw=grid_thw)
        else:
            image_features = self.model.visual(pixel_values, grid_thw=grid_thw)
        return self._split_and_pool(image_features, grid_thw)

    def _encode_image(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Encode images, optionally splitting into micro-batches for VRAM."""
        num_images = grid_thw.shape[0]
        if self.micro_batch_size is None or num_images <= self.micro_batch_size:
            return self._encode_image_chunk(pixel_values, grid_thw)

        # pixel_values are concatenated in pre-merge patch order
        pre_merge_tokens = grid_thw.prod(dim=1)
        cumsum = torch.cat(
            [torch.zeros(1, device=pre_merge_tokens.device, dtype=torch.long),
             pre_merge_tokens.cumsum(0)]
        )

        feats = []
        for start_img in range(0, num_images, self.micro_batch_size):
            end_img = min(start_img + self.micro_batch_size, num_images)
            p_start = int(cumsum[start_img].item())
            p_end = int(cumsum[end_img].item())
            feats.append(
                self._encode_image_chunk(
                    pixel_values[p_start:p_end], grid_thw[start_img:end_img]
                )
            )
        return torch.cat(feats, dim=0)

    def _compute_features(self, img: torch.Tensor) -> torch.Tensor:
        """Run the full vision encoder on a batch of images."""
        pixel_values, grid_thw = self._preprocess(img)
        image_features = self._encode_image(pixel_values, grid_thw)
        # cast to fp32 for the trainable projection and the diffusion model
        image_features = image_features.to(torch.float32)
        return self.projection(image_features)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            obs_dict: RGB tensors as (N, 3, H, W) or (B, T, 3, H, W);
                low-dim tensors concatenated as-is.
        Returns:
            (N, D) feature vector, D = output_dim * #rgb + sum(low_dim).
        """
        features: List[torch.Tensor] = []

        for key in self.rgb_keys:
            img = obs_dict[key]
            if img.dim() == 5:
                B, T, C, H, W = img.shape
                img = img.reshape(B * T, C, H, W)
            feat = self._compute_features(img)
            features.append(feat)

        for key in self.low_dim_keys:
            features.append(obs_dict[key].to(device=self.model_device))

        if len(features) == 1:
            return features[0]
        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def output_shape(self) -> Tuple[int, ...]:
        """Feature shape for a single sample (without batch)."""
        dummy: Dict[str, torch.Tensor] = {}
        for key in self.rgb_keys:
            shape = tuple(self.shape_meta["obs"][key]["shape"])
            dummy[key] = torch.zeros((1,) + shape, dtype=torch.float32, device=self.model_device)
        for key in self.low_dim_keys:
            shape = tuple(self.shape_meta["obs"][key]["shape"])
            dummy[key] = torch.zeros((1,) + shape, dtype=torch.float32, device=self.model_device)
        out = self.forward(dummy)
        return out.shape[1:]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


def test():
    shape_meta = {
        "obs": {
            "wrist_image": {"shape": [3, 84, 84], "type": "rgb"},
            "arm_pos": {"shape": [3]},
            "gripper_pos": {"shape": [1]},
        }
    }
    encoder = QwenVLVisionEncoder(
        shape_meta=shape_meta,
        output_dim=512,
        freeze_backbone=True,
    )
    batch = {
        "wrist_image": torch.rand(2, 3, 84, 84) * 2.0 - 1.0,
        "arm_pos": torch.randn(2, 3),
        "gripper_pos": torch.randn(2, 1),
    }
    out = encoder(batch)
    print("output shape:", out.shape)
    print("expected feature dim:", encoder.output_shape()[0])


if __name__ == "__main__":
    test()
