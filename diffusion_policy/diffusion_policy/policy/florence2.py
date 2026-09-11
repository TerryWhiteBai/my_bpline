# This is the new florence2.py with feature cache
from typing import Dict, List, Optional, Tuple, Union
import importlib.util
import sys
import types
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from transformers import AutoModelForCausalLM


# Florence-2's modeling file conditionally imports flash_attn, but transformers'
# dynamic module loader scans the file for all imports and fails if flash_attn
# is not installed.  On machines without nvcc this lets us load the model anyway
# (the vision encoder does not use flash attention).
def _ensure_flash_attn_dummy() -> None:
    if "flash_attn" in sys.modules:
        return
    if importlib.util.find_spec("flash_attn") is not None:
        return
    spec = importlib.util.spec_from_loader("flash_attn", loader=None)
    flash_mod = types.ModuleType("flash_attn")
    flash_mod.__spec__ = spec
    flash_mod.__version__ = "0.0.0"
    sys.modules["flash_attn"] = flash_mod

    bert_padding = types.ModuleType("flash_attn.bert_padding")
    bert_padding.index_first_axis = lambda *args, **kwargs: None  # noqa: E731
    bert_padding.pad_input = lambda *args, **kwargs: None  # noqa: E731
    bert_padding.unpad_input = lambda *args, **kwargs: None  # noqa: E731
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


class Florence2VisionEncoder(nn.Module):
    """
    Vision-only observation encoder backed by the Florence-2 vision tower.

    This encoder follows the same high-level convention as the robomimic/RN18
    visual encoders used by the B-spline image policy: it takes a dictionary of
    observations (rgb images + low-dimensional states) and returns a single flat
    feature vector.  The language/generation pathway of Florence-2 is **not**
    used; only the visual tokens emitted by the vision encoder are kept.

    Args:
        shape_meta: observation shape metadata, same format as the rest of the
            codebase (``{'obs': {'camera': {'shape': [3,H,W], 'type': 'rgb'}, ...}}``).
        model_name: HuggingFace model id, e.g. ``microsoft/Florence-2-base``.
        output_dim: target feature dimension for each RGB observation.  Set to
            the per-camera feature dimension required by your B-spline policy.
        freeze_backbone: if True (default), the Florence-2 vision tower is
            frozen and only the final projection layer is trainable.
        input_range: pixel value range of incoming images.  DiffusionPolicy
            normalizes images to ``[-1, 1]`` by default, so the default is
            ``(-1, 1)``.  Use ``(0, 1)`` if your images are already in that
            range, or ``None`` to skip rescaling.
        image_size: size the vision tower expects (Florence-2 uses 768x768).
        micro_batch_size: if set, the vision backbone processes images in
            micro-batches of this size to reduce peak VRAM.  Useful when the
            GPU is small and the number of cameras/obs-steps is large.
        dtype: dtype used to load Florence-2.  Default is float32 so that the
            encoder output is directly compatible with the diffusion model.
        device: device on which to load the model.
    """

    def __init__(
        self,
        shape_meta: dict,
        model_name: str = "microsoft/Florence-2-base",
        output_dim: int = 512,
        freeze_backbone: bool = True,
        input_range: Optional[Tuple[float, float]] = (-1.0, 1.0),
        image_size: Union[int, Tuple[int, int]] = (768, 768),
        micro_batch_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.model_name = model_name
        self.output_dim = output_dim
        self.freeze_backbone = freeze_backbone
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.micro_batch_size = micro_batch_size

        # parse rgb / low-dim keys
        self.rgb_keys: List[str] = []
        self.low_dim_keys: List[str] = []
        for key, attr in shape_meta.get("obs", {}).items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                self.rgb_keys.append(key)
            elif obs_type == "low_dim":
                self.low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")
        self.rgb_keys.sort()
        self.low_dim_keys.sort()

        # input rescaling: default DP images are in [-1, 1]
        if input_range is None:
            self.input_min = 0.0
            self.input_max = 1.0
            self.input_scale = 1.0
        else:
            self.input_min = float(input_range[0])
            self.input_max = float(input_range[1])
            self.input_scale = self.input_max - self.input_min
            if abs(self.input_scale) < 1e-6:
                raise ValueError("input_range must have non-zero span")

        # ImageNet normalization used by Florence-2
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean)
        self.register_buffer("imagenet_std", std)

        # load Florence-2 but keep only the vision pathway
        if dtype is None:
            dtype = torch.float32
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.replace("torch.", ""))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if device is not None:
            self.model = self.model.to(device)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        self.hidden_dim = int(self.model.config.vision_config.projection_dim)
        # keep only the vision pathway: drop the language model to save memory
        if hasattr(self.model, "language_model"):
            del self.model.language_model

        if output_dim == self.hidden_dim:
            self.projection = nn.Identity()
        else:
            # keep projection in float32 for downstream stability
            self.projection = nn.Linear(self.hidden_dim, output_dim).float()


    @property
    def model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    def model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def projection_dtype(self) -> torch.dtype:
        if isinstance(self.projection, nn.Linear):
            return self.projection.weight.dtype
        return torch.float32

    def _rescale_to_01(self, img: torch.Tensor) -> torch.Tensor:
        if self.input_scale != 1.0 or self.input_min != 0.0:
            img = (img - self.input_min) / self.input_scale
        return img

    def _preprocess(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (N, 3, H, W), float, in the user's input_range.
        Returns:
            pixel_values: (N, 3, image_size, image_size), in Florence-2 format.
        """
        img = self._rescale_to_01(img)
        # Florence-2 expects values in [0, 1] before ImageNet normalization
        img = torch.clamp(img, 0.0, 1.0)
        # Move to the model device *before* resizing so the expensive 768x768
        # resize runs on the GPU instead of the CPU dataloading workers.
        img = img.to(device=self.model_device)
        img = TF.resize(img, self.image_size, antialias=True)
        img = (img - self.imagenet_mean) / self.imagenet_std
        return img

    def _encode_image_chunk(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Single-chunk visual token encoding."""
        pixel_values = pixel_values.to(
            dtype=self.model_dtype, device=self.model_device
        )
        if self.freeze_backbone:
            with torch.no_grad():
                image_features = self.model._encode_image(pixel_values)
        else:
            image_features = self.model._encode_image(pixel_values)
        return image_features

    def _encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Returns visual token embeddings of shape (N, L, hidden_dim).
        If ``micro_batch_size`` is set and N is larger, the forward pass is
        split into smaller chunks to reduce peak GPU memory.
        """
        if self.micro_batch_size is None or pixel_values.shape[0] <= self.micro_batch_size:
            return self._encode_image_chunk(pixel_values)

        chunks = torch.split(pixel_values, self.micro_batch_size)
        outputs = [self._encode_image_chunk(c) for c in chunks]
        return torch.cat(outputs, dim=0)

    def _compute_features(self, img: torch.Tensor) -> torch.Tensor:
        """Run the full vision encoder on a batch of images."""
        pixel_values = self._preprocess(img)
        image_features = self._encode_image(pixel_values)
        # align dtype with the projection layer before mean-pooling
        image_features = image_features.to(self.projection_dtype)
        feat = image_features.mean(dim=1)
        feat = self.projection(feat)
        return feat

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            obs_dict: mapping from observation key to tensor.  RGB tensors are
                expected as (N, 3, H, W) or (B, T, 3, H, W); low-dim tensors
                are concatenated as-is along the last dimension.
        Returns:
            (N, D) feature vector, where D = output_dim * #rgb + sum(low_dim).
        """
        features: List[torch.Tensor] = []

        for key in self.rgb_keys:
            img = obs_dict[key]
            # (B, T, C, H, W) -> (B*T, C, H, W)
            if img.dim() == 5:
                B, T, C, H, W = img.shape
                img = img.reshape(B * T, C, H, W)
            elif img.dim() != 4:
                raise ValueError(
                    f"RGB obs '{key}' must be 4D or 5D, got {img.dim()}D"
                )

            feat = self._compute_features(img)
            features.append(feat)

        for key in self.low_dim_keys:
            features.append(obs_dict[key])

        if len(features) == 1:
            out = features[0]
        else:
            # cast to the projection dtype for a consistent concat
            out_dtype = self.projection_dtype
            features = [f.to(out_dtype) for f in features]
            out = torch.cat(features, dim=-1)
        return out

    @torch.no_grad()
    def output_shape(self) -> Tuple[int, ...]:
        """Returns the feature shape for a single sample (without batch)."""
        dummy: Dict[str, torch.Tensor] = {}
        for key in self.rgb_keys:
            shape = tuple(self.shape_meta["obs"][key]["shape"])
            dummy[key] = torch.zeros(
                (1,) + shape, dtype=torch.float32, device=self.device
            )
        for key in self.low_dim_keys:
            shape = tuple(self.shape_meta["obs"][key]["shape"])
            dummy[key] = torch.zeros(
                (1,) + shape, dtype=torch.float32, device=self.device
            )
        out = self.forward(dummy)
        return out.shape[1:]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


def test():
    """Quick sanity check for the encoder output shape."""
    shape_meta = {
        "obs": {
            "wrist_image": {"shape": [3, 84, 84], "type": "rgb"},
            "arm_pos": {"shape": [3]},
            "gripper_pos": {"shape": [1]},
        }
    }
    encoder = Florence2VisionEncoder(
        shape_meta=shape_meta,
        output_dim=512,
        freeze_backbone=True,
        input_range=(-1.0, 1.0),
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
