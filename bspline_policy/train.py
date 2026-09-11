"""
Usage:
cd bspline_policy
python train.py --config-name=clean_bspline_policy_unet_bspline
"""

import pathlib
import sys

import torch

# RTX 4090 (Ada) optimizations
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BSPLINE_POLICY_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = BSPLINE_POLICY_DIR.parent
DIFFUSION_POLICY_DIR = REPO_ROOT / "diffusion_policy"
ROBOMIMIC_DIR = REPO_ROOT / "robomimic"

# Force the local robomimic (in REPO_ROOT) to be loaded instead of any
# previously imported or pip-installed version.
for _mod_name in list(sys.modules.keys()):
    if _mod_name == "robomimic" or _mod_name.startswith("robomimic."):
        del sys.modules[_mod_name]

for path in (REPO_ROOT, BSPLINE_POLICY_DIR, DIFFUSION_POLICY_DIR, ROBOMIMIC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hydra
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace


sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(BSPLINE_POLICY_DIR.joinpath("bspline_policy", "config")),
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
