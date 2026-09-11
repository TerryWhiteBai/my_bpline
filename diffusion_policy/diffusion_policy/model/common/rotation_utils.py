"""Pure-PyTorch (+ scipy fallback) rotation conversions.

Replaces the pytorch3d.transforms dependency used by RotationTransformer.
"""
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F


def _to_tensor(arr: np.ndarray, device, dtype) -> torch.Tensor:
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


# ------------------------------------------------------------------
# axis_angle <-> matrix (torch path, accurate enough for forward)
# ------------------------------------------------------------------
def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula converting axis-angle to rotation matrix."""
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)
    sin, cos = torch.sin(angle), torch.cos(angle)

    K = torch.zeros(*axis.shape[:-1], 3, 3, dtype=axis.dtype, device=axis.device)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]

    I = torch.eye(3, dtype=axis.dtype, device=axis.device).expand_as(K)
    R = I + sin.unsqueeze(-1) * K + (1.0 - cos).unsqueeze(-1) * (K @ K)

    zero_mask = angle.squeeze(-1) < 1e-6
    R = torch.where(zero_mask.unsqueeze(-1).unsqueeze(-1), I, R)
    return R


# ------------------------------------------------------------------
# rotation_6d <-> matrix (torch path, fast, used by the bimanual task)
# ------------------------------------------------------------------
def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """First two rows of rotation matrix, flattened."""
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Orthonormalise 6D representation and build rotation matrix (rows)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (a2 * b1).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    # stack as rows -> shape (..., 3, 3)
    return torch.stack((b1, b2, b3), dim=-2)


# ------------------------------------------------------------------
# matrix -> {axis_angle, quaternion, euler_angles}
# Use scipy fallback for numerical robustness (gimbal lock, angle near pi).
# ------------------------------------------------------------------
def _matrix_to_repr(matrix: torch.Tensor, out_rep: str, convention: Optional[str] = None) -> torch.Tensor:
    from scipy.spatial.transform import Rotation
    shape = matrix.shape[:-2]
    m_np = matrix.detach().cpu().numpy().reshape(-1, 3, 3)
    R = Rotation.from_matrix(m_np)

    if out_rep == "axis_angle":
        y_np = R.as_rotvec()
    elif out_rep == "quaternion":
        y_np = R.as_quat()          # (x, y, z, w)
    elif out_rep == "euler_angles":
        y_np = R.as_euler(convention.lower())
    else:
        raise ValueError(out_rep)

    y_np = y_np.reshape(*shape, -1)
    return _to_tensor(y_np, matrix.device, matrix.dtype)


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    return _matrix_to_repr(matrix, "axis_angle")


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    return _matrix_to_repr(matrix, "quaternion")


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str = "XYZ") -> torch.Tensor:
    return _matrix_to_repr(matrix, "euler_angles", convention)


# ------------------------------------------------------------------
# {axis_angle, quaternion, euler_angles} -> matrix
# axis_angle has a fast torch path above; others use scipy fallback.
# ------------------------------------------------------------------
def _repr_to_matrix(x: torch.Tensor, in_rep: str, convention: Optional[str] = None) -> torch.Tensor:
    from scipy.spatial.transform import Rotation
    shape = x.shape[:-1]
    x_np = x.detach().cpu().numpy().reshape(-1, x.shape[-1])

    if in_rep == "axis_angle":
        R = Rotation.from_rotvec(x_np)
    elif in_rep == "quaternion":
        R = Rotation.from_quat(x_np)        # (x, y, z, w)
    elif in_rep == "euler_angles":
        R = Rotation.from_euler(convention.lower(), x_np)
    else:
        raise ValueError(in_rep)

    m_np = R.as_matrix().reshape(*shape, 3, 3)
    return _to_tensor(m_np, x.device, x.dtype)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    return _repr_to_matrix(quaternions, "quaternion")


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str = "XYZ") -> torch.Tensor:
    return _repr_to_matrix(euler_angles, "euler_angles", convention)


# ------------------------------------------------------------------
# Helpers not used by RotationTransformer directly, but keep API parity.
# ------------------------------------------------------------------
def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    return matrix_to_axis_angle(quaternion_to_matrix(quaternions))


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    return matrix_to_quaternion(axis_angle_to_matrix(axis_angle))
