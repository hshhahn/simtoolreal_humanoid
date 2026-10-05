"""Pure-torch goal-pose samplers for SimToolReal."""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import quat_from_angle_axis, quat_mul, random_orientation


def _scale_workspace_bounds(
    mins: torch.Tensor, maxs: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale workspace bounds about their center."""
    center = 0.5 * (mins + maxs)
    half = 0.5 * (maxs - mins) * scale
    return center - half, center + half


def sample_absolute_goal_pose(
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
    scale: float,
    n_envs: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly sample position and orientation inside the workspace."""
    mins_t = torch.as_tensor(mins, device=device, dtype=torch.float32)
    maxs_t = torch.as_tensor(maxs, device=device, dtype=torch.float32)
    lo, hi = _scale_workspace_bounds(mins_t, maxs_t, scale)
    pos = lo + (hi - lo) * torch.rand(n_envs, 3, device=device)
    return pos, random_orientation(n_envs, device=device)


def sample_delta_goal_pose(
    prev_pos: torch.Tensor,              # (N, 3)
    prev_quat_wxyz: torch.Tensor,         # (N, 4)
    delta_distance: float,
    delta_rotation_degrees: float,
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perturb the previous goal by a bounded random walk."""
    device = prev_pos.device
    n = prev_pos.shape[0]

    mins_t = torch.as_tensor(mins, device=device, dtype=torch.float32)
    maxs_t = torch.as_tensor(maxs, device=device, dtype=torch.float32)
    lo, hi = _scale_workspace_bounds(mins_t, maxs_t, scale)

    pos_noise = (torch.rand(n, 3, device=device) * 2.0 - 1.0) * delta_distance
    new_pos = torch.clamp(prev_pos + pos_noise, lo, hi)

    axis = torch.nn.functional.normalize(torch.randn(n, 3, device=device), dim=-1)
    angle = (torch.rand(n, device=device) * 2.0 - 1.0) * delta_rotation_degrees * (
        math.pi / 180.0
    )
    dq = quat_from_angle_axis(angle, axis)
    new_quat = quat_mul(dq, prev_quat_wxyz)
    return new_pos, new_quat


__all__ = ["sample_absolute_goal_pose", "sample_delta_goal_pose"]
