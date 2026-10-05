"""Detect unusable simulator states before they contaminate PPO rewards."""

from __future__ import annotations

import torch


def invalid_physics_mask(
    joint_velocity: torch.Tensor,
    velocity_limits: torch.Tensor,
    other_states: tuple[torch.Tensor, ...],
    velocity_limit_multiplier: float,
) -> torch.Tensor:
    """Reject nonfinite states or speeds far above the simulator motor limits."""
    invalid = (~torch.isfinite(joint_velocity)).any(dim=-1)
    invalid |= (
        joint_velocity.abs() > velocity_limit_multiplier * velocity_limits
    ).any(dim=-1)
    for state in other_states:
        invalid |= (~torch.isfinite(state.flatten(1))).any(dim=-1)
    return invalid


def mask_invalid_reward_terms(
    terms: dict[str, torch.Tensor], invalid: torch.Tensor, penalty: float
) -> dict[str, torch.Tensor]:
    """Replace only invalid transitions with a finite terminal failure penalty.

    Valid transitions retain their exact reward values. Invalid task components
    are zeroed, so neither NaNs nor extremely large velocities enter summaries.
    """
    masked = {
        name: torch.where(invalid, torch.zeros_like(value), value)
        for name, value in terms.items()
    }
    masked["physics_failure_penalty"] = -penalty * invalid.float()
    masked["total_reward"] = torch.where(
        invalid, masked["physics_failure_penalty"], terms["total_reward"]
    )
    return masked
