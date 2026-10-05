"""Three commands for a 20-joint Wuji Hand (original Wuji model).

Order: thumb opposition, thumb flexion, four-finger shared flexion.
The non-thumb second joint is abduction, so it stays at zero.
"""

from __future__ import annotations

import torch


def wuji_joint_names(side: str) -> tuple[str, ...]:
    if side not in ("left", "right"):
        raise ValueError(f"Invalid Wuji hand side: {side}")
    return tuple(
        f"{side}_finger{finger}_joint{joint}"
        for finger in range(1, 6)
        for joint in range(1, 5)
    )


def synergy_targets(
    commands: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    """Map (..., 3) normalized commands into (..., 20) bounded targets.

    -1 is open/minimum opposition; +1 is closed/maximum opposition.
    Flexion runs from neutral zero to each joint's positive URDF limit.
    This preserves neutral abduction and avoids hyperextension at 'open'.
    """
    if commands.shape[-1] != 3 or lower.shape[-1] != 20 or upper.shape[-1] != 20:
        raise ValueError("Wuji synergy expects 3 commands and 20 joint limits")
    closure = (commands.clamp(-1.0, 1.0) + 1.0) * 0.5
    neutral = torch.zeros_like(lower).clamp(min=lower, max=upper)
    targets = torch.broadcast_to(neutral, (*commands.shape[:-1], 20)).clone()
    targets[..., 0] = lower[..., 0] + closure[..., 0] * (upper[..., 0] - lower[..., 0])
    for joint in (1, 2, 3):
        targets[..., joint] = neutral[..., joint] + closure[..., 1] * (
            upper[..., joint] - neutral[..., joint]
        )
    for finger in range(1, 5):
        for joint in (0, 2, 3):
            idx = 4 * finger + joint
            targets[..., idx] = neutral[..., idx] + closure[..., 2] * (
                upper[..., idx] - neutral[..., idx]
            )
    return targets.clamp(min=lower, max=upper)
