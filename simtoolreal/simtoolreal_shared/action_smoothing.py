"""Joint-target smoothing that does not depend on a robot or simulator."""

import torch


def smooth_arm_targets(
    previous: torch.Tensor,
    requested: torch.Tensor,
    *,
    dt: float,
    speed_scale: float,
    moving_average: float,
) -> torch.Tensor:
    """Approach an absolute request with SimToolReal's arm movement limits.

    Clip the raw change to speed_scale * dt, then apply the dex-hand blend.
    Target speed is therefore bounded by moving_average * speed_scale.
    Inputs must already lie within joint limits. Configuration values are
    validated once by the environment, outside the per-step GPU path.
    """
    max_change = speed_scale * dt
    change = (requested - previous).clamp(-max_change, max_change)
    return previous + moving_average * change


def restore_g1_smoothing_config(cfg, saved_env: dict) -> None:
    """Restore recorded target dynamics, including pre-smoothing checkpoints."""
    if not hasattr(cfg, "sonic"):
        return
    cfg.sonic.smooth_right_arm_targets = saved_env.get("sonic", {}).get(
        "smooth_right_arm_targets", False
    )
    saved_action = saved_env.get("action", {})
    for name in ("arm_moving_average", "hand_moving_average", "dof_speed_scale"):
        if name in saved_action:
            setattr(cfg.action, name, saved_action[name])
