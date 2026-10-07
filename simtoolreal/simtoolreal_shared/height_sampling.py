"""Table-relative height bounds and replay configuration for G1 experiments."""

from __future__ import annotations

import math
from pathlib import Path
import torch


def table_relative_goal_bounds(
    mins, maxs, scale: float, table_root_z: torch.Tensor,
    surface_offset: float, clearance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Intersect the scaled workspace with each environment's tabletop floor.

    Positions use environment-local coordinates. The table buffer stores the
    rigid root, so the upper surface offset must be added explicitly.
    Configuration is checked once by validate_height_reset_config before launch.
    """
    mins_t = torch.as_tensor(mins, device=table_root_z.device, dtype=torch.float32)
    maxs_t = torch.as_tensor(maxs, device=table_root_z.device, dtype=torch.float32)
    center = (mins_t + maxs_t) * 0.5
    half = (maxs_t - mins_t) * (scale * 0.5)
    lo = (center - half).expand(table_root_z.numel(), 3).clone()
    hi = (center + half).expand_as(lo)
    lo[:, 2] = torch.maximum(lo[:, 2], table_root_z + surface_offset + clearance)
    return lo, hi


def validate_height_reset_config(cfg) -> None:
    """Reject invalid height volumes before constructing a simulation."""
    if not cfg.goal_z_relative_to_table:
        return
    values = (
        *cfg.target_volume_mins, *cfg.target_volume_maxs,
        cfg.target_volume_region_scale, cfg.table_reset_z,
        cfg.table_reset_z_range, cfg.table_surface_z_offset,
        cfg.goal_table_clearance,
    )
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Height-reset bounds must be finite")
    if cfg.table_reset_z_range < 0 or cfg.goal_table_clearance < 0:
        raise ValueError("Table range and goal clearance must be nonnegative")
    if cfg.target_volume_region_scale <= 0:
        raise ValueError("Target-volume scale must be positive")
    if any(a >= b for a, b in zip(cfg.target_volume_mins, cfg.target_volume_maxs)):
        raise ValueError("Every target-volume minimum must be below its maximum")
    min_z, max_z = cfg.target_volume_mins[2], cfg.target_volume_maxs[2]
    scaled_ceiling = (min_z + max_z) / 2 + (max_z - min_z) / 2 * cfg.target_volume_region_scale
    highest_floor = (
        cfg.table_reset_z + cfg.table_reset_z_range
        + cfg.table_surface_z_offset + cfg.goal_table_clearance
    )
    if highest_floor >= scaled_ceiling:
        raise ValueError("Highest tabletop plus clearance must be below the target ceiling")


def restore_g1_height_config(cfg, saved_environment: dict) -> None:
    """Restore recorded table/object/goal heights when rendering a G1 policy."""
    if not hasattr(cfg, "sonic"):
        return
    recorded = saved_environment.get("reset", {})
    for name in (
        "table_reset_z", "table_reset_z_range", "table_surface_z_offset",
        "table_object_z_offset", "table_reset_center_xy",
        "table_reset_xy_range_m", "table_reset_yaw_range_deg",
        "reset_position_center_xy", "reset_position_noise_x",
        "reset_position_noise_y", "reset_position_noise_z",
        "goal_z_relative_to_table", "goal_table_clearance",
        "goal_sampling_type", "target_volume_mins", "target_volume_maxs",
        "target_volume_region_scale",
    ):
        if name in recorded:
            value = recorded[name]
            if isinstance(getattr(cfg.reset, name), tuple):
                value = tuple(value)
            setattr(cfg.reset, name, value)
    # Old saved configurations predate the opt-in table-relative goal flag.
    cfg.reset.goal_z_relative_to_table = bool(recorded.get("goal_z_relative_to_table", False))
    table_urdf = saved_environment.get("assets", {}).get("table_urdf")
    if table_urdf:
        recorded_path = Path(table_urdf)
        bundled_tables = {"manipulation_table.urdf", "height_randomized_tabletop.urdf"}
        if (
            recorded_path.name in bundled_tables
            and recorded_path.parts[-4:-1] == ("assets", "urdf", "g1_wuji")
        ):
            # Saved configs name the training checkout. Replay bundled geometry
            # from this checkout even when the original checkout still exists.
            table_urdf = str(
                Path(__file__).resolve().parents[1]
                / "assets" / "urdf" / "g1_wuji" / recorded_path.name
            )
        cfg.assets.table_urdf = table_urdf
