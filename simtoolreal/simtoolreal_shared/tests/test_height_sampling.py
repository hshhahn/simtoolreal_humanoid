"""Physical workspace invariants for independently randomized tabletops."""

from types import SimpleNamespace
from pathlib import Path
import pytest
import torch
from simtoolreal_shared.height_sampling import (
    table_relative_goal_bounds, validate_height_reset_config,
    restore_g1_height_config,
)


def reset_cfg(**changes):
    fields = dict(
        goal_sampling_type="absolute", goal_z_relative_to_table=True, table_reset_z=0.255,
        table_reset_z_range=0.23, table_surface_z_offset=0.025,
        goal_table_clearance=0.025, target_volume_mins=(-0.30, 0.14, 0.0),
        target_volume_maxs=(0.30, 0.40, 1.45), target_volume_region_scale=1.0,
    )
    fields.update(changes)
    return SimpleNamespace(**fields)


def test_each_goal_obeys_its_own_tabletop_and_full_ceiling():
    torch.manual_seed(42)
    table_z = torch.linspace(0.025, 0.485, 20000)
    cfg = reset_cfg()
    lo, hi = table_relative_goal_bounds(
        cfg.target_volume_mins, cfg.target_volume_maxs, 1.0, table_z, 0.025, 0.025,
    )
    positions = lo + (hi - lo) * torch.rand_like(lo)
    assert (positions[:, 2] >= table_z + 0.05 - 1e-7).all()
    assert (positions[:, 2] <= 1.45).all()
    assert (positions[:, :2] >= lo[:, :2]).all()
    assert (positions[:, :2] <= hi[:, :2]).all()
    fractional_height = (positions[:, 2] - lo[:, 2]) / (hi[:, 2] - lo[:, 2])
    assert abs(float(fractional_height.mean()) - 0.5) < 0.01
    assert (fractional_height < 0.01).any() and (fractional_height > 0.99).any()


@pytest.mark.parametrize("scale", [0.5, 1.0, 1.5])
def test_workspace_scaling_cannot_move_goals_below_their_table(scale):
    table_z = torch.tensor([0.025, 0.255, 0.485])
    lo, hi = table_relative_goal_bounds(
        (-0.30, 0.14, 0.0), (0.30, 0.40, 1.45), scale, table_z, 0.025, 0.025,
    )
    assert (lo[:, 2] >= table_z + 0.05 - 1e-7).all()
    assert (hi >= lo).all()


def test_subset_bounds_use_selected_environment_heights():
    table_z = torch.tensor([0.025, 0.485, 0.125, 0.325])
    selected = torch.tensor([3, 1])
    lo, _ = table_relative_goal_bounds(
        (-0.30, 0.14, 0.0), (0.30, 0.40, 1.45), 1.0,
        table_z[selected], 0.025, 0.025,
    )
    torch.testing.assert_close(lo[:, 2], torch.tensor([0.375, 0.535]))
    torch.testing.assert_close(table_z, torch.tensor([0.025, 0.485, 0.125, 0.325]))


@pytest.mark.parametrize("changes", [
    {"target_volume_maxs": (0.30, 0.40, 0.50)},
    {"table_reset_z_range": -1.0},
    {"goal_table_clearance": -0.1},
    {"target_volume_region_scale": 0.0},
    {"table_reset_z": float("nan")},
    {"target_volume_mins": (0.30, 0.14, 0.0)},
])
def test_invalid_physical_workspace_fails_before_simulation(changes):
    with pytest.raises(ValueError):
        validate_height_reset_config(reset_cfg(**changes))


def test_legacy_reset_remains_opt_out():
    validate_height_reset_config(reset_cfg(
        goal_z_relative_to_table=False, table_reset_z=float("nan"),
    ))


def test_replay_restores_height_range_and_matching_table_asset():
    cfg = SimpleNamespace(sonic=object(), reset=reset_cfg(), assets=SimpleNamespace(table_urdf="wrong"))
    recorded = {
        "reset": {
            "goal_z_relative_to_table": True, "table_reset_z": 0.255,
            "table_reset_z_range": 0.23, "target_volume_maxs": [0.30, 0.40, 1.45],
            "goal_sampling_type": "absolute",
        },
        "assets": {"table_urdf": "height_table.urdf"},
    }
    restore_g1_height_config(cfg, recorded)
    assert cfg.reset.target_volume_maxs == (0.30, 0.40, 1.45)
    assert cfg.reset.goal_sampling_type == "absolute"
    assert cfg.assets.table_urdf == "height_table.urdf"
    restore_g1_height_config(cfg, {"reset": {}, "assets": {"table_urdf": "old.urdf"}})
    assert cfg.reset.goal_z_relative_to_table is False
    assert cfg.assets.table_urdf == "old.urdf"


@pytest.mark.parametrize("name", ["manipulation_table.urdf", "height_randomized_tabletop.urdf"])
def test_replay_rebases_bundled_table_from_old_checkout_and_preserves_custom_path(name):
    cfg = SimpleNamespace(sonic=object(), reset=reset_cfg(), assets=SimpleNamespace(table_urdf="wrong"))
    saved_table = f"/old-checkout/simtoolreal/assets/urdf/g1_wuji/{name}"
    restore_g1_height_config(cfg, {"assets": {"table_urdf": saved_table}})
    expected = Path(__file__).resolve().parents[2] / "assets" / "urdf" / "g1_wuji" / name
    assert cfg.assets.table_urdf == str(expected)
    assert expected.is_file()

    custom_table = f"/external/custom_tables/{name}"
    restore_g1_height_config(cfg, {"assets": {"table_urdf": custom_table}})
    assert cfg.assets.table_urdf == custom_table
