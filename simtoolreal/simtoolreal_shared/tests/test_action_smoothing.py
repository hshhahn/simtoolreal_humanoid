"""Physical invariants for the manipulation-arm target limiter."""

import pytest
import torch

from simtoolreal_shared.action_smoothing import smooth_arm_targets


@pytest.mark.parametrize("frequency", [50, 60, 120])
@pytest.mark.parametrize("blend,speed_scale", [(0.1, 1.5), (0.3, 5.0)])
def test_target_speed_limit_does_not_depend_on_controller_frequency(frequency, blend, speed_scale):
    previous = torch.zeros(4, 7, dtype=torch.float64)
    requested = torch.full_like(previous, 10.0)
    for _ in range(frequency):
        updated = smooth_arm_targets(
            previous, requested, dt=1 / frequency,
            speed_scale=speed_scale, moving_average=blend,
        )
        assert ((updated - previous).abs() <= blend * speed_scale / frequency + 1e-12).all()
        previous = updated
    torch.testing.assert_close(previous, torch.full_like(previous, blend * speed_scale))


def test_nearby_targets_converge_without_overshoot():
    previous = torch.zeros(2, 7, dtype=torch.float64)
    requested = torch.tensor([[0.01, -0.01, 0, 0.02, -0.02, 0.001, -0.001]] * 2)
    requested = requested.to(torch.float64)
    original_request = requested.clone()
    for _ in range(100):
        updated = smooth_arm_targets(
            previous, requested, dt=0.02,
            speed_scale=1.5, moving_average=0.1,
        )
        assert ((requested - updated).abs() <= (requested - previous).abs()).all()
        assert (updated >= torch.minimum(previous, requested)).all()
        assert (updated <= torch.maximum(previous, requested)).all()
        previous = updated
    torch.testing.assert_close(previous, requested, atol=1e-6, rtol=0)
    torch.testing.assert_close(requested, original_request, rtol=0, atol=0)


def test_command_reversals_respect_speed_and_joint_bounds():
    lower = torch.tensor([-0.4, -0.1, -1.2])
    upper = torch.tensor([0.1, 0.7, 1.0])
    previous = torch.zeros(8, 3)
    for sign in [1.0] * 150 + [-1.0] * 300 + [1.0] * 100:
        requested = torch.full_like(previous, sign).clamp(min=lower, max=upper)
        updated = smooth_arm_targets(
            previous, requested, dt=0.02,
            speed_scale=1.5, moving_average=0.1,
        )
        assert ((updated - previous).abs() <= 0.003 + 1e-7).all()
        assert (updated >= lower).all() and (updated <= upper).all()
        previous = updated


@pytest.mark.parametrize("enabled", [False, True])
def test_replay_restores_recorded_smoothing(enabled):
    from types import SimpleNamespace
    from simtoolreal_shared.action_smoothing import restore_g1_smoothing_config

    cfg = SimpleNamespace(
        sonic=SimpleNamespace(smooth_right_arm_targets=not enabled),
        action=SimpleNamespace(
            arm_moving_average=1.0, hand_moving_average=1.0, dof_speed_scale=5.0,
        ),
    )
    recorded = {
        "sonic": {"smooth_right_arm_targets": True} if enabled else {},
        "action": {
            "arm_moving_average": 0.1,
            "hand_moving_average": 0.1 if enabled else 0.3,
            "dof_speed_scale": 1.5,
        },
    }
    restore_g1_smoothing_config(cfg, recorded)
    assert cfg.sonic.smooth_right_arm_targets is enabled
    assert cfg.action.arm_moving_average * cfg.action.dof_speed_scale == pytest.approx(0.15)
    assert cfg.action.hand_moving_average == (0.1 if enabled else 0.3)
