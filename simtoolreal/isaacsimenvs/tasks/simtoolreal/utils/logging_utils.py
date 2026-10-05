"""Task metric publishing for SimToolReal."""

from __future__ import annotations

import torch


def log_step_metrics(env) -> None:
    """Publish step-level extras consumed by RL-Games observers."""
    term_cfg = env.cfg.termination
    if term_cfg.max_consecutive_successes > 0:
        all_goals_hit = env._successes >= term_cfg.max_consecutive_successes
    else:
        all_goals_hit = torch.zeros_like(env._successes, dtype=torch.bool)

    episode_final = {
        "successes": env._successes.float(),
        "all_goals_hit": all_goals_hit.float(),
    }
    episode_final.update(
        {
            f"done_{name}": value.float()
            for name, value in env._termination_reasons.items()
        }
    )

    env.extras["episode_cumulative"] = env._reward_terms
    env.extras["episode_final"] = episode_final
    env.extras["successes"] = env._prev_episode_successes.float()
    env.extras["current_success_tolerance"] = float(env._current_success_tolerance)


__all__ = ["log_step_metrics"]
