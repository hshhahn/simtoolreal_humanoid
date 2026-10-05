"""Helpers for interpreting results returned by the rl_games runner."""

from omegaconf import DictConfig


def is_pbt_restart_result(result) -> bool:
    """Return whether ``result`` is the config/environment pair from PBT.

    A normal finite training run also returns a two-tuple—mean reward and
    epoch—which must not be treated as a restart request.
    """

    return (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[0], DictConfig)
        and callable(getattr(result[1], "change_on_restart", None))
    )
