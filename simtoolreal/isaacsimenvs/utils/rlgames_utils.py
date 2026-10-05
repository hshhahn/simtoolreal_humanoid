"""Glue between Isaac Lab DirectRLEnv and our vendored `./rl_games/`.

Uses `isaaclab_rl.RlGamesVecEnvWrapper` as the env-side adapter (it handles obs/
action clipping, sim↔rl device bridging, obs-group routing, gym↔gymnasium
coercion). The training algorithm/Runner/PPO/SAPG code still comes from
`./rl_games/`.
"""

from __future__ import annotations

import copy
from collections import deque

import numpy as np
import torch
from rl_games.common.algo_observer import AlgoObserver


def _flatten_dict(d: dict, prefix: str = "", separator: str = "/") -> dict:
    out = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten_dict(value, name + separator, separator))
        else:
            out[name] = value
    return out


def _is_scalar(value) -> bool:
    return isinstance(value, (float, int)) or (
        isinstance(value, torch.Tensor) and value.ndim == 0
    )


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _mean_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.float().mean().detach().cpu().item())
    if isinstance(value, np.ndarray):
        return float(value.mean())
    return float(np.mean(value))


def _value_at(value, index: int) -> float:
    if isinstance(value, torch.Tensor):
        if value.ndim > 0:
            return float(value[index].detach().cpu().item())
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return float(value[index].item() if value.ndim > 0 else value.item())
    if isinstance(value, (list, tuple)):
        return float(value[index])
    return float(value)


def _has_env_axis(value, boundary: int) -> bool:
    if isinstance(value, torch.Tensor):
        return value.ndim > 0 and value.shape[0] > boundary
    if isinstance(value, np.ndarray):
        return value.ndim > 0 and value.shape[0] > boundary
    if isinstance(value, (list, tuple)):
        return len(value) > boundary
    return False


def _slice_env_axis(value, start: int):
    if isinstance(value, tuple):
        return value[start:]
    return value[start:]


def _remove_env_boundary_from_info(infos: dict, boundary: int) -> dict:
    """Drop SAPG non-leader env stats while preserving scalar log values."""
    trimmed = {}
    for key, value in infos.items():
        if isinstance(value, dict):
            trimmed[key] = _remove_env_boundary_from_info(value, boundary)
            continue

        if not _has_env_axis(value, boundary):
            trimmed[key] = value
            continue

        if key in {"successes", "closest_keypoint_max_dist"}:
            block_size = len(value) - boundary
            if block_size > 0 and len(value) % block_size == 0:
                for i in range(len(value) // block_size):
                    trimmed[f"{key}_per_block/block_{i}"] = value[
                        i * block_size : (i + 1) * block_size
                    ]
        trimmed[key] = _slice_env_axis(value, boundary)

    return trimmed


class EnvStatsAlgoObserver(AlgoObserver):
    """Log env-provided episode stats through rl_games' summary writer."""

    def __init__(self):
        super().__init__()
        self.algo = None
        self.writer = None
        self.episode_cumulative = {}
        self.episode_cumulative_avg = {}
        self.episode_final_avg = {}
        self.direct_info = {}
        self.new_finished_episodes = False

    def after_init(self, algo):
        self.algo = algo
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices, **kwargs):
        if not isinstance(infos, dict):
            return

        ignore_env_boundary = kwargs.get("ignore_env_boundary", 0)
        if ignore_env_boundary > 0:
            infos = _remove_env_boundary_from_info(copy.deepcopy(infos), ignore_env_boundary)
            done_indices = done_indices[done_indices >= ignore_env_boundary] - ignore_env_boundary

        done_indices = done_indices.reshape(-1).detach().cpu().tolist()
        self._process_episode_cumulative(infos.get("episode_cumulative"), done_indices)
        self._process_episode_final(infos.get("episode_final"), done_indices)

        self.direct_info = {
            key: value
            for key, value in _flatten_dict(infos).items()
            if _is_scalar(value)
        }
        self._process_vector_summaries(infos, tag="successes")

    def _process_episode_cumulative(self, terms, done_indices: list[int]) -> None:
        if not terms:
            return

        for key, value in terms.items():
            if key not in self.episode_cumulative:
                self.episode_cumulative[key] = torch.zeros_like(value)
            if key not in self.episode_cumulative_avg:
                self.episode_cumulative_avg[key] = deque([], maxlen=self.algo.games_to_track)
            self.episode_cumulative[key] += value

        for done_idx in done_indices:
            self.new_finished_episodes = True
            for key in terms:
                self.episode_cumulative_avg[key].append(
                    _value_at(self.episode_cumulative[key], done_idx)
                )
                self.episode_cumulative[key][done_idx] = 0

    def _process_episode_final(self, values, done_indices: list[int]) -> None:
        if not values or not done_indices:
            return

        self.new_finished_episodes = True
        for key, value in values.items():
            if key not in self.episode_final_avg:
                self.episode_final_avg[key] = deque([], maxlen=self.algo.games_to_track)
            for done_idx in done_indices:
                self.episode_final_avg[key].append(_value_at(value, done_idx))

    def _process_vector_summaries(self, infos, *, tag: str) -> None:
        if tag not in infos:
            return
        value = infos[tag]
        self.direct_info[tag] = _mean_float(value)
        if isinstance(value, torch.Tensor):
            self.direct_info[f"{tag}_median"] = float(torch.median(value.float()).detach().cpu().item())
            self.direct_info[f"{tag}_max"] = float(value.max().detach().cpu().item())
        else:
            array = np.asarray(value)
            self.direct_info[f"{tag}_median"] = float(np.median(array))
            self.direct_info[f"{tag}_max"] = float(np.max(array))

        prefix = f"{tag}_per_block/"
        for key, value in _flatten_dict(infos).items():
            if key.startswith(prefix):
                self.direct_info[key] = _mean_float(value)

    def after_clear_stats(self):
        self.episode_cumulative_avg.clear()
        self.episode_final_avg.clear()
        self.direct_info.clear()
        self.new_finished_episodes = False

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.writer is None:
            return

        if self.new_finished_episodes:
            for key, values in self.episode_cumulative_avg.items():
                if values:
                    self.writer.add_scalar(f"episode_cumulative/{key}", np.mean(values), frame)
                    self.writer.add_scalar(f"episode_cumulative_min/{key}_min", np.min(values), frame)
                    self.writer.add_scalar(f"episode_cumulative_max/{key}_max", np.max(values), frame)
            for key, values in self.episode_final_avg.items():
                if values:
                    self.writer.add_scalar(f"episode_final/{key}", np.mean(values), frame)
            self.new_finished_episodes = False

        for key, value in self.direct_info.items():
            self.writer.add_scalar(key, _as_float(value), frame)


class MultiObserver(AlgoObserver):
    """Fan out every `AlgoObserver` callback to a list of observers.

    Local copy of `isaacgymenvs.utils.rlgames_utils.MultiObserver` — we avoid
    importing from isaacgymenvs because that package's `__init__.py` pulls in
    `isaacgym`, which is absent from `.venv_isaacsim`.
    """

    def __init__(self, observers):
        super().__init__()
        self.observers = list(observers)

    def _call_multi(self, method, *args, **kwargs):
        ret = None
        for o in self.observers:
            fn = getattr(o, method, None)
            if fn is None:
                continue
            result = fn(*args, **kwargs)
            if result is not None:
                ret = result
        return ret

    def before_init(self, base_name, config, experiment_name):
        self._call_multi("before_init", base_name, config, experiment_name)

    def after_init(self, algo):
        self._call_multi("after_init", algo)

    def process_infos(self, infos, done_indices, **kwargs):
        self._call_multi("process_infos", infos, done_indices, **kwargs)

    def after_steps(self):
        return self._call_multi("after_steps")

    def after_clear_stats(self):
        self._call_multi("after_clear_stats")

    def after_print_stats(self, frame, epoch_num, total_time):
        self._call_multi("after_print_stats", frame, epoch_num, total_time)


def register_rlgames_env(
    env,
    rl_device: str = "cuda:0",
    clip_obs: float = 1e6,
    clip_actions: float = 1e6,
    name: str = "rlgpu",
):
    """Wrap an Isaac Lab env for rl_games and register under `name` ("rlgpu").

    If the env's ``single_observation_space`` includes a ``"teacher_obs"`` key
    (set by the depth-student PegInHoleEnv), the standard RlGamesVecEnvWrapper
    drops it because its hardcoded ``_obs_groups`` only carries ``"policy"``
    into ``"obs"`` and ``"critic"`` into ``"states"``. The DAggerA2CAgent
    needs that 3rd group to label rollouts with the frozen state-MLP teacher.
    Use a small subclass to pass it through as ``rl_games_obs["teacher"]``.
    """
    from rl_games.common import env_configurations, vecenv
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    wrapper_cls = RlGamesVecEnvWrapper
    if "teacher_obs" in env.unwrapped.single_observation_space.spaces:
        class _DAggerRlGamesVecEnvWrapper(RlGamesVecEnvWrapper):
            def _process_obs(self, obs_dict):
                rl_games_obs = super()._process_obs(obs_dict)
                # Pass through the teacher actor obs (already clipped by parent).
                rl_games_obs["teacher"] = obs_dict["teacher_obs"]
                return rl_games_obs

            @property
            def teacher_obs_space(self):
                """Teacher view of the central env's obs — same as the wrapper's
                own ``observation_space``/``state_space`` plumbing, but for the
                ``teacher_obs`` key. Returns a finite-bounded gym Box derived
                from the wrapper's clip_obs (which itself came from the agent
                yaml). Build the teacher's env_info from this — never from
                ``unwrapped.single_observation_space`` (those are unbounded).
                """
                import gymnasium as gym
                spec = self.unwrapped.single_observation_space.spaces["teacher_obs"]
                return gym.spaces.Box(-self._clip_obs, self._clip_obs, spec.shape, dtype=spec.dtype)
        wrapper_cls = _DAggerRlGamesVecEnvWrapper

    wrapped = wrapper_cls(env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw),
    )
    env_configurations.register(
        name,
        {
            "vecenv_type": "IsaacRlgWrapper",
            "env_creator": lambda **kw: wrapped,
        },
    )
    return wrapped


def teacher_env_info(wrapped) -> dict:
    """Build the teacher's rl_games env_info from the central wrapper's own
    clipped spaces.

    Source of truth: the agent yaml drives ``clip_observations`` /
    ``clip_actions`` → those flow into ``RlGamesVecEnvWrapper.{observation_space,
    state_space,action_space}`` and our ``teacher_obs_space`` extension here.
    Reading from the wrapper means there's exactly one place that defines
    bounds + shapes; the teacher loader doesn't reinvent them.

    Crucially, the action_space comes from ``wrapped.action_space`` (already
    ``Box(-clip_actions, +clip_actions, ...)``) rather than from
    ``unwrapped.single_action_space`` (``Box(-inf, +inf, ...)``). The latter
    silently NaNs the teacher's actions via ``rescale_actions`` in
    ``PpoPlayerContinuous.get_action``.
    """
    teacher_view = wrapped.teacher_obs_space if hasattr(wrapped, "teacher_obs_space") else wrapped.observation_space
    state_view = wrapped.state_space if wrapped.state_space is not None else teacher_view
    return {
        "observation_space": teacher_view,
        "state_space": state_view,
        "action_space": wrapped.action_space,
        "agents": 1,
        "value_size": 1,
    }
