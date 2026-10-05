"""SimToolReal object task driven by frozen SONIC 1.1 and Wuji synergies."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from isaaclab.envs import DirectRLEnv

from simtoolreal_shared.physics_guard import (
    invalid_physics_mask,
    mask_invalid_reward_terms,
)
from simtoolreal_shared.sonic import (
    FSQ_LEVELS,
    FrozenSonicDecoder,
    encode_standing,
    quantize_sonic_latent,
)
from simtoolreal_shared.wuji_synergy import synergy_targets

from ..simtoolreal.simtoolreal_env import SimToolRealEnv
from ..simtoolreal.utils.action_utils import apply_wrench_dr
from ..simtoolreal.utils.logging_utils import log_step_metrics
from ..simtoolreal.utils.obs_utils import compute_intermediate_values, compute_obs_dim
from ..simtoolreal.utils.reset_utils import allocate_state_buffers, reset_env_state
from ..simtoolreal.utils.reward_utils import compute_rewards
from ..simtoolreal.utils.scene_utils import apply_physx_material_properties
from ..simtoolreal.utils.termination_utils import (
    compute_terminations,
    update_tolerance_curriculum,
)
from .env_cfg import EXTRA_OBS_SIZES, G1WujiSonicEnvCfg
from .robot import (
    ALL_JOINT_NAMES,
    BODY_JOINT_NAMES,
    RIGHT_ARM_NAMES,
    LEFT_HAND_NAMES,
    RIGHT_HAND_NAMES,
    FINGERTIP_NAMES,
    body_motor_parameters,
    build_articulation_cfg,
    collision_adjacency,
)


class G1WujiSonicEnv(SimToolRealEnv):
    cfg: G1WujiSonicEnvCfg
    palm_center_offset = (0.0, 0.0, 0.05)
    fingertip_offset = (0.0, 0.0, 0.0)
    joint_names_canonical = ALL_JOINT_NAMES
    fingertip_link_names = tuple(
        f"{side}_finger{i}_tip_link" for side in ("left", "right") for i in range(1, 6)
    )

    def __init__(
        self, cfg: G1WujiSonicEnvCfg, render_mode: str | None = None, **kwargs
    ):
        if cfg.action_space != 67:
            raise ValueError(
                "G1/Wuji policy must output 64 SONIC latents + 3 right-hand synergies"
            )
        if not math.isclose(cfg.sim.dt * cfg.decimation, 0.02, abs_tol=1e-8):
            raise ValueError("SONIC 1.1 requires a 50 Hz controller period")
        if cfg.assets.robot_fix_base or cfg.assets.robot_disable_gravity:
            raise ValueError("SONIC G1 task requires a floating base and gravity")
        if cfg.sonic.latent_mode not in ("absolute", "residual"):
            raise ValueError("sonic.latent_mode must be 'absolute' or 'residual'")
        hand_action = cfg.sonic.initial_hand_action
        if len(hand_action) != 3 or any(
            not math.isfinite(value) or not -1.0 <= value <= 1.0
            for value in hand_action
        ):
            raise ValueError("initial_hand_action must be three values in [-1, 1]")
        if not math.isfinite(cfg.sonic.policy_hand_std_init) or cfg.sonic.policy_hand_std_init <= 0:
            raise ValueError("policy_hand_std_init must be finite and positive")
        if (
            not math.isfinite(cfg.sonic.physics_velocity_limit_multiplier)
            or cfg.sonic.physics_velocity_limit_multiplier <= 1.0
        ):
            raise ValueError("physics_velocity_limit_multiplier must be finite and >1")
        if (
            not math.isfinite(cfg.sonic.physics_failure_penalty)
            or cfg.sonic.physics_failure_penalty < 0.0
        ):
            raise ValueError("physics_failure_penalty must be finite and nonnegative")
        cfg.observation_space = compute_obs_dim(
            cfg.obs.obs_list, len(ALL_JOINT_NAMES), EXTRA_OBS_SIZES
        )
        cfg.state_space = compute_obs_dim(
            cfg.obs.state_list, len(ALL_JOINT_NAMES), EXTRA_OBS_SIZES
        )
        self.robot_collision_adjacency = collision_adjacency(cfg.assets.robot_urdf)
        self.pose_viewer_robot_urdf = cfg.assets.robot_urdf
        # Skip the legacy KUKA constructor while retaining the task hooks/math.
        DirectRLEnv.__init__(self, cfg, render_mode, **kwargs)
        apply_physx_material_properties(self)
        allocate_state_buffers(
            self,
            joint_names=ALL_JOINT_NAMES,
            arm_joint_names=BODY_JOINT_NAMES,
            hand_joint_names=RIGHT_HAND_NAMES,
            palm_body_name="right_palm_link",
            fingertip_body_names=FINGERTIP_NAMES,
        )
        if self.robot.num_joints != 69:
            raise ValueError(
                f"Expected 69 physical G1/Wuji joints, got {self.robot.num_joints}"
            )
        self._body_joint_ids = self.robot.find_joints(
            list(BODY_JOINT_NAMES), preserve_order=True
        )[0]
        # SimToolReal penalizes the task arm and hand, not balance joints.
        # Keep the 29-joint body control/observation mapping independent.
        self._reward_arm_joint_ids = self.robot.find_joints(
            list(RIGHT_ARM_NAMES), preserve_order=True
        )[0]
        self._reward_hand_joint_ids = self._hand_joint_ids
        self._left_hand_joint_ids = self.robot.find_joints(
            list(LEFT_HAND_NAMES), preserve_order=True
        )[0]
        self._body_defaults = self.robot.data.default_joint_pos[
            0, self._body_joint_ids
        ].clone()
        motors = body_motor_parameters()
        self._body_action_scale = torch.tensor(
            [motors[n]["action_scale"] for n in BODY_JOINT_NAMES], device=self.device
        )
        model_dir = Path(cfg.sonic.model_dir)
        self.sonic_decoder = FrozenSonicDecoder(
            model_dir / "model_decoder.onnx", device=self.device
        )
        self._standing_latent = torch.tensor(
            encode_standing(
                model_dir / "model_encoder.onnx", self._body_defaults.cpu().numpy()
            ),
            device=self.device,
        )
        self.initial_policy_action = torch.cat(
            (
                self._standing_latent
                if cfg.sonic.latent_mode == "absolute"
                else torch.zeros_like(self._standing_latent),
                torch.tensor(hand_action, device=self.device),
            )
        )
        self._meta_actions = self.initial_policy_action.expand(
            self.num_envs, -1
        ).clone()
        self._previous_meta_actions = self._meta_actions.clone()
        self._sonic_latent = (
            quantize_sonic_latent(self._standing_latent)
            .expand(self.num_envs, -1)
            .clone()
        )
        self._decoded_body_actions = torch.zeros(self.num_envs, 29, device=self.device)
        # Histories are oldest-to-newest and grouped in deployment YAML order.
        self._sonic_history = {
            name: torch.zeros(self.num_envs, 10, dim, device=self.device)
            for name, dim in (
                ("angular_velocity", 3),
                ("joint_position", 29),
                ("joint_velocity", 29),
                ("last_action", 29),
                ("gravity", 3),
            )
        }
        self._sonic_needs_reset = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._robot_fallen = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._physics_invalid = torch.zeros_like(self._robot_fallen)
        self._physics_joint_velocity_limits = self.robot.data.joint_vel_limits.clone()
        if not torch.all(
            torch.isfinite(self._physics_joint_velocity_limits)
            & (self._physics_joint_velocity_limits > 0)
        ):
            raise ValueError("G1/Wuji requires finite positive simulator velocity limits")
        self._cur_targets[:] = self.robot.data.default_joint_pos
        self._prev_targets[:] = self._cur_targets
        print(
            f"[G1/Wuji] SONIC 1.1 frozen, FSQ={FSQ_LEVELS} levels; policy=67, robot=69, actor={cfg.observation_space}, critic={cfg.state_space}; mode={cfg.sonic.latent_mode}",
            flush=True,
        )

    def build_robot_articulation_cfg(self, usd_path: str):
        return build_articulation_cfg(usd_path, self.cfg)

    def _setup_scene(self) -> None:
        super()._setup_scene()
        # Procedural object variants require replicate_physics=False; explicitly
        # isolate the parallel robot scenes while sharing the ground plane.
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    def _current_sonic_state(self) -> dict[str, torch.Tensor]:
        robot = self.robot.data
        return {
            "angular_velocity": robot.root_ang_vel_b,
            "joint_position": robot.joint_pos[:, self._body_joint_ids]
            - self._body_defaults,
            "joint_velocity": robot.joint_vel[:, self._body_joint_ids],
            "last_action": self._decoded_body_actions,
            "gravity": robot.projected_gravity_b,
        }

    def _flush_sonic_history(self, current: dict[str, torch.Tensor]) -> None:
        ids = self._sonic_needs_reset.nonzero(as_tuple=False).flatten()
        if ids.numel():
            for name, value in current.items():
                self._sonic_history[name][ids] = value[ids].unsqueeze(1)
            self._sonic_needs_reset[ids] = False

    def _sonic_proprioception(self) -> torch.Tensor:
        """Full 930D history for the frozen decoder only."""
        return torch.cat(
            [values.flatten(1) for values in self._sonic_history.values()], dim=-1
        )

    def additional_observations(self) -> dict[str, torch.Tensor]:
        current = self._current_sonic_state()
        self._flush_sonic_history(current)
        data = self.robot.data
        return {
            "base_position": data.root_pos_w - self.scene.env_origins,
            "base_gravity": data.projected_gravity_b,
            "base_linear_velocity": data.root_lin_vel_b,
            "base_angular_velocity": data.root_ang_vel_b,
            # Read the live state after physics, rather than the history's last
            # entry, which was inserted before the previous physics step.
            "sonic_current_proprioception": torch.cat(list(current.values()), dim=-1),
            "meta_actions": self._meta_actions,
        }

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        DirectRLEnv._reset_idx(self, ids)
        root_state = self.robot.data.default_root_state[ids].clone()
        root_state[:, :3] += self.scene.env_origins[ids]
        self.robot.write_root_state_to_sim(root_state, env_ids=ids)
        reset_env_state(self, ids)
        self._decoded_body_actions[ids] = 0.0
        self._meta_actions[ids] = self.initial_policy_action
        self._previous_meta_actions[ids] = self.initial_policy_action
        self._sonic_latent[ids] = quantize_sonic_latent(self._standing_latent)
        self._sonic_needs_reset[ids] = True
        self._robot_fallen[ids] = False
        self._physics_invalid[ids] = False

    @torch.no_grad()
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, 67):
            raise ValueError(
                f"Expected ({self.num_envs}, 67) actions, got {tuple(actions.shape)}"
            )
        commands = actions.to(self.device).clamp(-1.0, 1.0)
        dr = self.cfg.domain_randomization
        if dr.use_action_delay and dr.action_delay_max > 0:
            fresh = (self.episode_length_buf == 0) & (self._successes == 0)
            self._action_queue[fresh] = commands[fresh].unsqueeze(1)
            self._action_queue = torch.roll(self._action_queue, 1, dims=1)
            self._action_queue[:, 0] = commands
            delay = torch.randint(
                self._action_queue.shape[1], (self.num_envs,), device=self.device
            )
            commands = self._action_queue[
                torch.arange(self.num_envs, device=self.device), delay
            ]
        self._previous_meta_actions.copy_(self._meta_actions)
        self._meta_actions.copy_(commands)
        current = self._current_sonic_state()
        self._flush_sonic_history(current)
        for name, value in current.items():
            history = torch.roll(self._sonic_history[name], -1, dims=1)
            history[:, -1] = value
            self._sonic_history[name] = history
        latent = commands[:, :64]
        if self.cfg.sonic.latent_mode == "residual":
            latent = self._standing_latent + self.cfg.sonic.residual_scale * latent
        # Project after composing a residual so every decoder input is on-grid.
        # Keep _meta_actions continuous for observations and PPO log probabilities.
        self._sonic_latent.copy_(quantize_sonic_latent(latent))
        self._decoded_body_actions = self.sonic_decoder(
            self._sonic_latent, self._sonic_proprioception()
        ).clamp(-20.0, 20.0)
        body_targets = (
            self._body_defaults + self._decoded_body_actions * self._body_action_scale
        )
        self._cur_targets[:, self._body_joint_ids] = body_targets.clamp(
            min=self._arm_lower, max=self._arm_upper
        )
        hand_raw = synergy_targets(commands[:, 64:], self._hand_lower, self._hand_upper)
        alpha = self.cfg.action.hand_moving_average
        self._cur_targets[:, self._hand_joint_ids] = torch.lerp(
            self._prev_targets[:, self._hand_joint_ids], hand_raw, alpha
        )
        self._cur_targets[:, self._left_hand_joint_ids] = (
            self.robot.data.default_joint_pos[:, self._left_hand_joint_ids]
        )
        self._prev_targets.copy_(self._cur_targets)
        apply_wrench_dr(self)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg.sonic
        data = self.robot.data
        self._physics_invalid = invalid_physics_mask(
            data.joint_vel,
            self._physics_joint_velocity_limits,
            (
                data.joint_pos, data.root_state_w, data.body_state_w,
                data.projected_gravity_b, self.object.data.root_state_w,
            ),
            cfg.physics_velocity_limit_multiplier,
        )
        update_tolerance_curriculum(self)
        compute_intermediate_values(self)
        # A numerically invalid transition cannot count as a task success.
        self._is_success &= ~self._physics_invalid
        self._near_goal &= ~self._physics_invalid
        terminated, truncated = compute_terminations(self)
        root_local = data.root_pos_w - self.scene.env_origins
        start_xy = torch.tensor(cfg.base_position[:2], device=self.device)
        self._robot_fallen = (root_local[:, 2] < cfg.minimum_base_height) | (
            data.projected_gravity_b[:, 2]
            > -math.cos(math.radians(cfg.maximum_tilt_degrees))
        )
        wandered = (
            torch.linalg.vector_norm(root_local[:, :2] - start_xy, dim=-1)
            > cfg.maximum_base_distance
        )
        self._termination_reasons.update(
            robot_fall=self._robot_fallen, robot_wander=wandered,
            physics_failure=self._physics_invalid,
        )
        # Physics failures are terminal, so PPO does not bootstrap their value.
        return (
            terminated | self._robot_fallen | wandered | self._physics_invalid,
            truncated & ~self._physics_invalid,
        )

    def _get_rewards(self) -> torch.Tensor:
        reward = compute_rewards(self)
        self._reward_terms["tool_task_reward"] = reward
        fall = -self.cfg.sonic.fall_penalty * self._robot_fallen.float()
        latent_rate = -self.cfg.sonic.latent_rate_penalty * (
            self._meta_actions[:, :64] - self._previous_meta_actions[:, :64]
        ).square().mean(dim=-1)
        reward = reward + fall + latent_rate
        self._reward_terms.update(
            robot_fall_penalty=fall, latent_rate_penalty=latent_rate,
            total_reward=reward,
        )
        self._reward_terms = mask_invalid_reward_terms(
            self._reward_terms, self._physics_invalid,
            self.cfg.sonic.physics_failure_penalty,
        )
        reward = self._reward_terms["total_reward"]
        self.reward_buf = reward
        log_step_metrics(self)
        velocity = self.robot.data.joint_vel
        finite_velocity = torch.isfinite(velocity)
        physics_stats = torch.stack((
            self._physics_invalid.float().sum(),
            torch.where(finite_velocity, velocity.abs(), 0.0).max(),
            (~finite_velocity).float().sum(),
        )).detach().cpu().tolist()
        self.extras.update(
            physics_failure_envs=physics_stats[0],
            physics_max_joint_speed_rad_s=physics_stats[1],
            physics_nonfinite_joint_velocities=physics_stats[2],
        )
        return reward

    def configure_agent(self, agent_cfg: dict) -> None:
        """Warm-start the task policy mean at an encoded standing token."""
        from rl_games.algos_torch.model_builder import register_network
        from .network import SonicPolicyBuilder

        register_network(
            "sonic_actor_critic", lambda **kwargs: SonicPolicyBuilder(**kwargs)
        )
        network = agent_cfg["params"]["network"]
        if network["name"] == "sonic_actor_critic":
            network["initial_mean"] = self.initial_policy_action.cpu().tolist()
            network["hand_std_init"] = self.cfg.sonic.policy_hand_std_init
            network["bounded_hand_mean"] = self.cfg.sonic.policy_bounded_hand_mean
