"""GPU integration: current-frame PPO inputs, SONIC history, hands and reset."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaacsimenvs-G1-Wuji-Sonic-v0")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--smooth_right_arm", action="store_true")
    parser.add_argument(
        "--report", type=Path, default=Path("../.logs/g1-wuji-verification.json")
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    app = AppLauncher(args).app

    import gymnasium as gym
    import torch
    import isaacsimenvs
    from isaacsimenvs.tasks.g1_wuji_sonic.env_cfg import (
        BIMANUAL_OBS_SIZES, EXTRA_OBS_SIZES, G1WujiSonicEnvCfg,
    )
    from isaacsimenvs.tasks.g1_wuji_sonic.robot import (
        RIGHT_ARM_NAMES, RIGHT_HAND_NAMES, LEFT_ARM_NAMES, LEFT_HAND_NAMES,
    )
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_obs_dim
    from isaaclab_tasks.utils import load_cfg_from_registry
    from simtoolreal_shared.sonic import (
        FSQ_HALF_WIDTH,
        FSQ_LEVELS,
        FSQ_TOKEN_MAX,
        FSQ_TOKEN_MIN,
        FSQ_TOKEN_STEP,
        quantize_sonic_latent,
    )
    from simtoolreal_shared.wuji_synergy import synergy_targets

    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    if cfg.reset.fixed_start_pose is None:
        cfg.assets.num_assets_per_type = 1
    cfg.seed = 42
    if args.smooth_right_arm:
        cfg.sonic.smooth_right_arm_targets = True
        cfg.action.arm_moving_average = 0.1
        cfg.action.dof_speed_scale = 1.5
        cfg.action.hand_moving_average = 0.1
    env = gym.make(args.task, cfg=cfg)
    robot_env = env.unwrapped
    obs, _ = env.reset()
    bimanual = cfg.sonic.control_both_hands
    obs_sizes = BIMANUAL_OBS_SIZES if bimanual else EXTRA_OBS_SIZES
    assert cfg.action_space == (70 if bimanual else 67)
    assert robot_env.robot.num_joints == 69
    assert cfg.observation_space == (457 if bimanual else 432)
    assert cfg.state_space == (490 if bimanual else 454)
    assert obs["policy"].shape == (args.num_envs, cfg.observation_space)
    assert obs["critic"].shape == (args.num_envs, cfg.state_space)
    assert abs(robot_env.step_dt - 0.02) < 1e-8
    assert not robot_env.robot.is_fixed_base
    assert robot_env.scene.stage.GetPrimAtPath("/World/collisions").IsValid()
    print(f"[G1 test] reset and {cfg.action_space}-action / 69-joint contract passed", flush=True)

    assert cfg.reward.to_dict() == G1WujiSonicEnvCfg().reward.to_dict()
    assert cfg.reward.to_dict() == SimToolRealEnvCfg().reward.to_dict()
    assert cfg.sonic.latent_rate_penalty == 0.0
    reward_arm_names = [robot_env.robot.data.joint_names[i] for i in robot_env._reward_arm_joint_ids]
    reward_hand_names = [robot_env.robot.data.joint_names[i] for i in robot_env._reward_hand_joint_ids]
    assert reward_arm_names == list(RIGHT_ARM_NAMES + (LEFT_ARM_NAMES if bimanual else ()))
    assert reward_hand_names == list(RIGHT_HAND_NAMES + (LEFT_HAND_NAMES if bimanual else ()))
    assert len(robot_env._body_joint_ids) == 29

    # Exercise the real reward hook with known simulated joint velocities.
    # Non-task joints and latent changes must not change the default penalty.
    positions = robot_env.robot.data.joint_pos.clone()
    velocities = torch.full_like(positions, 0.7)
    velocities[:, robot_env._reward_arm_joint_ids] = torch.arange(
        -3, 4, device=robot_env.device, dtype=velocities.dtype
    ).repeat(2 if bimanual else 1)
    velocities[:, robot_env._reward_hand_joint_ids] = torch.linspace(
        -2.0, 2.0, len(robot_env._reward_hand_joint_ids), device=robot_env.device
    )
    robot_env.robot.write_joint_state_to_sim(positions, velocities)
    robot_env._get_dones()
    robot_env._meta_actions[:, :64] = 1.0
    robot_env._previous_meta_actions[:, :64] = -1.0
    robot_env._get_rewards()
    torch.testing.assert_close(
        robot_env._reward_terms["kuka_actions_penalty"],
        -0.03 * velocities[:, robot_env._reward_arm_joint_ids].abs().sum(dim=-1),
    )
    torch.testing.assert_close(
        robot_env._reward_terms["hand_actions_penalty"],
        -0.003 * velocities[:, robot_env._reward_hand_joint_ids].abs().sum(dim=-1),
    )
    assert (robot_env._reward_terms["latent_rate_penalty"] == 0).all()
    obs, _ = env.reset()
    print(f"[G1 test] original velocity penalties on {len(reward_arm_names)} arm and {len(reward_hand_names)} hand joints; latent penalty disabled", flush=True)
    palm_tip_names = [f"{side}_palm_link" for side in ("left", "right")] + [
        f"{side}_finger{i}_tip_link" for side in ("left", "right") for i in range(1, 6)
    ]
    hand_body_ids = robot_env.robot.find_bodies(palm_tip_names, preserve_order=True)[0]
    hand_local = robot_env.robot.data.body_pos_w[:, hand_body_ids] - robot_env.scene.env_origins[:, None, :]
    table_local = robot_env.table.data.root_pos_w - robot_env.scene.env_origins
    tabletop = table_local[:, 2] + 0.025
    min_clearance = float((hand_local[:, :, 2] - tabletop[:, None]).min())
    assert min_clearance > 0.02, f"Hand starts too low: {min_clearance} m"
    assert ((hand_local[:, :, 0] - table_local[:, None, 0]).abs() < 0.325).all()
    assert ((hand_local[:, :, 1] - table_local[:, None, 1]).abs() < 0.225).all()
    if cfg.reset.fixed_start_pose is not None:
        start_pose = torch.tensor(cfg.reset.fixed_start_pose, device=robot_env.device)
        goal_pose = torch.tensor(cfg.reset.fixed_goal_pose, device=robot_env.device)
        torch.testing.assert_close(robot_env.object.data.root_pos_w - robot_env.scene.env_origins, start_pose[:3].expand(args.num_envs, -1))
        torch.testing.assert_close(robot_env.goal_viz.data.root_pos_w - robot_env.scene.env_origins, goal_pose[:3].expand(args.num_envs, -1))
        torch.testing.assert_close(robot_env.object.data.root_quat_w, robot_env.goal_viz.data.root_quat_w)
        assert len(robot_env._object_urdf_paths) == 2
    print(f"[G1 test] both hands over tabletop; min tip/palm clearance={min_clearance:.4f} m; reward unchanged", flush=True)

    frame_slices = {}
    for group, fields in (("policy", cfg.obs.obs_list), ("critic", cfg.obs.state_list)):
        index = fields.index("sonic_current_proprioception")
        start = compute_obs_dim(fields[:index], 69, obs_sizes)
        frame_slices[group] = slice(start, start + 93)

    current_frame_observation_checks = 0

    def check_current_frame_observation(observations):
        nonlocal current_frame_observation_checks
        current = torch.cat(list(robot_env._current_sonic_state().values()), dim=-1)
        assert current.shape == (args.num_envs, 93)
        clip = cfg.obs.clamp_abs_observations
        for group, frame_slice in frame_slices.items():
            torch.testing.assert_close(
                observations[group][:, frame_slice], current.clamp(-clip, clip),
                rtol=0, atol=0,
            )
        current_frame_observation_checks += 1

    check_current_frame_observation(obs)
    # Older decoder frames must not leak into the MLP's proprioception input.
    saved_history = {name: values.clone() for name, values in robot_env._sonic_history.items()}
    original_decoder_input = robot_env._sonic_proprioception().clone()
    for values in robot_env._sonic_history.values():
        values[:, :-1] += 1.0
    assert not torch.equal(robot_env._sonic_proprioception(), original_decoder_input)
    check_current_frame_observation(robot_env._get_observations())
    for name, values in saved_history.items():
        robot_env._sonic_history[name].copy_(values)
    print("[G1 test] live 93D frame for both MLPs; older history stays decoder-only", flush=True)

    # Observe the actual decoder boundary for every call, including rollouts.
    decoder_input_checks = 0
    last_decoder_latent = torch.empty_like(robot_env._sonic_latent)

    def check_decoder_input(module, inputs):
        nonlocal decoder_input_checks
        latent, proprioception = inputs
        assert proprioception.shape == (args.num_envs, 930)
        torch.testing.assert_close(
            proprioception, robot_env._sonic_proprioception(), rtol=0, atol=0
        )
        offset = 0
        for value in robot_env._current_sonic_state().values():
            dim = value.shape[-1]
            torch.testing.assert_close(
                proprioception[:, offset + 9 * dim : offset + 10 * dim], value,
                rtol=0, atol=0,
            )
            offset += 10 * dim
        codes = latent * FSQ_HALF_WIDTH
        assert torch.all(latent >= FSQ_TOKEN_MIN) and torch.all(latent <= FSQ_TOKEN_MAX)
        torch.testing.assert_close(codes, codes.round(), rtol=0, atol=0)
        last_decoder_latent.copy_(latent)
        decoder_input_checks += 1

    decoder_hook = robot_env.sonic_decoder.register_forward_pre_hook(
        check_decoder_input
    )

    # Quantize only the body tokens in both modes; hand commands remain continuous.
    old_alpha = cfg.action.hand_moving_average
    old_mode = cfg.sonic.latent_mode
    cfg.action.hand_moving_average = 1.0
    test_action = robot_env.initial_policy_action.expand(args.num_envs, -1).clone()
    test_action[:, :64] = torch.linspace(-1.0, 1.0, 64, device=robot_env.device)
    test_action[:, 64:67] = torch.tensor([0.2, 0.4, 0.6], device=robot_env.device)
    if bimanual:
        test_action[:, 67:70] = torch.tensor([-0.7, 0.8, -0.2], device=robot_env.device)
    expected = synergy_targets(
        test_action[:, 64:67], robot_env._hand_lower, robot_env._hand_upper
    )
    for mode in ("absolute", "residual"):
        cfg.sonic.latent_mode = mode
        robot_env._pre_physics_step(test_action)
        latent = test_action[:, :64]
        if mode == "residual":
            latent = robot_env._standing_latent + cfg.sonic.residual_scale * latent
        torch.testing.assert_close(
            last_decoder_latent, quantize_sonic_latent(latent), rtol=0, atol=0
        )
        assert torch.any(last_decoder_latent != latent), (
            "Test needs off-grid predictions"
        )
        torch.testing.assert_close(robot_env._meta_actions, test_action)
        torch.testing.assert_close(
            robot_env._cur_targets[:, robot_env._hand_joint_ids], expected
        )
        left_expected = (
            synergy_targets(test_action[:, 67:70], robot_env._left_hand_lower, robot_env._left_hand_upper)
            if bimanual else robot_env.robot.data.default_joint_pos[:, robot_env._left_hand_joint_ids]
        )
        torch.testing.assert_close(
            robot_env._cur_targets[:, robot_env._left_hand_joint_ids], left_expected,
        )
    if bimanual:
        right_before = robot_env._cur_targets[:, robot_env._hand_joint_ids].clone()
        test_action[:, 67:70] = -test_action[:, 67:70]
        robot_env._pre_physics_step(test_action)
        torch.testing.assert_close(robot_env._cur_targets[:, robot_env._hand_joint_ids], right_before)
        left_changed = robot_env._cur_targets[:, robot_env._left_hand_joint_ids]
        assert not torch.allclose(left_changed, left_expected)
        torch.testing.assert_close(left_changed, synergy_targets(
            test_action[:, 67:70], robot_env._left_hand_lower, robot_env._left_hand_upper,
        ))
        # Either hand can retain the episode; both hands far away terminate it.
        assert len(robot_env._fingertip_body_ids) == 10
        saved_distances = robot_env._curr_fingertip_distances.clone()
        robot_env._curr_fingertip_distances[:, :5] = 2.0
        robot_env._curr_fingertip_distances[:, 5:] = 0.1
        assert not robot_env._hand_far_mask().any()
        robot_env._curr_fingertip_distances[:] = 2.0
        assert robot_env._hand_far_mask().all()
        robot_env._curr_fingertip_distances.copy_(saved_distances)
        print("[G1 test] independent hand commands, 10 reward fingertips and either-hand retention passed", flush=True)
    cfg.action.hand_moving_average = old_alpha
    cfg.sonic.latent_mode = old_mode
    env.reset()
    print(
        "[G1 test] native FSQ decoder inputs in both modes; continuous hand commands passed",
        flush=True,
    )

    # Check filtering at the actual motor boundary with a non-standing command.
    if robot_env._smoothed_arm_joint_ids:
        arm_ids = robot_env._smoothed_arm_joint_ids
        body_indices = robot_env._smoothed_arm_body_indices
        before = robot_env._prev_targets[:, arm_ids].clone()
        robot_env._pre_physics_step(test_action)
        target = robot_env._cur_targets[:, arm_ids]
        speed_limit = cfg.action.arm_moving_average * cfg.action.dof_speed_scale
        assert ((target - before).abs() <= speed_limit * robot_env.step_dt + 1e-6).all()
        torch.testing.assert_close(
            robot_env._applied_body_actions[:, body_indices],
            (target - robot_env._body_defaults[body_indices])
            / robot_env._body_action_scale[body_indices],
        )
        untouched = [i for i in range(29) if i not in body_indices]
        torch.testing.assert_close(
            robot_env._applied_body_actions[:, untouched],
            robot_env._decoded_body_actions[:, untouched],
            rtol=0, atol=0,
        )
        raw_arm_targets = (
            robot_env._body_defaults[body_indices]
            + robot_env._decoded_body_actions[:, body_indices]
            * robot_env._body_action_scale[body_indices]
        )
        assert torch.any((raw_arm_targets - target).abs() > 1e-4)
        torch.testing.assert_close(
            robot_env._current_sonic_state()["last_action"],
            robot_env._applied_body_actions, rtol=0, atol=0,
        )
        env.reset()
        assert not robot_env._applied_body_actions.any()
        print(f"[G1 test] {len(arm_ids)} arm rate limits, applied-action history and reset passed", flush=True)

    action = robot_env.initial_policy_action.expand(args.num_envs, -1).clone()
    falls = 0
    minimum_height = 10.0
    maximum_drift = 0.0
    maximum_arm_target_speed = 0.0
    for step in range(args.steps):
        arm_ids = robot_env._smoothed_arm_joint_ids
        before_arm = robot_env._prev_targets[:, arm_ids].clone()
        obs, reward, terminated, truncated, info = env.step(action)
        if arm_ids:
            survivors = ~(terminated | truncated)
            speeds = (
                robot_env._cur_targets[:, arm_ids] - before_arm
            ).abs()[survivors] / robot_env.step_dt
            if speeds.numel():
                maximum_arm_target_speed = max(maximum_arm_target_speed, float(speeds.max()))
                assert (speeds <= speed_limit + 1e-4).all(), step
        check_current_frame_observation(obs)
        torch.testing.assert_close(reward, info["episode_cumulative"]["total_reward"])
        component_sum = sum(
            value for name, value in info["episode_cumulative"].items()
            if name not in ("total_reward", "tool_task_reward")
        )
        torch.testing.assert_close(reward, component_sum, rtol=1e-5, atol=1e-5)
        for value in (
            *obs.values(),
            reward,
            robot_env.robot.data.joint_pos,
            robot_env.robot.data.root_state_w,
            robot_env._cur_targets,
        ):
            assert torch.isfinite(value).all(), f"Non-finite state at step {step}"
        falls += int(info["episode_final"]["done_robot_fall"].sum().item())
        local = robot_env.robot.data.root_pos_w - robot_env.scene.env_origins
        minimum_height = min(minimum_height, float(local[:, 2].min().item()))
        start_xy = torch.tensor(cfg.sonic.base_position[:2], device=robot_env.device)
        maximum_drift = max(
            maximum_drift,
            float(
                torch.linalg.vector_norm(local[:, :2] - start_xy, dim=-1).max().item()
            ),
        )
        if step % 100 == 0:
            print(
                f"[G1 test] step={step} base_z={local[:, 2].tolist()} falls={falls}",
                flush=True,
            )
    print(
        f"[G1 test] {args.steps} finite physics steps; falls={falls}, min base height={minimum_height:.3f}, max drift={maximum_drift:.3f}",
        flush=True,
    )

    # A falling floating-base robot must terminate and regain its root pose.
    ids = torch.tensor([0], device=robot_env.device)
    other_root = robot_env.robot.data.root_state_w[1:].clone()
    other_latents = robot_env._sonic_latent[1:].clone()
    root = robot_env.robot.data.root_state_w[ids].clone()
    root[:, 2] = robot_env.scene.env_origins[ids, 2] + 0.30
    robot_env.robot.write_root_state_to_sim(root, env_ids=ids)
    terminated, _ = robot_env._get_dones()
    assert terminated[0] and robot_env._termination_reasons["robot_fall"][0]
    robot_env._reset_idx(ids)
    torch.testing.assert_close(
        robot_env.robot.data.root_pos_w[0] - robot_env.scene.env_origins[0],
        torch.tensor(cfg.sonic.base_position, device=robot_env.device),
    )
    torch.testing.assert_close(robot_env.robot.data.root_state_w[1:], other_root)
    torch.testing.assert_close(
        robot_env._sonic_latent[0], quantize_sonic_latent(robot_env._standing_latent)
    )
    torch.testing.assert_close(robot_env._sonic_latent[1:], other_latents)
    check_current_frame_observation(robot_env._get_observations())
    assert not robot_env._sonic_needs_reset[0]
    print(
        "[G1 test] fall termination and isolated root/history reset passed", flush=True
    )

    # Reproduce the reported finite-but-catastrophic velocity in one environment.
    saved_other_velocities = robot_env.robot.data.joint_vel[1:].clone()
    positions = robot_env.robot.data.joint_pos[ids].clone()
    velocities = torch.zeros_like(positions)
    velocities[:, robot_env._body_joint_ids[0]] = 1e11
    robot_env.robot.write_joint_state_to_sim(positions, velocities, env_ids=ids)
    terminated, truncated = robot_env._get_dones()
    assert terminated[0] and not truncated[0]
    assert robot_env._termination_reasons["physics_failure"].tolist() == [True] + [False] * (args.num_envs - 1)
    failed_reward = robot_env._get_rewards()
    assert failed_reward[0] == -cfg.sonic.physics_failure_penalty
    assert robot_env.extras["episode_final"]["done_physics_failure"][0] == 1
    assert all(torch.isfinite(value).all() for value in robot_env._reward_terms.values())
    assert robot_env._reward_terms["kuka_actions_penalty"][0] == 0
    torch.testing.assert_close(
        robot_env.robot.data.joint_vel[1:], saved_other_velocities, rtol=0, atol=0
    )
    robot_env._reset_idx(ids)
    recovered_obs = robot_env._get_observations()
    check_current_frame_observation(recovered_obs)
    assert all(torch.isfinite(value).all() for value in recovered_obs.values())
    assert not robot_env._physics_invalid.any()
    obs, reward, terminated, truncated, _ = env.step(action)
    assert torch.isfinite(reward).all() and not robot_env._physics_invalid.any()
    check_current_frame_observation(obs)
    print("[G1 test] 1e11 rad/s fault terminated with -5 reward; isolated reset recovered", flush=True)

    report = {
        "task": args.task,
        "right_arm_target_smoothing": cfg.sonic.smooth_right_arm_targets,
        "left_arm_target_smoothing": cfg.sonic.smooth_left_arm_targets,
        "both_hands_controlled": bimanual,
        "maximum_right_arm_target_speed_rad_s": maximum_arm_target_speed,
        "hand_moving_average": cfg.action.hand_moving_average,
        "tabletop_height_m": cfg.reset.table_reset_z + 0.025,
        "minimum_initial_hand_frame_clearance_m": min_clearance,
        "both_hands_initially_over_table": True,
        "reward_matches_full_task": cfg.reward.to_dict() == G1WujiSonicEnvCfg().reward.to_dict(),
        "reward_weights_match_original": cfg.reward.to_dict() == SimToolRealEnvCfg().reward.to_dict(),
        "velocity_penalty_arm_joints": reward_arm_names,
        "velocity_penalty_hand_joints": reward_hand_names,
        "arm_velocity_penalty_coefficient": cfg.reward.kuka_actions_penalty_scale,
        "hand_velocity_penalty_coefficient": cfg.reward.hand_actions_penalty_scale,
        "latent_rate_penalty_coefficient": cfg.sonic.latent_rate_penalty,
        "sonic_version": "1.1",
        "sonic_latent_quantized": True,
        "sonic_fsq_levels": FSQ_LEVELS,
        "sonic_fsq_step": FSQ_TOKEN_STEP,
        "sonic_fsq_range": [FSQ_TOKEN_MIN, FSQ_TOKEN_MAX],
        "decoder_input_checks": decoder_input_checks,
        "latent_modes_checked": ["absolute", "residual"],
        "continuous_wuji": True,
        "action_dim": cfg.action_space,
        "reward_fingertips": len(robot_env._fingertip_body_ids),
        "independent_hand_commands_checked": bimanual,
        "physical_joints": 69,
        "policy_obs_dim": cfg.observation_space,
        "critic_obs_dim": cfg.state_space,
        "policy_proprioception_frames": 1,
        "critic_proprioception_frames": 1,
        "proprioception_frame_dim": 93,
        "decoder_proprioception_frames": 10,
        "decoder_proprioception_dim": 930,
        "current_frame_observation_checks": current_frame_observation_checks,
        "controller_hz": 50,
        "physics_hz": 200,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "standing_falls": falls,
        "minimum_base_height_m": minimum_height,
        "maximum_base_drift_m": maximum_drift,
        "finite_states": True,
        "isolated_reset": True,
        "collision_groups": True,
        "physics_failure_injection_rad_s": 1e11,
        "physics_failure_terminated": True,
        "physics_failure_reward": -cfg.sonic.physics_failure_penalty,
        "physics_failure_isolated_reset": True,
        "reward_component_accounting": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    assert falls == 0, (
        "SONIC standing reference fell; inspect controller/model integration before training"
    )
    print("[G1 test] PASS", flush=True)
    decoder_hook.remove()
    env.close()
    app.close()


if __name__ == "__main__":
    main()
