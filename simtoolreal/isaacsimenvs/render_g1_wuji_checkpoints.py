"""Capture fresh deterministic rollouts of saved MLP/LSTM policies for rendering."""

import argparse
import json
import os
from pathlib import Path
import re
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--stride", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import yaml
from rl_games.algos_torch.model_builder import ModelBuilder

import isaacsimenvs
from isaacsimenvs.tasks.g1_wuji_sonic.env_cfg import G1WujiSonicLiftEnvCfg
from isaacsimenvs.tasks.simtoolreal.pose_viewer import (
    build_pose_viewer_html, capture_pose_viewer_frame, object_urdf_for_env, table_urdf_for_env,
)


manifest = json.loads(args.manifest.read_text())
out = args.manifest.resolve().parent
cfg = G1WujiSonicLiftEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = 42
env = gym.make(manifest["task"], cfg=cfg)
inner = env.unwrapped
original_rewards = inner._get_rewards
probe = {}
frames = []
step = 0


def plain_frame(frame):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in frame.items()}


def capture_rewards():
    reward = original_rewards()
    probe.update(
        error=inner._keypoints_max_dist.clone(), near=inner._near_goal.clone(),
        success=inner._is_success.clone(), lift_event=(inner._reward_terms["lift_bonus_rew"] > 0).clone(),
        position=(inner.object.data.root_pos_w - inner.scene.env_origins).clone(),
        velocity=inner.object.data.root_lin_vel_w.clone(),
        invalid=inner._physics_invalid.clone(),
    )
    if step % args.stride == 0:
        for env_id in range(args.num_envs):
            if bool(inner._physics_invalid[env_id]):
                frame = dict(frames[env_id][-1])
            else:
                frame = plain_frame(capture_pose_viewer_frame(inner, env_id))
            frame.update(time_s=(step + 1) * inner.step_dt,
                         keypoint_error_m=float(probe["error"][env_id]),
                         lift_event=bool(probe["lift_event"][env_id]),
                         near_goal=bool(probe["near"][env_id]),
                         success=bool(probe["success"][env_id]),
                         reward=float(reward[env_id]), physics_invalid=bool(probe["invalid"][env_id]))
            frames[env_id].append(frame)
    return reward


inner._get_rewards = capture_rewards
reports = {}
for label, checkpoint in manifest["checkpoints"].items():
    agent = yaml.safe_load(Path(checkpoint["saved_config"]).read_text())["agent"]
    inner.configure_agent(agent)
    model = ModelBuilder().load(agent["params"]).build({
        "actions_num": 67, "input_shape": (432,), "num_seqs": args.num_envs,
        "value_size": 1, "normalize_value": True, "normalize_input": True,
    }).to(inner.device).eval()
    payload = torch.load(checkpoint["checkpoint"], map_location=inner.device, weights_only=False)[0]
    model.load_state_dict(payload["model"], strict=True)
    torch.manual_seed(42)
    obs, _ = env.reset(seed=42)
    rnn_states = tuple(state.to(inner.device) for state in model.get_default_rnn_state()) if model.is_rnn() else None
    frames = [[plain_frame(capture_pose_viewer_frame(inner, env_id))] for env_id in range(args.num_envs)]
    completed = lifted_completed = lift_events = goals = near_frames = invalid_steps = 0
    episode_returns = torch.zeros(args.num_envs, device=inner.device)
    episode_lifted = torch.zeros(args.num_envs, dtype=torch.bool, device=inner.device)
    completed_returns, termination_counts, trace = [], {}, []
    best_error = torch.full((args.num_envs,), float("inf"), device=inner.device)
    for step in range(args.steps):
        with torch.no_grad():
            prediction = model({"obs": obs["policy"].clamp(-10, 10), "is_train": False, "rnn_states": rnn_states})
            actions = prediction["mus"]
            if model.is_rnn():
                rnn_states = prediction["rnn_states"]
            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated | truncated
            if model.is_rnn():
                for state in rnn_states:
                    state[:, done] = 0
        assert torch.isfinite(reward).all() and torch.isfinite(obs["policy"]).all()
        episode_returns += reward
        episode_lifted |= probe["lift_event"]
        lifted_completed += int((episode_lifted & done).sum())
        completed += int(done.sum())
        completed_returns.extend(episode_returns[done].tolist())
        episode_returns[done] = 0
        episode_lifted[done] = False
        lift_events += int(probe["lift_event"].sum())
        goals += int(probe["success"].sum())
        near_frames += int(probe["near"].sum())
        invalid_steps += int(probe["invalid"].sum())
        best_error = torch.minimum(best_error, probe["error"])
        for name, flags in info["episode_final"].items():
            if name.startswith("done_"):
                termination_counts[name] = termination_counts.get(name, 0) + int((flags.bool() & done).sum())
        trace.append({"time_s": (step + 1) * inner.step_dt,
                      "object_xyz": probe["position"][0].tolist(),
                      "object_speed_m_s": float(probe["velocity"][0].norm()),
                      "keypoint_error_m": float(probe["error"][0]),
                      "lift_event": bool(probe["lift_event"][0]), "success": bool(probe["success"][0]),
                      "near_goal": bool(probe["near"][0]), "done": bool(done[0]),
                      "hand_commands": actions[0, 64:].clamp(-1, 1).tolist()})
        if step % 100 == 0:
            print(f"ROLLOUT {label} step={step} lifts={lift_events} goals={goals}", flush=True)
    directory = out / label
    directory.mkdir(exist_ok=True)
    object_text, object_path = object_urdf_for_env(inner, 0)
    table_text, table_path = table_urdf_for_env(inner, 0)
    (directory / "object.urdf").write_text(object_text)
    (directory / "table.urdf").write_text(table_text)
    replay_frames = frames[0][1:]
    html = build_pose_viewer_html(frames=replay_frames, object_urdf_text=object_text,
        table_urdf_text=table_text, object_urdf_path=object_path, table_urdf_path=table_path,
        robot_urdf_path=Path(cfg.assets.robot_urdf), timestep=inner.step_dt * args.stride)
    scene_match = re.search(r'(<script id="scene-json" type="application/json">)(.*?)(</script>)', html, re.S)
    if scene_match is None:
        raise RuntimeError("Interactive viewer has no scene payload")
    scene = json.loads(scene_match.group(2))
    for robot in scene["robots"]:
        if robot["name"] == "object":
            robot["color_override"] = [.98, .36, .045]
    html = html[:scene_match.start(2)] + json.dumps(scene, separators=(",", ":")) + html[scene_match.end(2):]
    html = html.replace("W&B Interactive Robot Viewer", f"{label.upper()} best checkpoint — iteration {checkpoint['epoch']}")
    html = html.replace("Interactive Robot Viewer</strong>", f"{label.upper()} best checkpoint · iteration {checkpoint['epoch']}</strong>")
    html = html.replace("Three.js + urdf-loader prototype for W&amp;B", "Deterministic policy rollout · Orange: tool · Green: target")
    html = html.replace("camera.position.set(0.0, -1.0, 1.03);", "camera.position.set(1.6, -2.1, 1.7);")
    html = html.replace("controls.target.set(0.0, 0.0, 0.53);", "controls.target.set(0.0, 0.25, 0.8);")
    (directory / "replay.html").write_text(html)
    (directory / "frames.json").write_text(json.dumps(replay_frames, separators=(",", ":")))
    (directory / "trace.json").write_text(json.dumps(trace, indent=2))
    report = {"architecture": label, "checkpoint_epoch": checkpoint["epoch"],
        "best_training_mean_reward": checkpoint["best_training_mean_reward"],
        "deterministic": True, "seed": 42, "envs": args.num_envs, "steps": args.steps,
        "simulated_seconds_per_env": args.steps * inner.step_dt,
        "completed_episodes": completed, "lifted_completed_episodes": lifted_completed,
        "lift_events": lift_events, "goal_success_events": goals, "near_goal_frames": near_frames,
        "physics_invalid_transitions": invalid_steps, "termination_counts": termination_counts,
        "mean_completed_episode_return": float(np.mean(completed_returns)) if completed_returns else None,
        "best_keypoint_error_m_per_env": best_error.tolist(), "effective_keypoint_tolerance_m":
        inner._current_success_tolerance * cfg.reward.keypoint_scale,
        "replay_env_id": 0, "replay_frame_stride": args.stride, "robot_urdf": cfg.assets.robot_urdf,
        "tabletop_height_m": 0.5, "frame_timestep": inner.step_dt * args.stride,
        "replay": str(directory / "replay.html"), "source_checkpoint": checkpoint["checkpoint"],
        "lstm_memory_carried_and_reset_on_episode_end": bool(model.is_rnn())}
    (directory / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    reports[label] = report
    (out / "evaluation.json").write_text(json.dumps(reports, indent=2) + "\n")
    print("EVALUATION", json.dumps(report), flush=True)
    del model, payload, rnn_states
inner._get_rewards = original_rewards
env.close()
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
