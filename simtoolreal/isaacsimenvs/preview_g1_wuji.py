"""Render the current G1/Wuji task and save an interactive standing replay.

The scene and rewards use the training configuration. Only the environment
count, camera and display colors change for the preview. SONIC receives its
encoded standing token; this does not demonstrate a trained grasping policy.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", default="Isaacsimenvs-G1-Wuji-Sonic-v0",
        choices=("Isaacsimenvs-G1-Wuji-Sonic-v0", "Isaacsimenvs-G1-Wuji-Sonic-Lift-v0"),
    )
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--settle_steps", type=int, default=50)
    parser.add_argument(
        "--isaac_rgb",
        action="store_true",
        help="Also capture with the Isaac RTX renderer",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("../visualizations/g1_wuji_sonic")
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.frames < 1 or args.settle_steps < 1:
        parser.error("frames and settle_steps must be positive")
    args.headless = True
    args.enable_cameras = args.isaac_rgb
    app = AppLauncher(args).app

    import gymnasium as gym
    import numpy as np
    import torch
    from pxr import Gf, Sdf, UsdShade

    import isaacsimenvs  # noqa: F401 -- registers Gym tasks after Kit launch
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaacsimenvs.tasks.simtoolreal.pose_viewer import (
        build_pose_viewer_html,
        capture_pose_viewer_frame,
        object_urdf_for_env,
        table_urdf_for_env,
    )

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.viewer.resolution = (1440, 1080)
    cfg.seed = 42
    env = gym.make(
        args.task,
        cfg=cfg,
        render_mode="rgb_array" if args.isaac_rgb else None,
    )
    inner = env.unwrapped
    env.reset()
    action = inner.initial_policy_action.unsqueeze(0).clone()
    orange = (0.95, 0.42, 0.08)
    green = (0.20, 0.72, 0.31)

    # Apply the same display colors in Isaac and the browser; physics is unchanged.
    for name, color, prim in (
        ("Tool", orange, "/World/envs/env_0/Object"),
        ("Goal", green, "/World/envs/env_0/GoalViz"),
    ):
        material = UsdShade.Material.Define(inner.scene.stage, f"/World/Preview{name}")
        shader = UsdShade.Shader.Define(
            inner.scene.stage, f"/World/Preview{name}/Shader"
        )
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        UsdShade.MaterialBindingAPI.Apply(inner.scene.stage.GetPrimAtPath(prim)).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )

    for _ in range(args.settle_steps):
        env.step(action)

    frames = []
    reward_trace = []
    for index in range(args.frames):
        obs, reward, terminated, truncated, _ = env.step(action)
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
        frames.append(capture_pose_viewer_frame(inner, 0))
        terms = {name: float(value[0]) for name, value in inner._reward_terms.items()}
        # The environment publishes both the tool subtotal and full G1 reward.
        terms["total_reward"] = float(reward[0])
        reward_trace.append({"time_s": index * inner.step_dt, **terms})
        if bool(terminated[0]) or bool(truncated[0]):
            raise RuntimeError(
                "Preview ended unexpectedly; inspect the standing controller"
            )

    object_text, object_path = object_urdf_for_env(inner, 0)
    table_text, table_path = table_urdf_for_env(inner, 0)
    html = build_pose_viewer_html(
        frames=frames,
        object_urdf_text=object_text,
        table_urdf_text=table_text,
        object_urdf_path=object_path,
        table_urdf_path=table_path,
        robot_urdf_path=Path(cfg.assets.robot_urdf),
        timestep=inner.step_dt,
    )
    match = re.search(
        r'(<script id="scene-json" type="application/json">)(.*?)(</script>)',
        html,
        re.S,
    )
    if match is None:
        raise RuntimeError("Viewer template is missing its scene payload")
    payload = json.loads(match.group(2))
    for robot in payload["robots"]:
        if robot["name"] == "object":
            robot["color_override"] = list(orange)
    html = (
        html[: match.start(2)]
        + json.dumps(payload, separators=(",", ":"))
        + html[match.end(2) :]
    )
    html = html.replace(
        "W&B Interactive Robot Viewer", "G1 + Wuji training environment"
    )
    html = html.replace(
        "Interactive Robot Viewer</strong>", "G1 + Wuji training environment</strong>"
    )
    html = html.replace(
        "Three.js + urdf-loader prototype for W&amp;B",
        "Orange: physical tool · Green: target pose · Standing reference preview",
    )
    html = html.replace(
        "camera.position.set(0.0, -1.0, 1.03);", "camera.position.set(1.6, -2.1, 1.7);"
    )
    html = html.replace(
        "controls.target.set(0.0, 0.0, 0.53);", "controls.target.set(0.0, 0.25, 0.8);"
    )
    (out / "environment.html").write_text(html)
    (out / "reward_trace.json").write_text(json.dumps(reward_trace, indent=2) + "\n")
    (out / "frame.json").write_text(
        json.dumps(frames[-1], default=lambda value: value.tolist(), indent=2) + "\n"
    )
    (out / "object.urdf").write_text(object_text)
    (out / "table.urdf").write_text(table_text)

    def capture_image(name: str, eye, target) -> None:
        from PIL import Image

        inner.sim.set_camera_view(eye=eye, target=target)
        frame = None
        for _ in range(12):
            inner.sim.render()
            frame = env.render()
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[-1] < 3 or frame.std() < 1.0:
            raise RuntimeError(f"Invalid RGB render for {name}: {frame.shape}")
        Image.fromarray(frame[..., :3]).save(out / name)
        print(f"[preview] rendered {name}: {frame.shape}", flush=True)

    if args.isaac_rgb:
        capture_image("environment.png", cfg.viewer.eye, cfg.viewer.lookat)
        capture_image("hands_and_tool.png", (0.75, -0.95, 1.28), (0.0, 0.18, 1.00))

    manifest = {
        "task": args.task,
        "preview_controller": "SONIC 1.1 encoded standing reference",
        "manipulation_policy_trained": False,
        "sonic_latent_quantized": True,
        "policy_actions": 67,
        "physical_joints": 69,
        "preview_environments": 1,
        "default_training_environments": type(cfg)().scene.num_envs,
        "actor_observations": cfg.observation_space,
        "critic_observations": cfg.state_space,
        "control_hz": 1.0 / inner.step_dt,
        "physics_hz": 1.0 / cfg.sim.dt,
        "settle_time_s": args.settle_steps * inner.step_dt,
        "frames": len(frames),
        "tool_asset": str(object_path),
        "robot_urdf": cfg.assets.robot_urdf,
        "tool_types": list(cfg.assets.handle_head_types),
        "tool_variants": len(inner._object_urdf_paths),
        "table_size_m": [0.65, 0.45],
        "tabletop_height_m": cfg.reset.table_reset_z + 0.025,
        "reward": cfg.reward.to_dict(),
        "sonic_rewards": {
            "robot_fall": -cfg.sonic.fall_penalty,
            "latent_rate": -cfg.sonic.latent_rate_penalty,
            "physics_failure": -cfg.sonic.physics_failure_penalty,
        },
        "reset": cfg.reset.to_dict(),
        "termination": cfg.termination.to_dict(),
        "current_keypoint_tolerance_m": inner._current_success_tolerance
        * cfg.reward.keypoint_scale,
        "sample_reward": reward_trace[-1],
        "output_directory": str(out),
    }
    (out / "environment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    if not args.isaac_rgb:
        renderer = Path(__file__).resolve().parents[1] / "scripts/render_g1_preview.py"
        subprocess.run([sys.executable, str(renderer), str(out)], check=True)
    print(
        f"[preview] wrote {len(frames)} frames and scene/reward metadata to {out}",
        flush=True,
    )
    env.close()
    app.close()


if __name__ == "__main__":
    main()
