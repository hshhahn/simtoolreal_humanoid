"""PhysX Wuji marker capability search, independent of SONIC and PPO rewards.

Uses the exact production three-command hand mapping, both procedural training
markers, hand gains/limits/materials/self-collisions, and 200 Hz physics. A
driven Cartesian wrist fixture isolates finger capability. Objects are always
dynamic with gravity; only the wrist is positioned by the diagnostic.

Success requires a 15 cm wrist lift, five seconds of retention, low final
object speed, and dropping the object after opening. Height crossings alone
are not successes. Failed searches do not prove a command space impossible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=192)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--mode", choices=("3", "6", "auto"), default="auto")
    parser.add_argument("--candidate-file", type=Path,
                        help="Replay explicit wrist poses and normalized hand commands for a controlled comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("../visualizations/wuji_marker_capability"))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.candidates < 1 or args.batches < 1:
        parser.error("--candidates and --batches must be positive")
    if args.candidate_file and args.mode == "auto":
        parser.error("--candidate-file requires an explicit --mode 3 or --mode 6")
    args.headless = True
    app = AppLauncher(args).app

    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.envs import DirectRLEnv
    from isaaclab.sim import UsdFileCfg
    from isaacsimenvs.tasks.g1_wuji_sonic.env_cfg import G1WujiSonicLiftEnvCfg
    from isaacsimenvs.tasks.g1_wuji_sonic.robot import collision_adjacency
    from isaacsimenvs.tasks.simtoolreal.utils.scene_utils import apply_physx_material_properties, setup_scene
    from simtoolreal_shared.wuji_synergy import synergy_targets, wuji_joint_names
    from scripts.wuji_grasp_fixture import FIXTURE_JOINT_NAMES, build_fixture, independent_targets

    torch.set_num_threads(2)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[2] / "assets/urdf/g1_wuji/g1_wuji.urdf"
    fixture = build_fixture(source, output / "wuji_wrist_fixture.urdf")
    parameters = json.loads((source.parent / "wuji_actuators.json").read_text())
    hand_names = wuji_joint_names("right")
    fields = ("stiffness", "damping", "armature", "effort_limit_sim", "velocity_limit_sim")

    class FixtureEnv(DirectRLEnv):
        fingertip_link_names = tuple(f"right_finger{i}_tip_link" for i in range(1, 6))

        def __init__(self, cfg):
            self.robot_collision_adjacency = collision_adjacency(str(fixture))
            super().__init__(cfg)
            apply_physx_material_properties(self)
            self.fixture_ids = self.robot.find_joints(list(FIXTURE_JOINT_NAMES), preserve_order=True)[0]
            self.hand_ids = self.robot.find_joints(list(hand_names), preserve_order=True)[0]
            limits = self.robot.data.joint_pos_limits[:, self.hand_ids]
            self.lower, self.upper = limits[..., 0], limits[..., 1]
            self.targets = self.robot.data.default_joint_pos.clone()
            self.hand_commands = torch.full((self.num_envs, 3), -1.0, device=self.device)

        def build_robot_articulation_cfg(self, usd):
            defaults = {name: 0.0 for name in FIXTURE_JOINT_NAMES + hand_names}
            defaults.update(fixture_z=0.9, right_finger1_joint1=0.0475)
            return ArticulationCfg(
                prim_path="/World/envs/env_.*/Robot", spawn=UsdFileCfg(usd_path=usd),
                init_state=ArticulationCfg.InitialStateCfg(joint_pos=defaults),
                soft_joint_pos_limit_factor=0.9,
                actuators={
                    "fixture": ImplicitActuatorCfg(
                        joint_names_expr=list(FIXTURE_JOINT_NAMES), stiffness=20000.0,
                        damping=400.0, effort_limit_sim=5000.0, velocity_limit_sim=2.0,
                        armature=0.1,
                    ),
                    "hand": ImplicitActuatorCfg(
                        joint_names_expr=list(hand_names),
                        **{f: {n: parameters[n][f] for n in hand_names} for f in fields},
                    ),
                },
            )

        def _setup_scene(self):
            setup_scene(self)
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)
            if not hasattr(self, "targets"):
                return
            ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
            q = self.robot.data.default_joint_pos[ids].clone()
            self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
            self.targets[ids] = q
            for asset, z in ((self.table, 0.475), (self.object, 0.54), (self.goal_viz, 0.70)):
                state = asset.data.default_root_state[ids].clone()
                state[:, :3] = self.scene.env_origins[ids] + torch.tensor((0.0, 0.0, z), device=self.device)
                state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device)
                state[:, 7:] = 0.0
                asset.write_root_state_to_sim(state, env_ids=ids)
            self.hand_commands.fill_(-1.0)

        def _pre_physics_step(self, actions):
            mapper = synergy_targets if self.hand_commands.shape[-1] == 3 else independent_targets
            hand_target = mapper(self.hand_commands, self.lower, self.upper)
            self.targets[:, self.hand_ids] = torch.lerp(self.targets[:, self.hand_ids], hand_target, 0.3)

        def _apply_action(self):
            self.robot.set_joint_position_target(self.targets)

        def _get_dones(self):
            empty = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return empty, empty

        def _get_rewards(self):
            return torch.zeros(self.num_envs, device=self.device)

        def _get_observations(self):
            return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}

    explicit_batch = None
    if args.candidate_file:
        data = json.loads(args.candidate_file.read_text())
        explicit_batch = {key: np.asarray(data[key], dtype=float)
                          for key in ("local_centers", "rotations", "commands")}
        args.candidates = len(explicit_batch["commands"])
        assert explicit_batch["local_centers"].shape == (args.candidates, 3)
        assert explicit_batch["rotations"].shape == (args.candidates, 3, 3)
        assert explicit_batch["commands"].shape == (args.candidates, int(args.mode))
        assert np.isfinite(explicit_batch["commands"]).all()
        assert (np.abs(explicit_batch["commands"]) <= 1.0).all()
    cfg = G1WujiSonicLiftEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.candidates * 2
    cfg.scene.env_spacing = 1.5
    cfg.assets.robot_urdf = str(fixture)
    cfg.assets.robot_fix_base = True
    cfg.action_space = 12
    cfg.observation_space = 1
    cfg.state_space = 0
    cfg.episode_length_s = 1000.0
    env = FixtureEnv(cfg)
    env.reset()
    device = env.device
    dummy = torch.zeros(env.num_envs, 12, device=device)
    rng = np.random.default_rng(args.seed)
    # Palm normal points down; marker's long axis follows the palm's width.
    base_rotation = np.array(((0.0, 1.0, 0.0), (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0)))
    trials = []
    traces = {}

    def run_batch(mode, batch, negative_control=False):
        count = env.num_envs // 2
        if batch is None:
            local = np.column_stack((rng.uniform(0.035, 0.080, count),
                                     rng.uniform(-0.015, 0.015, count),
                                     rng.uniform(0.065, 0.125, count)))
            angles = rng.uniform((-0.35, -0.45, -0.35), (0.35, 0.45, 0.35), (count, 3))
            rotation = Rotation.from_euler("xyz", angles).as_matrix() @ base_rotation
            closure = rng.uniform((0.10, 0.15) + (0.50,) * (mode - 2),
                                  (0.95, 0.90) + (1.0,) * (mode - 2), (count, mode))
            batch = {"local_centers": local, "rotations": rotation, "commands": 2 * closure - 1}
        local = np.repeat(batch["local_centers"], 2, axis=0)
        rotation = np.repeat(batch["rotations"], 2, axis=0)
        closed = torch.tensor(np.repeat(batch["commands"], 2, axis=0), device=device, dtype=torch.float32)
        if negative_control:
            closed.fill_(-1.0)
        env.hand_commands = torch.full_like(closed, -1.0)
        env.reset()
        eulers = Rotation.from_matrix(rotation).as_euler("xyz")[:, ::-1].copy()
        eulers = torch.tensor(eulers, device=device, dtype=torch.float32)
        env.targets[:, env.fixture_ids[3:]] = eulers
        # Initialize above the table with an open hand. Joint writes occur only
        # on reset; throughout approach/lift/hold the fixture uses physical drives.
        q = env.targets.clone()
        env.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        for _ in range(25):
            env.step(dummy)
        object_start = (env.object.data.root_pos_w - env.scene.env_origins).clone()
        grasp_xyz = object_start.cpu().numpy() - np.einsum("nij,nj->ni", rotation, local)
        grasp_xyz = torch.tensor(grasp_xyz, device=device, dtype=torch.float32)
        hover_xyz = grasp_xyz.clone()
        hover_xyz[:, 2] += 0.16
        env.targets[:, env.fixture_ids[:3]] = hover_xyz
        env.robot.write_joint_state_to_sim(env.targets, torch.zeros_like(env.targets))
        history = []
        hold_z = []
        phases = (("hover", 25), ("approach", 75), ("close", 75),
                  ("lift", 100), ("hold", 250), ("release", 100))
        max_speed = torch.zeros(env.num_envs, device=device)
        hold_final = None
        for phase, steps in phases:
            for step in range(steps):
                fraction = (step + 1) / steps
                smooth = fraction * fraction * (3.0 - 2.0 * fraction)
                wrist = grasp_xyz.clone()
                if phase == "hover":
                    wrist = hover_xyz
                elif phase == "approach":
                    wrist = hover_xyz + smooth * (grasp_xyz - hover_xyz)
                elif phase in ("lift", "hold", "release"):
                    wrist[:, 2] += 0.15 * (smooth if phase == "lift" else 1.0)
                env.targets[:, env.fixture_ids[:3]] = wrist
                if phase == "close":
                    env.hand_commands = -1.0 + smooth * (closed + 1.0)
                elif phase in ("lift", "hold"):
                    env.hand_commands = closed
                elif phase == "release":
                    env.hand_commands = closed + smooth * (-1.0 - closed)
                else:
                    env.hand_commands = torch.full_like(closed, -1.0)
                env.step(dummy)
                obj = env.object.data.root_pos_w - env.scene.env_origins
                speed = env.object.data.root_lin_vel_w.norm(dim=-1)
                assert torch.isfinite(env.robot.data.joint_pos).all() and torch.isfinite(obj).all()
                max_speed = torch.maximum(max_speed, speed)
                if phase == "hold":
                    hold_z.append(obj[:, 2].clone())
                if step % 5 == 0:
                    history.append({"phase": phase, "time_s": len(history) * 0.1,
                                    "q": env.robot.data.joint_pos.clone(),
                                    "obj": env.object.data.root_state_w.clone(),
                                    "commands": env.hand_commands.clone()})
            if phase == "hold":
                hold_final = {"xyz": obj.clone(), "speed": speed.clone()}
            print(json.dumps({"mode": mode, "phase": phase,
                              "objects_above_10cm": int((obj[:, 2] > object_start[:, 2] + 0.10).sum())}), flush=True)
        minimum_hold_z = torch.stack(hold_z).min(dim=0).values
        retained = minimum_hold_z > object_start[:, 2] + 0.10
        stable = hold_final["speed"] < 0.15
        dropped = hold_final["xyz"][:, 2] - obj[:, 2] > 0.08
        success = retained & stable & dropped
        result = {"mode": mode, "negative_control": negative_control,
                  "successful_grasps": int(success.sum()),
                  "successes_per_marker": [int(success[i::2].sum()) for i in range(2)],
                  "retained_per_marker": [int(retained[i::2].sum()) for i in range(2)],
                  "retained_and_stable_per_marker": [int((retained & stable)[i::2].sum()) for i in range(2)],
                  "candidates": []}
        score = minimum_hold_z - object_start[:, 2]
        for i in range(env.num_envs):
            c = i // 2
            result["candidates"].append({
                "candidate": c, "marker": i % 2, "success": bool(success[i]),
                "minimum_hold_height_gain_m": float(score[i]),
                "final_hold_speed_m_s": float(hold_final["speed"][i]),
                "drop_after_opening_m": float(hold_final["xyz"][i, 2] - obj[i, 2]),
                "maximum_object_speed_m_s": float(max_speed[i]),
                "local_center": local[i].tolist(), "rotation": rotation[i].tolist(),
                "commands": closed[i].cpu().tolist(),
            })
        for marker in range(2):
            ranked = score[marker::2] + success[marker::2].float()
            best_id = 2 * int(ranked.argmax()) + marker
            key = f"mode{mode}_marker{marker}_trial{len(trials)}"
            trace = []
            for frame in history:
                state = frame["obj"][best_id].cpu().tolist()
                state[:3] = (frame["obj"][best_id, :3] - env.scene.env_origins[best_id]).cpu().tolist()
                trace.append({"time_s": frame["time_s"], "phase": frame["phase"],
                              "robot_base_pose": env.robot.data.root_state_w[best_id, :7].cpu().tolist(),
                              "robot_joint_names": env.robot.data.joint_names,
                              "robot_joint_pos": frame["q"][best_id].cpu().tolist(),
                              "object_state": state, "hand_commands": frame["commands"][best_id].cpu().tolist()})
                trace[-1]["robot_base_pose"][:3] = [0.0, 0.0, 0.0]
            (output / f"{key}_trace.json").write_text(json.dumps(trace))
            traces[key] = result["candidates"][best_id]
        trials.append(result)
        (output / "trials.json").write_text(json.dumps(trials, indent=2) + "\n")
        return success, batch

    modes = (3, 6) if args.mode == "auto" else (int(args.mode),)
    proven = None
    for mode in modes:
        for index in range(args.batches):
            success, batch = run_batch(mode, explicit_batch)
            if all(success[i::2].any() for i in range(2)):
                proven = mode
                # Repeat the winning poses with all hand commands open. This
                # rejects a fixture/geometry support trick masquerading as grasp.
                control, _ = run_batch(mode, batch, negative_control=True)
                assert not control.any(), "Open-hand control also lifted/retained an object"
                break
        if proven == 3:
            break
    report = {"diagnostic": "physical_wuji_grasp_capability", "learned_policy": False,
              "fixture_isolates_wrist_from_sonic_and_balance": True,
              "production_hand_commands_changed": False,
              "verified_hand_dof": proven, "seed": args.seed,
              "candidate_file": str(args.candidate_file.resolve()) if args.candidate_file else None,
              "criteria": {"wrist_lift_m": 0.15, "minimum_retained_lift_m": 0.10,
                           "hold_seconds": 5.0, "maximum_final_speed_m_s": 0.15,
                           "minimum_drop_after_opening_m": 0.08},
              "hand_urdf": str(source), "hand_urdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "hand_actuators": {n: parameters[n] for n in hand_names},
              "robot_friction": cfg.assets.robot_friction,
              "fingertip_friction": cfg.assets.finger_tip_friction,
              "object_friction_configuration": cfg.assets.object_friction,
              "object_assets": env._object_urdf_paths, "best_traces": traces,
              "trials_summary": [{k: v for k, v in t.items() if k != "candidates"} for t in trials]}
    for i, path in enumerate(env._object_urdf_paths):
        (output / f"marker{i}.urdf").write_bytes(Path(path).read_bytes())
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"verified_hand_dof": proven, "report": str(output / "report.json")}), flush=True)
    env.close()
    del app
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
