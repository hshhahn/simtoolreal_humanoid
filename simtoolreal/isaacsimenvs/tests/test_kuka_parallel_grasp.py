"""Check physical marker grasp/lift capability using a scripted joint trajectory.

This diagnostic disables DR and follows IK targets. It is not a learned-policy
evaluation and does not modify the training reward or reset configuration.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joints", type=Path,
                        default=Path(__file__).resolve().parents[2] / "assets/urdf/kuka_parallel_gripper/scripted_grasp_joints.json")
    parser.add_argument("--report", type=Path, default=Path("../.logs/kuka-gripper-scripted-grasp.json"))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    import gymnasium as gym
    import torch
    import isaacsimenvs
    from isaaclab.utils.math import quat_apply
    from isaacsimenvs.tasks.kuka_parallel_gripper.env_cfg import KukaParallelGripperLiftEnvCfg
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import DomainRandomizationCfg
    cfg = KukaParallelGripperLiftEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = 2
    cfg.termination.episode_length = 2000
    cfg.episode_length_s = 2000/60
    cfg.termination.max_consecutive_successes = 0
    # Fixed poses only make this capability diagnostic reproducible. Training
    # inherits the original randomized object resets and delta-pose goals.
    cfg.reset.fixed_start_pose = (0., 0., .60, 1., 0., 0., 0.)
    cfg.reset.fixed_goal_pose = (0., 0., .73, 1., 0., 0., 0.)
    cfg.reset.table_reset_z_range = 0.
    cfg.reset.reset_dof_pos_random_interval_arm = 0.
    cfg.reset.reset_dof_pos_random_interval_fingers = 0.
    cfg.reset.reset_dof_vel_random_interval = 0.
    cfg.domain_randomization = DomainRandomizationCfg(
        use_action_delay=False, use_obs_delay=False, use_object_state_delay_noise=False,
        force_scale=0., torque_scale=0.,
    )
    env = gym.make("Isaacsimenvs-Kuka-Parallel-Gripper-Lift-v0", cfg=cfg)
    raw = env.unwrapped
    env.reset()
    print(json.dumps({"joint_names": raw.robot.data.joint_names,
                      "joint_limits": raw.robot.data.joint_pos_limits[0].cpu().tolist()}), flush=True)
    original_dones = raw._get_dones
    def capture_dones():
        result = original_dones()
        if (result[0] | result[1]).any():
            print(json.dumps({"termination_reasons": {name: mask.cpu().tolist()
                              for name, mask in raw._termination_reasons.items()},
                              "object_positions": (raw.object.data.root_pos_w-raw.scene.env_origins).cpu().tolist(),
                              "joint_positions": raw.robot.data.joint_pos.cpu().tolist(),
                              "joint_velocities": raw.robot.data.joint_vel.cpu().tolist()}), flush=True)
        return result
    raw._get_dones = capture_dones
    solutions = json.loads(args.joints.read_text())
    initial = raw.robot.data.joint_pos.clone()
    actions = torch.zeros(2, 8, device=raw.device)
    phases = (("approach", 150, .05), ("grasp", 150, .05),
              ("close", 120, 0.), ("lift", 240, 0.), ("hold", 120, 0.))
    history = []
    maximum_z = torch.zeros(2, device=raw.device)
    for phase, steps, opening in phases:
        start = initial.clone()
        target = initial.clone()
        arm_phase = "grasp" if phase == "close" else "lift" if phase == "hold" else phase
        target[:, raw._arm_joint_ids] = torch.tensor(solutions[arm_phase], device=raw.device)
        target[:, raw._hand_joint_ids] = opening
        for step in range(steps):
            fraction = (step+1)/steps
            smooth = fraction*fraction*(3-2*fraction)
            raw._replay_target_lab_order = start + smooth*(target-start)
            _, reward, terminated, truncated, _ = env.step(actions)
            assert torch.isfinite(reward).all()
            assert not (terminated | truncated).any(), f"Script interrupted during {phase}"
            obj = raw.object.data.root_pos_w - raw.scene.env_origins
            maximum_z = torch.maximum(maximum_z, obj[:, 2])
        initial = target
        obj = raw.object.data.root_pos_w - raw.scene.env_origins
        history.append({"phase": phase, "object_positions": obj.cpu().tolist(),
                        "jaw_positions": raw.robot.data.joint_pos[:, raw._hand_joint_ids].cpu().tolist(),
                        "arm_target_error": (raw.robot.data.joint_pos[:, raw._arm_joint_ids]-target[:, raw._arm_joint_ids]).cpu().tolist(),
                        "palm_tcp": (raw.robot.data.body_pos_w[:, raw._palm_body_id]-raw.scene.env_origins
                                     + quat_apply(raw.robot.data.body_quat_w[:, raw._palm_body_id],
                                                  torch.tensor(raw.palm_center_offset, device=raw.device).expand(2, -1))).cpu().tolist(),
                        "goal_hits": raw._successes.cpu().tolist()})
        print(json.dumps(history[-1]), flush=True)
    final_pos = raw.object.data.root_pos_w - raw.scene.env_origins
    report = {"scripted_capability_test": True, "learned_policy": False,
              "history": history, "maximum_object_z": maximum_z.cpu().tolist(),
              "final_object_positions": final_pos.cpu().tolist(),
              "goal_hits": raw._successes.cpu().tolist()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    assert (final_pos[:, 2] > .70).all(), "The scripted gripper did not retain both markers above lift height"
    print("[KUKA/gripper scripted grasp] PASS", flush=True)
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
