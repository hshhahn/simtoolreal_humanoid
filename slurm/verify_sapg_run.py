"""Check a completed SAPG run's ranks, saved tensors, logs, and resource use."""
import argparse
import json
import math
import os
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
args = parser.parse_args()
torch.set_num_threads(max(1, min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))))
root = args.run_dir.resolve()
distributed = json.loads((root / "distributed-verification.json").read_text())
resources = json.loads((root / "resources.summary.json").read_text())
ranks = distributed["ranks"]
world = len(ranks)
experiment = root / "0_g1_wuji_sonic11_sapg"
events = EventAccumulator(str(experiment / "summaries"))
events.Reload()
scalars = [(tag, event) for tag in events.Tags()["scalars"] for event in events.Scalars(tag)]
bad_scalars = [{"tag": tag, "step": event.step, "value": str(event.value)}
               for tag, event in scalars if not math.isfinite(event.value)]
checkpoints = sorted((experiment / "nn").glob("*.pth"), key=lambda p: p.stat().st_mtime)
if not checkpoints:
    raise RuntimeError("No SAPG checkpoint was written")
checkpoint = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
checkpoint_ranks = sorted(checkpoint)
if checkpoint_ranks != list(range(world)):
    raise RuntimeError(f"Expected all {world} rank states; found {checkpoint_ranks}")
tensor_count = 0
finite_checkpoint = True
def inspect(value):
    global tensor_count, finite_checkpoint
    if isinstance(value, torch.Tensor):
        tensor_count += 1
        finite_checkpoint = finite_checkpoint and bool(torch.isfinite(value).all())
    elif isinstance(value, dict):
        for child in value.values():
            inspect(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            inspect(child)
inspect(checkpoint)
summary = {
    "world_size": world,
    "global_environments": ranks[0]["global_environments"],
    "global_nominal_minibatch": ranks[0]["global_minibatch"],
    "epochs": ranks[0]["epochs"],
    "devices": [r["environment_device"] for r in ranks],
    "seeds": [r["seed"] for r in ranks],
    "actor_and_critic_synchronized": distributed["parameters_synchronized"],
    "parameters_finite": distributed["parameters_finite"],
    "tensorboard_scalar_count": len(scalars),
    "tensorboard_scalars_finite": bool(scalars) and not bad_scalars,
    "nonfinite_scalars": bad_scalars,
    "checkpoint": str(checkpoints[-1]),
    "checkpoint_ranks": checkpoint_ranks,
    "checkpoint_tensor_count": tensor_count,
    "checkpoint_tensors_finite": finite_checkpoint,
    "resources": resources,
}
summary["passed"] = all((
    summary["actor_and_critic_synchronized"],
    summary["parameters_finite"],
    summary["tensorboard_scalars_finite"],
    finite_checkpoint,
    len(set(summary["devices"])) == world,
    len(set(summary["seeds"])) == world,
    all(r["backend"] == "nccl" and r["epochs"] == summary["epochs"] for r in ranks),
    resources["exit_code"] == 0 and not resources["ram_guard_triggered"],
    all(v > 0 for v in resources["gpu_peak_utilization_percent"].values()),
))
(root / "verification-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if not summary["passed"]:
    raise SystemExit("SAPG validation failed; see verification-summary.json")
