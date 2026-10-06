"""Verify that a completed distributed run trained a single finite policy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist


def _model_fingerprint(model):
    parameters = hashlib.sha256()
    buffers = hashlib.sha256()
    finite = True
    count = 0
    for digest, tensors in (
        (parameters, model.named_parameters()),
        (buffers, model.named_buffers()),
    ):
        for name, tensor in tensors:
            value = tensor.detach().cpu().contiguous()
            finite = finite and bool(torch.isfinite(value).all())
            digest.update(name.encode())
            digest.update(value.numpy().tobytes())
            if digest is parameters:
                count += tensor.numel()
    return parameters.hexdigest(), buffers.hexdigest(), finite, count


def verify_training_ranks(algo, seed, environment_device):
    """Save per-rank evidence and reject divergent actor/critic parameters."""
    actor = _model_fingerprint(algo.model)
    critic = _model_fingerprint(algo.central_value_net.model) if algo.has_central_value else None
    device = torch.device(algo.ppo_device)
    report = {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "backend": dist.get_backend(),
        "rl_device": str(device),
        "environment_device": str(environment_device),
        "seed": seed,
        "epochs": algo.epoch_num,
        "environments_per_rank": algo.num_actors,
        "global_environments": algo.num_actors * dist.get_world_size(),
        "minibatch_per_rank": algo.minibatch_size,
        "global_minibatch": algo.minibatch_size * dist.get_world_size(),
        "exploration_block_per_rank": getattr(algo, "intr_coef_block_size", None),
        "exploration_type": algo.expl_type,
        "use_others_experience": algo.use_others_experience,
        "actor_parameters_sha256": actor[0],
        "actor_buffers_sha256": actor[1],
        "actor_parameter_count": actor[3],
        "parameters_finite": actor[2] and (critic is None or critic[2]),
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
    if critic is not None:
        report.update(critic_parameters_sha256=critic[0], critic_buffers_sha256=critic[1])
    output = Path(algo.train_dir)
    (output / f"rank_{report['rank']}_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    reports = [None] * dist.get_world_size()
    dist.all_gather_object(reports, report)
    keys = ["actor_parameters_sha256", "actor_buffers_sha256"]
    if critic is not None:
        keys += ["critic_parameters_sha256", "critic_buffers_sha256"]
    synchronized = all(len({r[key] for r in reports}) == 1 for key in keys)
    finite = all(r["parameters_finite"] for r in reports)
    if dist.get_rank() == 0:
        summary = {"parameters_synchronized": synchronized, "parameters_finite": finite, "ranks": reports}
        (output / "distributed-verification.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[distributed] synchronized={synchronized} finite={finite} ranks={len(reports)}", flush=True)
    if not synchronized or not finite:
        raise RuntimeError("Distributed actor/critic verification failed; see distributed-verification.json")
