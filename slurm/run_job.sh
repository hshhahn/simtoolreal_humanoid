#!/usr/bin/env bash
# Run through the sbatch templates; each process uses one allocated GPU.
set -euo pipefail

g1_mode="${1:-verify}"
case "${g1_mode}" in
    verify|train) ;;
    *) printf 'Usage: %s verify|train\n' "$0" >&2; exit 2 ;;
esac
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf 'Run this script inside a Slurm GPU allocation.\n' >&2
    exit 1
fi
g1_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
g1_job_id="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
g1_scratch_base="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
g1_scratch="$(mktemp -d "${g1_scratch_base%/}/simtoolreal_${g1_job_id}_XXXXXX")"
export TMPDIR="${g1_scratch}/tmp"
export XDG_CACHE_HOME="${g1_scratch}/cache"
export OMNI_KIT_CACHE_PATH="${g1_scratch}/omniverse"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${OMNI_KIT_CACHE_PATH}" "${g1_root}/slurm_logs"
# Activation preserves the node-local cache variables above.
source "${g1_root}/activate_simtoolreal.sh"
export HF_HOME="${g1_root}/.cache/huggingface"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

g1_arch="${ARCH:-mlp}"
g1_seed="${SEED:-42}"
g1_array_index="${SLURM_ARRAY_TASK_ID:-0}"
g1_num_envs="${NUM_ENVS:-128}"
g1_max_epochs="${MAX_EPOCHS:-40000}"
g1_minibatch="${MINIBATCH_SIZE:-2048}"
for g1_number in "${g1_seed}" "${g1_array_index}" "${g1_num_envs}" "${g1_max_epochs}" "${g1_minibatch}"; do
    if [[ ! "${g1_number}" =~ ^[0-9]+$ ]]; then
        printf 'Seed, array index, environment count, epochs and minibatch must be integers.\n' >&2
        exit 2
    fi
done
if [[ "${g1_arch}" == compare ]]; then
    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        printf 'ARCH=compare requires a Slurm array; use --array=0-3.\n' >&2
        exit 2
    fi
    if (( g1_array_index % 2 == 0 )); then g1_arch=mlp; else g1_arch=lstm; fi
    g1_seed=$((g1_seed + g1_array_index / 2))
else
    g1_seed=$((g1_seed + g1_array_index))
fi
case "${g1_arch}" in
    mlp) g1_agent=rl_games_cfg_entry_point ;;
    lstm) g1_agent=rl_games_lstm_cfg_entry_point ;;
    *) printf 'ARCH must be mlp, lstm or compare.\n' >&2; exit 2 ;;
esac
if (( g1_num_envs < 1 || g1_max_epochs < 1 || g1_minibatch < 1 )); then
    printf 'Environment count, epochs and minibatch must be positive.\n' >&2
    exit 2
fi
if (( (g1_num_envs * 16) % g1_minibatch != 0 )); then
    printf 'NUM_ENVS * 16 must be divisible by MINIBATCH_SIZE. Defaults are 128 and 2048.\n' >&2
    exit 2
fi
if [[ "${g1_arch}" == lstm ]] && (( g1_minibatch % 16 != 0 )); then
    printf 'The LSTM minibatch must be divisible by its sequence length, 16.\n' >&2
    exit 2
fi
g1_run_dir="${RUN_DIR:-${g1_root}/g1_wuji_runs/${g1_job_id}_${g1_mode}_${g1_arch}_seed${g1_seed}}"
mkdir -p "${g1_run_dir}"
g1_run_dir="$(cd -- "${g1_run_dir}" && pwd)"
printf 'Mode=%s architecture=%s seed=%s environments=%s output=%s\n' \
    "${g1_mode}" "${g1_arch}" "${g1_seed}" "${g1_num_envs}" "${g1_run_dir}"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-managed by Slurm}"
python "${g1_root}/slurm/check_runtime.py" > "${g1_run_dir}/runtime.json"
cat "${g1_run_dir}/runtime.json"

if [[ "${g1_mode}" == verify ]]; then
    python isaacsimenvs/tests/test_sonic_wuji_contract.py
    python isaacsimenvs/tests/test_sonic_policy_exploration.py
    python isaacsimenvs/tests/test_sonic_lstm_policy.py
    python isaacsimenvs/tests/test_physics_guard.py
    python rl_games/tests/test_checkpoint_symlink.py
    python isaacsimenvs/tests/test_g1_wuji_sonic.py \
        --task Isaacsimenvs-G1-Wuji-Sonic-Lift-v0 --num_envs 4 --steps 500 \
        --headless --report "${g1_run_dir}/environment-verification.json"
    if [[ "${SMOKE_PPO:-1}" == 1 ]]; then
        for g1_smoke_agent in rl_games_cfg_entry_point rl_games_lstm_cfg_entry_point; do
            python isaacsimenvs/train.py \
                --task Isaacsimenvs-G1-Wuji-Sonic-Lift-v0 \
                --agent "${g1_smoke_agent}" --headless \
                --sim_device cuda:0 --rl_device cuda:0 \
                env.scene.num_envs=128 agent.params.config.max_epochs=2 \
                "hydra.run.dir=${g1_run_dir}/smoke_${g1_smoke_agent}"
        done
        python - "${g1_run_dir}" <<'PY'
from pathlib import Path
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

root = Path(sys.argv[1])
for agent in ("rl_games_cfg_entry_point", "rl_games_lstm_cfg_entry_point"):
    run = root / f"smoke_{agent}"
    checkpoints = list(run.rglob("*.pth"))
    events = list(run.rglob("events.out.tfevents.*"))
    assert checkpoints, f"No checkpoint saved in {run}"
    assert events, f"No TensorBoard events written in {run}"
    scalar_count = 0
    for event in events:
        accumulator = EventAccumulator(str(event)).Reload()
        for tag in accumulator.Tags()["scalars"]:
            values = [item.value for item in accumulator.Scalars(tag)]
            assert np.isfinite(values).all(), f"Nonfinite TensorBoard scalar: {tag}"
            scalar_count += len(values)
    assert scalar_count, f"No scalar values written in {run}"
    print(f"Smoke artifacts verified: {agent}, {len(checkpoints)} checkpoints, {scalar_count} finite scalars")
PY
    fi
    printf 'Verification passed. Reports and smoke-run outputs: %s\n' "${g1_run_dir}"
    exit 0
fi

g1_args=(
    --task "${TASK:-Isaacsimenvs-G1-Wuji-Sonic-Lift-v0}"
    --agent "${g1_agent}" --headless --sim_device cuda:0 --rl_device cuda:0
    "env.scene.num_envs=${g1_num_envs}"
    "agent.params.seed=${g1_seed}"
    "agent.params.config.max_epochs=${g1_max_epochs}"
    "agent.params.config.minibatch_size=${g1_minibatch}"
    "agent.params.config.central_value_config.minibatch_size=${g1_minibatch}"
    "hydra.run.dir=${g1_run_dir}"
)
if [[ -n "${CHECKPOINT:-}" ]]; then
    if [[ "${CHECKPOINT}" != /* || ! -f "${CHECKPOINT}" ]]; then
        printf 'CHECKPOINT must be the absolute path to an existing checkpoint.\n' >&2
        exit 2
    fi
    g1_args+=(--checkpoint "${CHECKPOINT}" --checkpoint_load_mode "${CHECKPOINT_LOAD_MODE:-resume}")
fi
exec python isaacsimenvs/train.py "${g1_args[@]}"
