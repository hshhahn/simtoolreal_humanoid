#!/usr/bin/env bash
# Run inside one Slurm step exposing all GPUs. Counts are global across ranks.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf 'Run SAPG inside a Slurm GPU allocation.\n' >&2
    exit 1
fi
sapg_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
sapg_gpus="${GPUS:-4}"
sapg_envs="${NUM_ENVS:-24576}"
sapg_batch="${MINIBATCH_SIZE:-98304}"
sapg_epochs="${MAX_EPOCHS:-1000000}"
sapg_seed="${SEED:-42}"
for number in "${sapg_gpus}" "${sapg_envs}" "${sapg_batch}" "${sapg_epochs}" "${sapg_seed}"; do
    [[ "${number}" =~ ^[0-9]+$ ]] || { printf 'GPU, environment, minibatch, epoch and seed counts must be integers.\n' >&2; exit 2; }
done
if (( sapg_gpus < 2 || sapg_envs < 1 || sapg_batch < 1 || sapg_epochs < 1 || sapg_envs % (6 * sapg_gpus) != 0 || sapg_batch % (16 * sapg_gpus) != 0 || (sapg_envs * 16) % sapg_batch != 0 )); then
    printf 'Use >=2 GPUs, six equal exploration groups per GPU, and a minibatch dividing NUM_ENVS * 16 with sequences of 16.\n' >&2
    exit 2
fi
export GPUS="${sapg_gpus}"
export SIMTOOLREAL_RUN_DIR="${RUN_DIR:-${sapg_root}/g1_wuji_runs/${SLURM_JOB_ID}_sapg_${sapg_envs}env_${sapg_epochs}ep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${SIMTOOLREAL_RUN_DIR}"
export SIMTOOLREAL_RUN_DIR="$(cd -- "${SIMTOOLREAL_RUN_DIR}" && pwd)"
if [[ -e "${SIMTOOLREAL_RUN_DIR}/rank_0.log" ]]; then
    printf 'Choose a fresh RUN_DIR to preserve existing training artifacts.\n' >&2
    exit 2
fi
export SIMTOOLREAL_SCRATCH="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/simtoolreal_sapg_${SLURM_JOB_ID}_XXXXXX")"
trap 'rm -rf -- "${SIMTOOLREAL_SCRATCH}"' EXIT
source "${sapg_root}/activate_simtoolreal.sh"
export HF_HOME="${sapg_root}/.cache/huggingface"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / sapg_gpus ))
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export PXR_WORK_THREAD_LIMIT="${OMP_NUM_THREADS}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
python - "${sapg_gpus}" <<'PY'
import sys, torch
expected = int(sys.argv[1])
if torch.cuda.device_count() != expected:
    raise SystemExit(f"Expected {expected} GPUs in this Slurm step; found {torch.cuda.device_count()}")
PY
printf 'One distributed SAPG policy: GPUs=%s global_envs=%s global_minibatch=%s epochs=%s output=%s\n' "${sapg_gpus}" "${sapg_envs}" "${sapg_batch}" "${sapg_epochs}" "${SIMTOOLREAL_RUN_DIR}"
python "${sapg_root}/slurm/check_runtime.py" > "${SIMTOOLREAL_RUN_DIR}/runtime.json"
python "${sapg_root}/slurm/monitor_sapg.py" --output "${SIMTOOLREAL_RUN_DIR}/resources.jsonl" -- \
    torchrun --standalone --nnodes=1 --nproc-per-node="${sapg_gpus}" \
    "${sapg_root}/slurm/sapg_worker.py" \
    --task "${TASK:-Isaacsimenvs-G1-Wuji-Sonic-Lift-v0}" \
    --agent rl_games_sapg_cfg_entry_point --distributed --headless \
    "env.scene.num_envs=$((sapg_envs / sapg_gpus))" \
    "agent.params.config.expl_coef_block_size=$((sapg_envs / sapg_gpus / 6))" \
    "agent.params.config.minibatch_size=$((sapg_batch / sapg_gpus))" \
    "agent.params.config.central_value_config.minibatch_size=$((sapg_batch / sapg_gpus))" \
    "agent.params.config.max_epochs=${sapg_epochs}" \
    "agent.params.seed=${sapg_seed}" "$@"
