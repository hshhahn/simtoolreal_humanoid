#!/usr/bin/env bash
# Reproduce the 2026-10-06 right-hand G1 SAPG runs; execute inside Slurm.
set -euo pipefail
gentle_variant="${1:-}"
case "${gentle_variant}" in
    original) export TASK=Isaacsimenvs-G1-Wuji-Sonic-v0 ;;
    height145) export TASK=Isaacsimenvs-G1-Wuji-Sonic-Height-v0 ;;
    -h|--help)
        printf 'Usage: bash slurm/run_g1_gentle.sh {original|height145} [training arguments / Hydra overrides]\n'
        exit 0 ;;
    *)
        printf 'Usage: bash slurm/run_g1_gentle.sh {original|height145} [training arguments / Hydra overrides]\n' >&2
        exit 2 ;;
esac
shift
gentle_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export GPUS="${GPUS:-4}"
export NUM_ENVS="${NUM_ENVS:-30720}"
export MINIBATCH_SIZE="${MINIBATCH_SIZE:-98304}"
export MAX_EPOCHS="${MAX_EPOCHS:-1000000}"
export SEED="${SEED:-42}"
exec bash "${gentle_root}/slurm/run_sapg.sh" \
    env.assets.num_assets_per_type=100 \
    env.sonic.smooth_right_arm_targets=true \
    env.action.arm_moving_average=0.3 \
    env.action.dof_speed_scale=5.0 \
    env.action.hand_moving_average=0.3 "$@"
