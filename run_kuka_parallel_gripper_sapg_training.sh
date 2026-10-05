#!/usr/bin/env bash
set -euo pipefail
_kuka_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_kuka_workspace}/activate_simtoolreal.sh"
export PYTHONPATH="${SIMTOOLREAL_ROOT}/rl_games:${SIMTOOLREAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Reuse the upstream SAPG YAML, including its actor LSTM, asymmetric critic,
# coefficient-conditioned exploration and leader/follower experience sharing.
# Keep six exploration blocks when scaling num_envs and expl_coef_block_size.
exec python isaacsimenvs/train.py \
    --task Isaacsimenvs-Kuka-Parallel-Gripper-Lift-v0 \
    --agent rl_games_sapg_cfg_entry_point --headless --capture_viewer \
    --capture_viewer_github_raw_base https://raw.githubusercontent.com/hshhahn/simtoolreal_humanoid/main/simtoolreal/ \
    env.scene.num_envs=24576 \
    agent.params.config.expl_coef_block_size=4096 \
    agent.params.config.name=0_kuka_parallel_gripper_sapg \
    agent.params.config.max_epochs=40000 \
    "$@"
