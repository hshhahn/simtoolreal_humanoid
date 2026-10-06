#!/usr/bin/env bash
set -euo pipefail
_bimanual_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_bimanual_workspace}/activate_simtoolreal.sh"
export HF_HOME="${_bimanual_workspace}/.cache/huggingface"
export PYTHONPATH="${SIMTOOLREAL_ROOT}/rl_games:${SIMTOOLREAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python isaacsimenvs/train.py \
  --task Isaacsimenvs-G1-Wuji-Sonic-Bimanual-v0 \
  --agent rl_games_sapg_cfg_entry_point --headless --capture_viewer \
  --capture_viewer_len 500 \
  'env.scene.num_envs=6144' \
  'agent.params.config.expl_coef_block_size=1024' \
  'agent.params.config.minibatch_size=24576' \
  'agent.params.config.central_value_config.minibatch_size=24576' \
  'agent.params.config.name=0_g1_wuji_bimanual_sapg' \
  'agent.params.config.max_epochs=40000' \
  'agent.params.config.save_frequency=1000' \
  'agent.params.config.save_latest_frequency=100' "$@"
