#!/usr/bin/env bash
set -euo pipefail
_sharpa_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_sharpa_workspace}/activate_simtoolreal.sh"
export PYTHONPATH="${SIMTOOLREAL_ROOT}/rl_games:${SIMTOOLREAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Use the original SimToolReal environment and original 22-DoF SHARPA hand.
# Only the procedural object pool is restricted to the two marker variants.
# Scale num_envs, exploration block and minibatches together for six SAPG groups.
exec python isaacsimenvs/train.py \
    --task Isaacsimenvs-SimToolReal-Direct-v0 \
    --agent rl_games_sapg_cfg_entry_point --headless --capture_viewer \
    --capture_viewer_github_raw_base https://raw.githubusercontent.com/hshhahn/simtoolreal_humanoid/main/simtoolreal/ \
    'env.assets.handle_head_types=[marker]' \
    env.assets.num_assets_per_type=2 \
    env.assets.shuffle_assets=False \
    env.scene.num_envs=24576 \
    agent.params.config.expl_coef_block_size=4096 \
    agent.params.config.name=0_kuka_sharpa_sapg \
    agent.params.config.max_epochs=40000 \
    "$@"
