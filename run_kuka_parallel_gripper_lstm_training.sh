#!/usr/bin/env bash
set -euo pipefail
_kuka_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_kuka_workspace}/activate_simtoolreal.sh"
export PYTHONPATH="${SIMTOOLREAL_ROOT}/rl_games:${SIMTOOLREAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Keep the original scene settings; scale only the number of parallel workers
# for this workstation. Later Hydra arguments can override this default.
exec python isaacsimenvs/train.py --task Isaacsimenvs-Kuka-Parallel-Gripper-Lift-v0 --headless env.scene.num_envs=2048 "$@"
