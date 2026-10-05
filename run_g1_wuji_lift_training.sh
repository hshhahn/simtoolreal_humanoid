#!/usr/bin/env bash
set -euo pipefail
_g1_lift_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_g1_lift_workspace}/activate_simtoolreal.sh"
export HF_HOME="${_g1_lift_workspace}/.cache/huggingface"
exec python isaacsimenvs/train.py --task Isaacsimenvs-G1-Wuji-Sonic-Lift-v0 --headless "$@"
