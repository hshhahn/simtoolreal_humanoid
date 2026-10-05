#!/usr/bin/env bash
set -euo pipefail
_g1_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_g1_workspace}/activate_simtoolreal.sh"
export HF_HOME="${_g1_workspace}/.cache/huggingface"
exec python isaacsimenvs/train.py --task Isaacsimenvs-G1-Wuji-Sonic-v0 --headless "$@"
