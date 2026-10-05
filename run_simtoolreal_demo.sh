#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/activate_simtoolreal.sh"
exec python dextoolbench/eval_interactive_isaacsim.py \
    --config-path pretrained_policy/config.yaml \
    --checkpoint-path pretrained_policy/model.pth "$@"
