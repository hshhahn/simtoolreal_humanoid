#!/usr/bin/env bash
set -euo pipefail
_g1_lstm_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${_g1_lstm_workspace}/run_g1_wuji_lift_training.sh" \
  --agent rl_games_lstm_cfg_entry_point "$@"
