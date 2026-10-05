#!/usr/bin/env bash
set -euo pipefail
_g1_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_g1_workspace}/activate_simtoolreal.sh"
python isaacsimenvs/tests/test_sonic_wuji_contract.py
python isaacsimenvs/tests/test_g1_wuji_sonic.py "$@"
