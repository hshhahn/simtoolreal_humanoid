#!/usr/bin/env bash
set -euo pipefail
_g1_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_g1_workspace}/activate_simtoolreal.sh"
exec python isaacsimenvs/preview_g1_wuji.py "$@"
