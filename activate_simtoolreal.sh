#!/usr/bin/env bash
# Source this file to activate SimToolReal and enter the repository.
_simtoolreal_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SIMTOOLREAL_ROOT="${_simtoolreal_workspace}/simtoolreal"
if [[ -x "${_simtoolreal_workspace}/.tools/uv/uv" ]]; then
    export PATH="${_simtoolreal_workspace}/.tools/uv:${PATH}"
fi
if [[ ! -f "${SIMTOOLREAL_ROOT}/.venv_isaacsim/bin/activate" ]]; then
    printf 'SimToolReal environment is missing. Run setup_simtoolreal.sh first.\n' >&2
    return 1 2>/dev/null || exit 1
fi
source "${SIMTOOLREAL_ROOT}/.venv_isaacsim/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_CACHE_PATH="${OMNI_KIT_CACHE_PATH:-${_simtoolreal_workspace}/.cache/omniverse}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${_simtoolreal_workspace}/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${_simtoolreal_workspace}/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${_simtoolreal_workspace}/.tools/python}"
export PYTHONPATH="${SIMTOOLREAL_ROOT}/rl_games:${SIMTOOLREAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OMNI_KIT_CACHE_PATH}"
cd -- "${SIMTOOLREAL_ROOT}" || return 1
unset _simtoolreal_workspace
