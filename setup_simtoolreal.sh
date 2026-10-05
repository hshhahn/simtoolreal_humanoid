#!/usr/bin/env bash
set -euo pipefail
simtoolreal_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
simtoolreal_demo_policy=false
for simtoolreal_arg in "$@"; do
    case "${simtoolreal_arg}" in
        --with-demo-policy) simtoolreal_demo_policy=true ;;
        --help)
            printf 'Usage: %s [--with-demo-policy]\n' "$0"
            printf 'Install the pinned Isaac Sim environment. The optional flag downloads the original arm demo policy.\n'
            exit 0 ;;
        *) printf 'Unknown argument: %s\n' "${simtoolreal_arg}" >&2; exit 2 ;;
    esac
done
export UV_CACHE_DIR="${UV_CACHE_DIR:-${simtoolreal_workspace}/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${simtoolreal_workspace}/.tools/python}"
if [[ -x /snap/astral-uv/current/bin/uv ]]; then
    simtoolreal_uv=/snap/astral-uv/current/bin/uv
else
    simtoolreal_uv="$(command -v uv || true)"
fi
if [[ -z "${simtoolreal_uv}" ]]; then
    printf 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/\n' >&2
    exit 1
fi
if [[ ! -f "${simtoolreal_workspace}/simtoolreal/pyproject.toml" ]]; then
    printf 'The modified simtoolreal source is missing. Clone the complete simtoolreal_humanoid repository.\n' >&2
    exit 1
fi
cd -- "${simtoolreal_workspace}/simtoolreal"
if [[ ! -x .venv_isaacsim/bin/python ]]; then
    "${simtoolreal_uv}" --no-config venv .venv_isaacsim --python 3.11.16 --seed
fi
simtoolreal_python=.venv_isaacsim/bin/python
"${simtoolreal_uv}" --no-config pip install --python "${simtoolreal_python}" \
    'torch==2.7.0' 'torchvision==0.22.0' \
    --index-url https://download.pytorch.org/whl/cu128
"${simtoolreal_uv}" --no-config pip install --python "${simtoolreal_python}" \
    --constraint "${simtoolreal_workspace}/isaacsim-constraints.txt" \
    --extra-index-url https://pypi.nvidia.com \
    -e ./rl_games/ 'isaaclab[isaacsim,all]==2.3.2.post1' \
    omegaconf hydra-core 'gym==0.23.1' scipy yourdfpy requests tqdm tyro \
    'imageio[ffmpeg]' wandb termcolor 'setuptools<81' \
    'wheel==0.45.1' 'chardet==5.2.0'
# Upstream requires the typing override for tyro's NoExtraItems import.
# The demo additionally needs the GUI image and box-side APIs from Viser 1.1.1.
# This requires
# websockets 13.1, overriding Isaac Sim's 12.0 metadata pin after installation.
"${simtoolreal_uv}" --no-config pip install --python "${simtoolreal_python}" \
    'coacd==1.0.5' 'typing_extensions==4.15.0' \
    'viser==1.1.1' 'websockets==13.1'
# The root dependency metadata describes legacy Isaac Gym; follow the upstream
# Isaac Sim instructions and register the local packages without those deps.
"${simtoolreal_uv}" --no-config pip install --python "${simtoolreal_python}" -e . --no-deps
if [[ "${simtoolreal_demo_policy}" == true ]] && \
    [[ ! -f pretrained_policy/config.yaml || ! -f pretrained_policy/model.pth ]]; then
    "${simtoolreal_python}" download_pretrained_policy.py
fi
printf '\nSetup complete. Activate with: source %s/activate_simtoolreal.sh\n' \
    "${simtoolreal_workspace}"
