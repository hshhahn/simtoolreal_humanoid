#!/usr/bin/env bash
set -euo pipefail
_g1_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_g1_rebuild_assets=false
for _g1_arg in "$@"; do
    case "${_g1_arg}" in
        --rebuild-assets) _g1_rebuild_assets=true ;;
        --help)
            printf 'Usage: %s [--rebuild-assets]\n' "$0"
            printf 'Install the environment and pinned SONIC 1.1 model; use the committed robot assets by default.\n'
            exit 0 ;;
        *) printf 'Unknown argument: %s\n' "${_g1_arg}" >&2; exit 2 ;;
    esac
done
if [[ ! -x "${_g1_workspace}/simtoolreal/.venv_isaacsim/bin/python" ]]; then
    "${_g1_workspace}/setup_simtoolreal.sh"
fi
source "${_g1_workspace}/activate_simtoolreal.sh"
export HF_HOME="${_g1_workspace}/.cache/huggingface"
_g1_uv="${_g1_workspace}/.tools/uv/uv"
if [[ ! -x "${_g1_uv}" ]]; then
    _g1_uv=/snap/astral-uv/current/bin/uv
fi
if [[ ! -x "${_g1_uv}" ]]; then
    _g1_uv="$(command -v uv)"
fi
"${_g1_uv}" --no-config pip install --python "${SIMTOOLREAL_ROOT}/.venv_isaacsim/bin/python" \
    --constraint "${_g1_workspace}/isaacsim-constraints.txt" \
    onnx==1.23.1 onnxruntime==1.23.2 tensorboard==2.21.0

_g1_checkout() {
    local name="$1" url="$2" commit="$3"
    shift 3
    local destination="${_g1_workspace}/third_party/${name}"
    if [[ ! -d "${destination}/.git" ]]; then
        git clone --depth 1 --filter=blob:none --sparse "${url}" "${destination}"
    fi
    if ! git -C "${destination}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
        git -C "${destination}" fetch --depth 1 origin "${commit}"
    fi
    git -C "${destination}" checkout --detach "${commit}"
    git -C "${destination}" sparse-checkout set "$@"
}

if [[ "${_g1_rebuild_assets}" == true || \
      ! -f "${SIMTOOLREAL_ROOT}/assets/urdf/g1_wuji/g1_wuji.urdf" || \
      ! -f "${SIMTOOLREAL_ROOT}/assets/urdf/g1_wuji/wuji_actuators.json" ]]; then
    mkdir -p "${_g1_workspace}/third_party"
    _g1_checkout unitree_ros https://github.com/unitreerobotics/unitree_ros.git 5994d4faef0a9cadd3287f8de0199a67eeb2a259 robots/g1_description
    _g1_checkout wuji-description https://github.com/wuji-technology/wuji-description.git c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95 hand/body/urdf hand/body/meshes hand/body/mjcf hand/attachment/unitree-g1-attachment
    _g1_checkout GR00T-WholeBodyControl https://github.com/NVlabs/GR00T-WholeBodyControl.git b042411fae38ee4d1af9aac82a37a1f8d14d6dd0 gear_sonic/data/assets/robot_description/urdf/g1 gear_sonic/data/assets/robot_description/meshes/g1
    python "${SIMTOOLREAL_ROOT}/scripts/build_g1_wuji_assets.py"
fi

# Use an isolated CLI if hf is unavailable; keep the training SDK pinned.
if command -v hf >/dev/null 2>&1; then
    _g1_hf=(hf)
else
    _g1_hf=("${_g1_uv}" --no-config tool run --from huggingface_hub==1.30.0 hf)
fi
"${_g1_hf[@]}" download nvidia/GEAR-SONIC \
    --revision 6733128a3d8a523b1418b06bca3cdf61c8b0987f \
    --include 'sonic_v1_1/*' config.json LICENSE README.md \
    --local-dir "${_g1_workspace}/models/GEAR-SONIC"
(
    cd -- "${_g1_workspace}/models/GEAR-SONIC/sonic_v1_1"
    sha256sum --check "${_g1_workspace}/sonic11-model-sha256.txt"
)
python "${SIMTOOLREAL_ROOT}/isaacsimenvs/tests/test_sonic_wuji_contract.py"
printf 'G1 + dual Wuji + SONIC 1.1 installed. Run run_g1_wuji_lift_training.sh for the initial marker task.\n'
