#!/usr/bin/env bash

# Replay an Isaac Gym checkpoint into a local, camera-free Three.js viewer.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/replay_checkpoint_gym.sh \
  --checkpoint PATH --output-dir DIR [options]

Options:
  --checkpoint PATH       Immutable rl_games .pth checkpoint to replay.
  --output-dir DIR        Directory in which videos/*.html is written.
  --task NAME             Hydra task config (default: SimToolReal).
  --train NAME            Hydra train config (default: SimToolRealMLPAsymmetricPPO).
  --frames N              Viewer frames (default: 600 = 10 s at 60 Hz).
  --num-envs N            Evaluation environments (default: 6).
  --venv PATH             Python venv (default: $SIMTOOLREAL_VENV or .venv).
  --override VALUE        Additional Hydra override; may be repeated.
  -h, --help              Show this help.

This command writes one local HTML trajectory and does not initialize Isaac
camera rendering or log to WandB. Select task/train configs and overrides that
match the checkpoint architecture.
EOF
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
CHECKPOINT=
OUTPUT_DIR=
TASK_CONFIG=SimToolReal
TRAIN_CONFIG=SimToolRealMLPAsymmetricPPO
FRAMES=600
NUM_ENVS=6
VENV_PATH=${SIMTOOLREAL_VENV:-$REPO_ROOT/.venv}
EXTRA_OVERRIDES=()

while (($#)); do
  case "$1" in
    --checkpoint) CHECKPOINT=${2:?}; shift 2 ;;
    --output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
    --task) TASK_CONFIG=${2:?}; shift 2 ;;
    --train) TRAIN_CONFIG=${2:?}; shift 2 ;;
    --frames) FRAMES=${2:?}; shift 2 ;;
    --num-envs) NUM_ENVS=${2:?}; shift 2 ;;
    --venv) VENV_PATH=${2:?}; shift 2 ;;
    --override) EXTRA_OVERRIDES+=("${2:?}"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n $CHECKPOINT ]] || { echo "Missing --checkpoint" >&2; usage >&2; exit 2; }
[[ -n $OUTPUT_DIR ]] || { echo "Missing --output-dir" >&2; usage >&2; exit 2; }
[[ -f $CHECKPOINT ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 2; }
[[ -f $VENV_PATH/bin/activate ]] || { echo "Venv not found: $VENV_PATH" >&2; exit 2; }
[[ $FRAMES =~ ^[1-9][0-9]*$ ]] || { echo "--frames must be positive" >&2; exit 2; }
[[ $NUM_ENVS =~ ^[1-9][0-9]*$ ]] || { echo "--num-envs must be positive" >&2; exit 2; }

CHECKPOINT=$(cd -- "$(dirname -- "$CHECKPOINT")" && pwd)/$(basename -- "$CHECKPOINT")
mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR=$(cd -- "$OUTPUT_DIR" && pwd)

# Activating also places the venv's ninja binary on PATH for gymtorch.
source "$VENV_PATH/bin/activate"

# A reused venv may have editable installs pointing at another checkout.
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/rl_games${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/${USER}_simtoolreal_torch_extensions}
export WANDB_MODE=disabled

# rl_games considers an episode of length N complete after N-1 reported steps.
# One extra episode step guarantees that the viewer receives FRAMES samples.
EPISODE_LENGTH=$((FRAMES + 1))
COMMAND=(
  python -u isaacgymenvs/train.py
  "task=$TASK_CONFIG" "train=$TRAIN_CONFIG"
  test=true "checkpoint=$CHECKPOINT"
  headless=true capture_video=false force_render=false wandb_activate=false
  "task.env.numEnvs=$NUM_ENVS"
  task.env.capture_viewer=true task.env.capture_viewer_freq=1
  "task.env.capture_viewer_len=$FRAMES" task.env.capture_viewer_once=true
  task.env.capture_video=false "task.env.episodeLength=$EPISODE_LENGTH"
  train.params.config.name=0_local-replay
  "train.params.config.player.games_num=$NUM_ENVS"
  train.params.config.player.deterministic=true
  "hydra.run.dir=$OUTPUT_DIR"
)
COMMAND+=("${EXTRA_OVERRIDES[@]}")

cd "$REPO_ROOT"
"${COMMAND[@]}"

LATEST_VIEWER=$(find "$OUTPUT_DIR/videos" -maxdepth 1 -type f -name '*.html' -printf '%T@ %p\n' \
  | sort -nr | head -1 | cut -d' ' -f2-)
[[ -n $LATEST_VIEWER ]] || { echo "Replay completed without writing an HTML viewer" >&2; exit 1; }
echo "Local viewer: $LATEST_VIEWER"
