#!/usr/bin/env bash
#SBATCH --job-name=g1-wuji-sapg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail
exec bash "${SIMTOOLREAL_WORKSPACE:-${SLURM_SUBMIT_DIR}}/slurm/run_sapg.sh" "$@"
