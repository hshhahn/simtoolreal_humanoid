#!/usr/bin/env bash
#SBATCH --job-name=g1-wuji-verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
g1_job_root="${SIMTOOLREAL_WORKSPACE:-${SLURM_SUBMIT_DIR}}"
exec bash "${g1_job_root}/slurm/run_job.sh" verify
