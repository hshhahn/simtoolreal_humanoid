# Slurm setup

Install this checkout on persistent shared storage accessible from the login and GPU nodes. The scripts run one simulator and policy process on one allocated GPU. Four A6000 GPUs can run four independent jobs, including paired MLP and LSTM runs at two seeds.

## Inspect the cluster

Use the cluster's documentation to select a GPU partition, account, and GPU request syntax. Inside a short GPU allocation, inspect:

```bash
cat /etc/os-release
ldd --version
nvidia-smi
```

The native installation targets Ubuntu 22.04/24.04. Check the node against [Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html) and [Python installation requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html). NVIDIA lists Linux driver 580.65.06 as a tested baseline for this release. A Python venv cannot replace incompatible system libraries or a host GPU driver.

If the cluster requires Apptainer or has incompatible system libraries, prepare a compatible Ubuntu container, run installation and verification inside it, and adapt the batch entrypoint to use `apptainer exec --nv`. Bind the checkout at a consistent path and create the venv using the container's userspace. The host still supplies the GPU driver. Container construction and GPU library exposure depend on the cluster; these batch templates use native execution by default.

## Install the pinned environment

On a node with the compatible userspace and outbound network access:

```bash
git clone https://github.com/hshhahn/simtoolreal_humanoid.git
cd simtoolreal_humanoid
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if the cluster does not provide it, then run:

```bash
./setup_g1_wuji_sonic.sh
```

This installs Python 3.11.16, the pinned Isaac Sim/Lab stack, editable vendored rl_games, ONNX tools, and TensorBoard. It downloads the SONIC 1.1 revision recorded in `SOURCE_PROVENANCE.json` and runs the CPU controller contract check. The committed robot assets make third-party source downloads optional.

Allow storage for the Python environment, package downloads, 1.3 GB of model files, and future checkpoints. Installation uses checkout-local `.cache/` and `.tools/` directories unless `UV_CACHE_DIR` or `UV_PYTHON_INSTALL_DIR` are supplied. Concurrent training jobs use separate node-local Kit caches while writing results to shared storage.

The installer overrides Isaac Sim's older `typing_extensions` and `websockets` metadata pins to match our working Tyro/Viser setup. The root project is registered with `--no-deps` because its dependency metadata describes the legacy Isaac Gym environment. Do not run an unconstrained `pip install -e simtoolreal` afterward.

If compute nodes cannot reach the Internet, download/install on a compatible node or in the chosen container first. Copy the model files and make the environment visible on shared storage. Isaac extensions may also need initial network access; verify the GPU job under the actual compute-node network restrictions.

## Verify the allocated GPU

Run submissions from the checkout root. Create the scheduler log directory before submitting because Slurm opens the log files before the script starts:

```bash
mkdir -p slurm_logs
sbatch --partition=YOUR_GPU_PARTITION --account=YOUR_ACCOUNT slurm/verify.sbatch
```

Omit `--account` if the cluster supplies it by default. Adjust `--gres=gpu:1` to the site's GPU syntax, such as `--gres=gpu:a6000:1` or `--gpus=1`, without adding conflicting requests. The template requests eight CPU cores, 32 GB host memory, and 30 minutes.

Verification checks package versions, import of our vendored rl_games, CUDA computation, the SONIC/hand contract, policy exploration and LSTM memory resets, checkpoint aliases, and the physics guard. It then runs four simulated environments for 500 steps and two PPO iterations of each architecture at 128 environments. Reports and smoke-run checkpoints are saved under `g1_wuji_runs/<job-id>_0_verify_mlp_seed42/` by default. Set `SMOKE_PPO=0` when submitting to skip the PPO smoke runs.

The template respects Slurm's `CUDA_VISIBLE_DEVICES`. It uses `cuda:0` within the allocation and leaves `multi_gpu=False`.

## Submit training

The templates default to the marker lift task, 128 environments, seed 42, and 40,000 total iterations. Submit separate MLP and LSTM jobs after verification:

```bash
sbatch --partition=YOUR_GPU_PARTITION --account=YOUR_ACCOUNT \
  --export=ALL,ARCH=mlp,MAX_EPOCHS=40000 slurm/train.sbatch

sbatch --partition=YOUR_GPU_PARTITION --account=YOUR_ACCOUNT \
  --export=ALL,ARCH=lstm,MAX_EPOCHS=40000 slurm/train.sbatch
```

For four independent jobs, with an MLP/LSTM pair at seed 42 and another at seed 43:

```bash
sbatch --partition=YOUR_GPU_PARTITION --account=YOUR_ACCOUNT \
  --array=0-3%4 --export=ALL,ARCH=compare,SEED=42,MAX_EPOCHS=40000 \
  slurm/train.sbatch
```

Array index 0 is MLP/42, index 1 LSTM/42, index 2 MLP/43, and index 3 LSTM/43. This schedules separate PPO runs. Each job requests one GPU and writes its own output directory.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ARCH` | `mlp` | `mlp`, `lstm`, or `compare` for paired array runs |
| `SEED` | `42` | Base seed; ordinary arrays add the task index |
| `NUM_ENVS` | `128` | Parallel simulation environments |
| `MAX_EPOCHS` | `40000` | Total PPO iterations, including restored iterations |
| `MINIBATCH_SIZE` | `2048` | Actor and privileged critic minibatch size |
| `TASK` | `Isaacsimenvs-G1-Wuji-Sonic-Lift-v0` | Registered task ID |
| `CHECKPOINT` | unset | Absolute path to a checkpoint with the matching architecture |
| `CHECKPOINT_LOAD_MODE` | `resume` | Restore training state, or use `weights` for a fresh optimizer |
| `RUN_DIR` | generated per job | Persistent output path; give every job a unique path |
| `SIMTOOLREAL_WORKSPACE` | submission directory | Checkout path used by the batch entrypoint |

With the 16-step rollout horizon, `NUM_ENVS * 16` must be divisible by `MINIBATCH_SIZE`. The LSTM minibatch must also be divisible by sequence length 16. Keep the defaults for the initial migration so the experiment configuration matches the workstation.

To resume a matching checkpoint, add these variables to a single-job submission:

```bash
sbatch --partition=YOUR_GPU_PARTITION --account=YOUR_ACCOUNT \
  --export=ALL,ARCH=mlp,MAX_EPOCHS=40000,CHECKPOINT=/shared/path/model.pth \
  slurm/train.sbatch
```

A checkpoint at iteration 10,000 with `MAX_EPOCHS=40000` runs up to iteration 40,000. Existing workstation checkpoints are not uploaded to Git; transfer a chosen checkpoint separately if you want to resume it.

## Read results

Scheduler logs go to `slurm_logs/`. Each job writes `runtime.json` with package versions and GPU details. Its Hydra run directory is under `g1_wuji_runs/`; the rl_games experiment folder below it contains `nn/` checkpoints, `best/model.pth`, and `summaries/` TensorBoard events.

On a host that can access the results:

```bash
source ./activate_simtoolreal.sh
tensorboard --logdir ../g1_wuji_runs --host 127.0.0.1 --port 6006
```

Use the cluster's permitted SSH tunnel or dashboard service to view TensorBoard. A rising lift reward can represent throwing a marker, so review goal success and rollouts as well as total reward. The reward formulation is unchanged by these deployment scripts.
