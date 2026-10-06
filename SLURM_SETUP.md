# Slurm setup

Install this checkout on persistent shared storage accessible from the login and GPU nodes. SAPG can train one policy across four allocated GPUs. The separate PPO scripts run one simulator and policy process per GPU, including paired MLP and LSTM runs at two seeds.

## Cornell fang installation

This checkout was installed and passed headless verification on
`fang-compute-01` in allocation `991133` on 2026-10-05. All four allocated
48 GB A6000s passed CUDA computation. The node uses Ubuntu 24.04.4,
glibc 2.39, and NVIDIA driver 570.124.06. See [VALIDATION.txt](VALIDATION.txt)
for the package versions and verification evidence.

Run setup, downloads, CPU checks, and simulation inside Slurm allocations.
Use the login node to dispatch `srun` or `sbatch`. To open a shell with one
GPU in the existing allocation and activate the installed environment:

```bash
srun --jobid=991133 --overlap --exact --ntasks=1 --cpus-per-task=8 \
  --gpus-per-task=1 --pty bash
cd /home/sh2776/simtoolreal_humanoid
source ./activate_simtoolreal.sh
```

The installed uv is at `.tools/uv/uv`; setup and activation find it
automatically. This Python environment is on persistent shared storage.

For future allocations, submit from the checkout root. These site-specific
commands request one A6000 per process:

```bash
mkdir -p slurm_logs
sbatch --partition=fang --account=fang --gres=gpu:nvidia_rtx_a6000:1 \
  --export=ALL,ARCH=mlp,MAX_EPOCHS=40000 slurm/train.sbatch
sbatch --partition=fang --account=fang --gres=gpu:nvidia_rtx_a6000:1 \
  --export=ALL,ARCH=lstm,MAX_EPOCHS=40000 slurm/train.sbatch
```

Alternatively, submit the four independent MLP/LSTM runs at seeds 42/43:

```bash
sbatch --partition=fang --account=fang --gres=gpu:nvidia_rtx_a6000:1 \
  --array=0-3%4 --export=ALL,ARCH=compare,SEED=42,MAX_EPOCHS=40000 \
  slurm/train.sbatch
```

These submission commands request new allocations and may queue while
`991133` holds its GPUs. No long training jobs were submitted during setup.

## Distributed G1 SAPG

`slurm/run_sapg.sh` launches four `torchrun` ranks. Each rank runs G1 physics
and the frozen SONIC decoder on its own GPU. Actor and privileged-critic
gradients are averaged with NCCL; both networks are synchronized before
the first update. Environment seeds differ by rank.

Run this command from the login node to dispatch the entire run to the
existing four-GPU allocation:

```bash
srun --jobid=991133 --overlap --exact --ntasks=1 --cpus-per-task=32 \
  --gpus-per-task=4 bash -lc \
  'cd /home/sh2776/simtoolreal_humanoid && TASK=Isaacsimenvs-G1-Wuji-Sonic-v0 NUM_ENVS=24576 MINIBATCH_SIZE=98304 MAX_EPOCHS=1000000 bash slurm/run_sapg.sh'
```

For a future allocation:

```bash
sbatch --partition=fang --account=fang --gres=gpu:nvidia_rtx_a6000:4 \
  --export=ALL,TASK=Isaacsimenvs-G1-Wuji-Sonic-v0,NUM_ENVS=24576,MINIBATCH_SIZE=98304,MAX_EPOCHS=1000000 \
  slurm/sapg.sbatch
```

Counts in these launch commands are global. Hydra overrides passed after the
script name operate on each rank's configuration. Defaults use 6,144 envs,
a nominal minibatch of 24,576, and six exploration blocks of 1,024 envs per GPU.
`NUM_ENVS` must be divisible by `6 * GPUS`; `MINIBATCH_SIZE` must divide
`NUM_ENVS * 16` and be divisible by `16 * GPUS`. SAPG augments leader samples
with follower experience, so its final minibatch can exceed the nominal size.

The default 24,576 env count matches the paper's environment pool. A smaller
trial can use `NUM_ENVS=6144` with the same nominal global minibatch.
G1's articulated scene also uses substantial host RAM; the full pool measured
150.1 GiB peak sampled host usage in the 256 GiB allocation.

`RUN_DIR` chooses a fresh output directory. Only rank 0 writes the shared
checkpoint and TensorBoard summaries. Each rank has its own log, Hydra
metadata, and temporary Kit/asset caches. `resources.jsonl` records actual
GPU use and host memory; `resources.summary.json` gives measured peaks.
`distributed-verification.json` checks finite parameters and exact agreement
of actor/critic parameters and normalization buffers after training.

The two-epoch, 384-env distributed smoke test passed on allocation 991133:
all four simulator/RL devices were distinct, both networks and buffers matched,
and all 60 TensorBoard scalars were finite. Its evidence is under
`g1_wuji_runs/991133_sapg_smoke_4gpu/`.

Before the velocity-penalty update, the full-scale 12-epoch trial also passed: 24,576 envs total, nominal
minibatch 98,304, 1,086 finite TensorBoard scalars, all 468 saved checkpoint
tensors finite, and exact actor/critic parameter and buffer agreement.
All four GPUs reached 100% utilization. Peak sampled GPU usage was
19.4 / 18.0 / 17.4 / 17.9 GiB. Evidence:
`g1_wuji_runs/991133_sapg_fullscale_12ep/verification-summary.json`.

Marker-lift training with GitHub velocity-penalty commit `1598192`
ran as Slurm step `991133.99`, under
`g1_wuji_runs/991133_sapg_g1_full_1598192/`. Its artifacts and rendered
checkpoint remain available. The previous 100-epoch run completed
with the older reward; its artifacts remain under
`g1_wuji_runs/991133_sapg_marker_100ep/`.

The unsmoothed full tool-set run was launched as Slurm step `991133.148`, under
`g1_wuji_runs/991133_sapg_g1_all_tools_1598192/`, using
`Isaacsimenvs-G1-Wuji-Sonic-v0` and 100 samples from each of the 12
procedural shape distributions (1,200 objects across six tool families).
Initial positions/orientations and pose goals are randomized. All four GPUs
train one SAPG actor with an asymmetric critic, 24,576 environments, and a
nominal global minibatch of 98,304. The iteration ceiling remains 1,000,000.

The run initializes actor and critic weights from marker checkpoint 566
(222,560,256 global environment steps). It starts fresh optimizers, AMP
scaler, simulation/rollout state, and iteration counters. The complete
marker checkpoint and a smaller weights-only checkpoint are preserved
under `initialization/`; their hashes are in `checkpoint-info.json`.
The full-task physics check passed 300 steps across all 12 distributions
with zero falls, finite states, reward accounting, and isolated fault resets.
Startup validation passed at iteration 42: all four workers restored actor/critic weights, 3,264 TensorBoard scalars were finite, and peak sampled host memory was 161.02 GiB. See the run's `startup-verification.json`.

An earlier smoothed run resumed full training in Slurm step `991133.294` under
`g1_wuji_runs/991133_sapg_g1_all_tools_smoothed_20261006_123802/`.
It resumes checkpoint 7,700 (3,027,763,200 global environment steps), retaining actor/critic weights, normalization, actor optimizer, and iteration/frame counters. Simulation rollouts, recurrent state, episode statistics, and best-reward tracking start fresh under the changed command dynamics. The previous run's artifacts remain available; its training step was stopped.

Overrides are `env.sonic.smooth_right_arm_targets=true`, `env.action.arm_moving_average=0.1`, `env.action.dof_speed_scale=1.5`, and `env.action.hand_moving_average=0.1`. The right-arm target limit is 0.15 rad/s; legs, waist, and left arm retain direct SONIC outputs. Full training still uses all four GPUs, 24,576 environments, a nominal global minibatch of 98,304, the asymmetric critic, and an iteration ceiling of 1,000,000.

Validation passed seven CPU tests and a 500-step, 12-environment simulation with no standing falls. A same-checkpoint comparison over 500 steps and 24 environments per setting had no falls or physics failures. Mean actual right-arm joint speed fell from 1.49 to 0.42 rad/s, while task reward fell; the existing policy requires further training with the slower commands. Reports and source snapshots are in the new run directory. Its TensorBoard entry is `full_tools_smoothed` at the existing address.

The user requested training from scratch, so step `991133.294` was stopped at observed iteration 7,771. Its checkpoints and curves remain available. Fresh full-tool SAPG training launched as Slurm step `991133.315` under
`g1_wuji_runs/991133_sapg_g1_all_tools_smoothed_scratch_20261006_132146/`.
It loads no task-policy checkpoint: the actor, asymmetric critic, optimizers, normalization, recurrent/rollout state, and iteration/frame counters start fresh at iteration 0 and 0 environment steps. The existing frozen pretrained SONIC decoder and default standing latent reference are retained. The same four-GPU configuration, full 1,200-object randomized task, velocity penalties, and target smoothing apply. The run's `launch.command.sh` records the exact command without `--checkpoint`.

TensorBoard runs in Slurm step `991133.150` on `127.0.0.1:6006`
with `marker_lift`, `full_tools`, `full_tools_smoothed`, and `full_tools_smoothed_scratch` runs. Keep the existing SSH tunnel:

```bash
ssh -N -J sh2776@en-cc-unicorn-login-02.coecis.cornell.edu \
  -L 6006:127.0.0.1:6006 sh2776@fang-compute-01.cs.cornell.edu
```

Open http://localhost:6006 and select `full_tools_smoothed_scratch`. The original rendered
checkpoint remains available in Images under `marker_lift/rendered_checkpoints`.

The updated reward uses the seven right-arm joints with velocity scale 0.03
and the twenty right-hand joints with scale 0.003. SONIC latent-rate penalty
is zero; balance and left-arm velocities do not enter the manipulation penalty.
A 500-step, four-environment marker-lift test passed finite-state, reward
accounting, standing, and physics-fault reset checks. CPU tests also confirmed
that legacy reward calculations remain unchanged.

The launcher defaults to 1,000,000 epochs. Rolling recovery checkpoints are
written every 100 epochs to `0_g1_wuji_sonic11_sapg/last/model.pth`; the previous
file is retained as `model.pth.old`. Best-policy files remain under
`0_g1_wuji_sonic11_sapg/nn/`. The run's `training-plan.json`, source snapshots,
and per-rank Hydra files record the reward version and training settings.
Allocation 991133 ends on 2026-10-08 at 11:18:30 EDT; the epoch limit does not
extend the allocation. Continue from a saved checkpoint in a later allocation
if training is still needed.

Scene creation and physics initialization take about ten minutes at this scale;
steady-state epochs in the 12-epoch validation took roughly 5–6 seconds.
Checkpoint gathering occurs only when saving. Gradient diagnostics are collected
only when explicitly enabled, avoiding unnecessary CPU transfers.

In rl_games, one epoch is one rollout-and-update iteration: 16 steps from
24,576 environments, then two PPO passes. This generates 393,216 new global
environment steps per iteration. A target of 120 billion environment steps
corresponds to 305,176 iterations; the 1,000,000-epoch ceiling permits
393.216 billion steps. At the measured 5–6 seconds of compute per iteration,
120B needs about 18–21 uninterrupted days before checkpoint overhead.
The historical 100-iteration run averaged approximately 12.3 wall seconds
per iteration including frequent best-policy saves, which extrapolates to
44 days. These are early measurements, not a guarantee of long-run speed.


To validate a completed run, execute this inside a compute allocation:

```bash
source ./activate_simtoolreal.sh
python ../slurm/verify_sapg_run.py ../g1_wuji_runs/991133_sapg_g1_all_tools_1598192
```

The G1 configuration uses the released code's SAPG entropy scale 0.002.
The full task uses all six tool families, 1,200 procedural objects, randomized object starts and sampled pose goals. It retains SONIC standing initialization,
bounded hand means, and wider hand exploration. Timing delays and external perturbations follow the G1 configuration. It is an Isaac Sim G1
adaptation; the original SimToolReal experiments used Isaac Gym and a
different robot.

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
