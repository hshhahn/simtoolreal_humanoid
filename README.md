# SimToolReal humanoid

This repository contains our modified [SimToolReal](https://github.com/tylerlum/simtoolreal) environment for a free-standing Unitree G1 with two original Wuji hands. PPO predicts a frozen SONIC 1.1 controller's latent command and three right-hand commands jointly. Feed-forward PPO, LSTM PPO, and distributed SAPG policies are included.

The current humanoid training task uses all six tool families with randomized initial object poses and pose goals. Full four-GPU SAPG training starts from scratch with right-arm and hand target smoothing. This is an experimental G1/SONIC adaptation of SimToolReal. The fixed-KUKA comparison uses the original SHARPA dexterous hand and SAPG on the two-marker pool; the earlier parallel-gripper baseline is also included. Robot assets, environment code, agent configurations, and setup scripts are included. SONIC weights are downloaded from their pinned upstream revision during humanoid setup.

## Installation

The working stack uses Python 3.11.16, Isaac Sim 5.1.0.0, Isaac Lab 2.3.2.post1, PyTorch 2.7.0 with CUDA 12.8, NumPy 1.26.0, and vendored rl_games 1.6.1. Start with Ubuntu 22.04 or 24.04 and an NVIDIA RTX GPU. Check the compute node's driver and system libraries against [Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html). A6000 GPUs have RTX cores; the setup must still pass the GPU verification on the cluster.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone https://github.com/hshhahn/simtoolreal_humanoid.git
cd simtoolreal_humanoid
./setup_g1_wuji_sonic.sh
source ./activate_simtoolreal.sh
```

The installer creates `simtoolreal/.venv_isaacsim`, uses the committed G1/Wuji URDFs and meshes, downloads about 1.3 GB of SONIC 1.1 files into `models/GEAR-SONIC/`, and checks the native FSQ grid, decoder equivalence, and hand synergies. Installation needs network access to PyPI, NVIDIA, PyTorch, and Hugging Face. Cache paths default to this checkout and can be overridden with environment variables.

The code remains under `simtoolreal/`, with workspace launchers at the repository root. Preserve this layout: asset and model paths are derived from it. The root setup script recognizes an existing source tree without requiring nested Git metadata.

To rebuild the robot assets from the pinned Unitree, Wuji, and NVIDIA source repositories:

```bash
./setup_g1_wuji_sonic.sh --rebuild-assets
```

The original arm demo policy is optional and separate from humanoid training. Install it with `./setup_simtoolreal.sh --with-demo-policy` if needed.

## Controller and policy

| Setting | Current configuration |
| --- | --- |
| Robot | 29 G1 joints plus 20 joints per Wuji hand |
| Policy action | 64 SONIC latent coordinates plus 3 right-hand commands |
| Body command | Clamp to normalized bounds, then quantize to SONIC's native 32-level FSQ grid before decoding |
| Native FSQ values | Step 1/16, range from -1 to 0.9375 |
| Hand command | Thumb rotation, thumb bending, shared bending of the other four fingers |
| Hand angle mapping | Normalized commands map to physical joint limits; non-thumb abduction stays neutral |
| Actor input | 432 values, including current-frame observations and the previous 67-dimensional action |
| Privileged critic input | 454 values |
| SONIC decoder history | 10 frames of 93-dimensional proprioception; 930 history values plus 64 latent values |
| PPO MLP | 1024, 512, 256, 128 units with ELU |
| LSTM PPO | One 1024-unit layer before the PPO MLP; sequence length 16 |
| SAPG | LSTM 1024 + MLP 1024, 1024, 512, 512; six exploration groups; asymmetric MLP critic |
| Simulation | 200 Hz physics, 50 Hz control, 128 parallel environments by default |

The decoder is frozen. PPO evaluates the continuous policy samples for its likelihood calculation; FSQ projection happens at the environment's decoder boundary. The left hand is held open. The right-hand policy starts with standard deviation 0.5, independently of the body latent exploration.

The standing reference is encoded with the released SONIC encoder and used to initialize the policy's body mean. It does not anchor the robot to the floor.

## Task and rewards

`Isaacsimenvs-G1-Wuji-Sonic-Lift-v0` uses two procedural marker asset instances, a fixed initial pose at `(-0.23, 0.32, 0.54)` metres, and an elevated target pose at `(-0.23, 0.32, 0.70)` metres. The table surface is 0.50 m high. Both wrists begin above the table.

The task retains SimToolReal's fingertip approach, lift shaping, one-shot lift bonus, object keypoint progress, and goal reward. Its motion penalty uses the original measured-joint-velocity formulation and weights: `-0.03 * sum(abs(right_arm_joint_velocity)) - 0.003 * sum(abs(right_hand_joint_velocity))`. The arm group contains the seven right shoulder/elbow/wrist joints; the hand group contains all 20 right Wuji joints. Leg, waist, left-arm and left-hand joints are outside this manipulation penalty. Body control and observations still cover all 29 G1 body joints. The added latent-change penalty is disabled by default (`env.sonic.latent_rate_penalty=0.0`); its optional setting remains for reproducing older runs. Fall penalties remain active. There is no upright reward. Falls and invalid physics terminate the affected environment.

Objects use the existing four selected bounding-box corners for the goal formulation. The marker box is `(0.141, 0.03025, 0.0271)` metres; keypoint scale 1.5 and position tolerance 0.075 produce an effective 0.1125 m threshold. Success requires ten accumulated qualifying steps. The lift reward alone does not establish a grasp and can reward tossing an object.

The complete settings are in [env_cfg.py](simtoolreal/isaacsimenvs/tasks/g1_wuji_sonic/env_cfg.py), [env.py](simtoolreal/isaacsimenvs/tasks/g1_wuji_sonic/env.py), and the inherited [SimToolReal configuration](simtoolreal/isaacsimenvs/tasks/simtoolreal/simtoolreal_env_cfg.py). The full task, `Isaacsimenvs-G1-Wuji-Sonic-v0`, uses all six tool families (hammer, screwdriver, marker, spatula, eraser, brush), randomized initial object poses, and sampled position/orientation goals. Its default pool contains 1,200 procedural objects: 100 size/density samples from each of the 12 shape distributions. The lift curriculum retains its explicit two-marker pool.

## Manipulation target smoothing

Enable dex-hand-style limits with these Hydra overrides:

    env.sonic.smooth_right_arm_targets=true
    env.action.arm_moving_average=0.1
    env.action.dof_speed_scale=1.5
    env.action.hand_moving_average=0.1

The seven right-arm joint targets approach the decoded SONIC pose with a maximum raw change of 1.5 times the policy timestep, followed by the 0.1 blend. Their effective target speed is limited to 0.15 rad/s (0.003 rad per 50 Hz update). This preserves absolute SONIC pose requests; it does not reinterpret latent coordinates as joint velocities. Finger targets move 10% toward the requested synergy pose each update. Actual joint speeds can exceed target speeds while tracking a command.

Leg, waist, and left-arm outputs retain the direct SONIC path. Its last-action history records the filtered right-arm commands actually applied. Smoothing is an explicit run setting and is restored by checkpoint video replay; older saved configurations keep their original unsmoothed arm behavior.

## Verification and training

The two-hand full-tool task is `Isaacsimenvs-G1-Wuji-Sonic-Bimanual-v0`.
It jointly predicts SONIC64, right-hand3, and left-hand3 (70 actions). Each
hand uses independent thumb opposition, thumb bending, and shared bending
of the other four fingers. There is one tool per environment, which can be
handled by either hand or both. Initial object positions span both sides of
the table; all six tool families and the 1,200-object pool are retained.

The actor receives 457 current-frame values, including all ten fingertips,
left and right palm poses, and the previous 70-dimensional policy action.
The asymmetric critic receives 490 values. SAPG additionally conditions on
its exploration coefficient. The SONIC decoder retains its own ten-frame
history. Both arms use the target limiter and both hands use finger-target
smoothing. The approach term covers all ten fingertips; the existing 0.03
and 0.003 velocity coefficients apply to the 14 arm joints and 40 finger
joints. Other reward coefficients, keypoint goals, and fall handling follow
the existing G1 task. The episode can continue while either hand remains
near the tool; it does not require simultaneous two-hand contact.

```bash
./verify_g1_wuji_sonic.sh \
  --task Isaacsimenvs-G1-Wuji-Sonic-Bimanual-v0 \
  --num_envs 12 --steps 500 --headless

./run_g1_wuji_bimanual_sapg_training.sh \
  'hydra.run.dir=../g1_wuji_runs/${now:%Y%m%d_%H%M%S}_bimanual_sapg'
```

The local SAPG launcher defaults to 6,144 environments, six exploration
groups, 24,576-sample actor/critic minibatches, and 40,000 iterations. It
starts a fresh 70-action policy. Its checkpoints and curves are under the
run's `0_g1_wuji_bimanual_sapg/nn/`, `best/model.pth`, and `summaries/`.
Reduce environment, block, and both minibatch counts together if needed.
The Slurm launcher also accepts `TASK=Isaacsimenvs-G1-Wuji-Sonic-Bimanual-v0`.

On a GPU workstation or inside a GPU allocation:

```bash
./verify_g1_wuji_sonic.sh \
  --task Isaacsimenvs-G1-Wuji-Sonic-Lift-v0 \
  --num_envs 4 --steps 500 --headless
```

For Slurm, follow [SLURM_SETUP.md](SLURM_SETUP.md). It includes verification, separate MLP/LSTM training submissions, and a single SAPG run distributed across four GPUs. [CODEX_SETUP_PROMPT.txt](CODEX_SETUP_PROMPT.txt) is a prompt for the agent setting up the server.

The fresh local installation passed the controller checks, 13 CPU regression tests, 500 simulation steps, and two PPO iterations of each architecture with checkpoint and TensorBoard output. See [VALIDATION.txt](VALIDATION.txt); the native `fang-compute-01` Slurm installation also passed these checks on allocation `991133`.

To train directly on an allocated GPU, run one of these from the checkout root:

```bash
./run_g1_wuji_lift_training.sh \
  agent.params.config.max_epochs=40000 \
  'hydra.run.dir=../g1_wuji_runs/${now:%Y%m%d_%H%M%S}_mlp'

./run_g1_wuji_lift_lstm_training.sh \
  agent.params.config.max_epochs=40000 \
  'hydra.run.dir=../g1_wuji_runs/${now:%Y%m%d_%H%M%S}_lstm'
```

To train one SAPG policy using all four GPUs in the current allocation:

```bash
srun --jobid=991133 --overlap --exact --ntasks=1 --cpus-per-task=32 \
  --gpus-per-task=4 bash -lc \
  'cd /home/sh2776/simtoolreal_humanoid && TASK=Isaacsimenvs-G1-Wuji-Sonic-v0 MAX_EPOCHS=1000000 bash slurm/run_sapg.sh \
    env.sonic.smooth_right_arm_targets=true \
    env.action.arm_moving_average=0.1 env.action.dof_speed_scale=1.5 \
    env.action.hand_moving_average=0.1'
```

The SAPG launcher defaults to 24,576 environments and a nominal minibatch of
98,304 samples **across all four GPUs**. Each rank owns 6,144 environments
and a 24,576-sample minibatch. Override global counts with `NUM_ENVS` and
`MINIBATCH_SIZE`; six exploration groups run on each rank. The SAPG
configuration follows the released SimToolReal code's entropy scale 0.002.
The launcher defaults to 1,000,000 epochs and writes a rolling recovery
checkpoint every 100 epochs, keeping the previous recovery file as well.
This is a G1/SONIC adaptation on Isaac Sim. The original paper used Isaac Gym
and a different robot.

Rank logs, TensorBoard curves, checkpoints, resource measurements, and final
actor/critic synchronization checks are written under `g1_wuji_runs/`.
Before the velocity-penalty update, the full 24,576-env setup passed 12
epochs on all four A6000s, with finite checkpoints and 1,086 finite
TensorBoard scalars. Peak sampled memory was
19.4 GiB on the busiest GPU and 150.1 GiB of host RAM. The launcher rejects reused output directories and stops the training process
if host memory reaches 90% of the Slurm allocation's limit.
See [SLURM_SETUP.md](SLURM_SETUP.md) for configuration and batch submission.

Omit `--checkpoint` to start the task actor and asymmetric critic from their default initialization, with fresh optimizers, normalization, and iteration counters. The frozen pretrained SONIC controller and standing latent reference remain part of the G1 control architecture.

`max_epochs` is the total PPO iteration limit, including iterations restored from a checkpoint. Use `--checkpoint /absolute/path/model.pth` to resume with the matching architecture, or add `--checkpoint_load_mode weights` to start a new optimizer from model weights.

Each Hydra run directory contains the rl_games experiment subdirectory. Checkpoints are under `nn/`, the best checkpoint alias is `best/model.pth`, and training curves are TensorBoard event files under `summaries/`:

```bash
source ./activate_simtoolreal.sh
tensorboard --logdir ../g1_wuji_runs --host 127.0.0.1 --port 6006
```

## KUKA + SHARPA dexterous-hand SAPG comparison

This experiment uses the upstream `Isaacsimenvs-SimToolReal-Direct-v0` environment directly, including its original KUKA iiwa14 arm, left SHARPA hand, table, robot placement, reset and goal sampling, joint control, reward coefficients, bounding-box keypoints, domain randomization, and termination logic. Only the procedural object pool is restricted to the same two marker variants used in the humanoid experiment. It uses direct joint commands: seven arm actions plus all 22 SHARPA hand actions. No SONIC controller or three-command finger mapping is used. The actor observation has 140 values and the asymmetric critic has 162.

The purpose is to compare manipulation behavior, including object flicking, on a fixed arm with the original dexterous hand. This is the upstream Isaac Sim port; it retains the known lift-reference difference relative to the paper's Isaac Gym implementation. The two-marker task restriction also differs from the full paper training distribution.

Install the base environment with `./setup_simtoolreal.sh`, then start training from the checkout root on an allocated GPU:

```bash
./run_kuka_sharpa_sapg_training.sh \
  'hydra.run.dir=../kuka_gripper_runs/${now:%Y%m%d_%H%M%S}_sharpa_sapg'
```

The launcher reuses upstream `SimToolRealSAPG.yaml`: LSTM 1024, actor and asymmetric critic MLPs `[1024, 1024, 512, 512]`, 16-step rollouts, two PPO passes, six exploration groups, and a fresh policy trained for 40,000 iterations. Defaults are 24,576 environments, 4,096 environments per group, and actor/critic minibatches of 98,304. Reduce all four sizes together when required by host RAM or GPU memory, for example:

```bash
./run_kuka_sharpa_sapg_training.sh \
  env.scene.num_envs=6144 \
  agent.params.config.expl_coef_block_size=1024 \
  agent.params.config.minibatch_size=24576 \
  agent.params.config.central_value_config.minibatch_size=24576 \
  'hydra.run.dir=../kuka_gripper_runs/${now:%Y%m%d_%H%M%S}_sharpa_sapg'
```

Checkpoints, the best checkpoint alias, and training curves are written under each run's `0_kuka_sharpa_sapg/nn/`, `0_kuka_sharpa_sapg/best/model.pth`, and `0_kuka_sharpa_sapg/summaries/`. Ten-second pose viewers are written to the run's `interactive_viewer/` directory. TensorBoard can display both KUKA experiments:

```bash
source ./activate_simtoolreal.sh
tensorboard --logdir ../kuka_gripper_runs --host 127.0.0.1 --port 6007
```

## KUKA parallel-gripper LSTM baseline

`Isaacsimenvs-Kuka-Parallel-Gripper-Lift-v0` uses the original fixed KUKA iiwa14 arm with a parametric simulation gripper. The seven arm joints, original starting joint values, base at `(0, 0.8, 0)` metres, arm drive gains, and narrow table are retained. The table root is at `0.38 ± 0.01` m and its surface at `0.53 ± 0.01` m. Only the object pool is simplified to the two procedural marker instances.

The task inherits the upstream Isaac Sim port's randomized resets, goal sampling, rewards, bounding-box keypoints, termination/curriculum, action smoothing, domain randomization, and 60 Hz control / 120 Hz physics. Full resets sample absolute goals; later goal changes use delta position/orientation goals. The gripper has two opposing prismatic joints, driven by one shared command, with a 0–100 mm opening. Its adapter and box/pad geometry are simulation estimates, not a specific commercial gripper model.

Reward formulas and coefficients match the legacy Isaac Gym implementation for identical inputs. The gripper supplies two pad distances instead of five fingertip distances, and two prismatic hand velocities instead of 22 angular hand velocities. The upstream Isaac Sim port also includes object spawn-height noise in the lift reference, while Isaac Gym excludes it; with the inherited reset distribution the reference differs by up to 2 cm. These adaptations mean the experiment is not an exact reproduction of the original robot's reward magnitudes or physics.

The policy predicts seven arm velocity-delta commands and one normalized jaw-opening command. Actor/critic observations have 71/90 values. The original baseline network is a 1024-unit, one-layer LSTM followed by an ELU MLP with `[1024, 1024, 512, 512]` units; the asymmetric critic is feed-forward. Sequence length is 16. The launcher uses 2,048 environments and scales minibatches to preserve four minibatches per rollout. These settings require no SONIC model files.

For a fresh checkout, install the base environment with `./setup_simtoolreal.sh`. Then verify and train on an allocated GPU:

```bash
source ./activate_simtoolreal.sh
python isaacsimenvs/tests/test_kuka_parallel_gripper.py \
  --num_envs 32 --steps 1200 --headless
python isaacsimenvs/tests/test_kuka_parallel_grasp.py --headless
cd ..

./run_kuka_parallel_gripper_lstm_training.sh \
  agent.params.config.max_epochs=40000 \
  'hydra.run.dir=../kuka_gripper_runs/${now:%Y%m%d_%H%M%S}_lstm'
```

The second test follows a scripted grasp trajectory with fixed diagnostic poses and randomization disabled. Those diagnostic settings do not enter training; a passing test confirms physical grasp capability, not learned-policy success. The first test compares the task settings against the original SimToolReal configuration and verifies finite simulation, coupled jaw targets, and the goal-success signal.

Checkpoints and TensorBoard events use the same `nn/`, `best/model.pth`, and `summaries/` layout inside each run's `0_kuka_parallel_gripper_lstm/` directory:

```bash
source ./activate_simtoolreal.sh
tensorboard --logdir ../kuka_gripper_runs --host 127.0.0.1 --port 6007
```

To scale environment count, override both actor and critic minibatches. For example, use `env.scene.num_envs=4096 agent.params.config.minibatch_size=16384 agent.params.config.central_value_config.minibatch_size=16384`. The command can run directly within a Slurm GPU allocation; the existing `slurm/train.sbatch` is configured for the G1 comparison.

For the upstream SAPG training recommendation, use:

```bash
./run_kuka_parallel_gripper_sapg_training.sh \
  'hydra.run.dir=../kuka_gripper_runs/${now:%Y%m%d_%H%M%S}_sapg'
```

This launcher reuses `simtoolreal/isaacsimenvs/cfg/train/SimToolRealSAPG.yaml` through the task's `rl_games_sapg_cfg_entry_point`. It starts a fresh policy with a 1024-unit LSTM and `[1024, 1024, 512, 512]` actor MLP, a separate feed-forward asymmetric critic with the same MLP widths, six exploration groups, coefficient-conditioned action variance, entropy exploration rewards, and leader/follower experience sharing. The defaults are 24,576 environments, 4,096 environments per group, 98,304-transition actor/critic minibatches, 16-step rollouts, two PPO passes, and a 40,000-iteration limit. Pose viewers are captured without enabling cameras. Checkpoints and events use the `0_kuka_parallel_gripper_sapg/nn/` and `0_kuka_parallel_gripper_sapg/summaries/` directories under the run.

When RAM or GPU memory requires a smaller run, scale the environment count, exploration block, and both minibatches together while keeping six groups. For example:

```bash
./run_kuka_parallel_gripper_sapg_training.sh \
  env.scene.num_envs=6144 \
  agent.params.config.expl_coef_block_size=1024 \
  agent.params.config.minibatch_size=24576 \
  agent.params.config.central_value_config.minibatch_size=24576 \
  'hydra.run.dir=../kuka_gripper_runs/${now:%Y%m%d_%H%M%S}_sapg'
```

## Wuji marker grasp capability

The current three hand commands can physically grasp both procedural markers. A controlled-wrist diagnostic demonstrated a 15 cm wrist lift, five seconds of object retention above 10 cm, and release on opening. An open-hand control retained neither marker. This establishes finger-control capability at suitable wrist poses; it does not demonstrate grasping by the SONIC/PPO policy.

The diagnostic uses the production right-hand meshes, collisions, joint limits, actuator settings, friction setup, and three-command mapping. A driven Cartesian wrist fixture isolates the hand from humanoid balance and SONIC. Its six positioning joints are diagnostic equipment; the training policy remains 64 SONIC commands plus three hand commands. A six-command comparison can independently bend each non-thumb finger, but is not enabled in training.

Replay the saved successful wrist poses on an allocated GPU:

```bash
source ./activate_simtoolreal.sh
python isaacsimenvs/tests/test_wuji_marker_grasp.py --headless \
  --mode 3 --batches 1 \
  --candidate-file isaacsimenvs/tests/data/wuji_marker_grasp_3dof.json \
  --output ../visualizations/wuji_marker_capability/replay
```

Results and recorded physical poses are written to the output directory. `report.json` records `verified_hand_dof`, successful grasps per marker, hold/release criteria, and the open-hand control. A process exit code of zero alone does not mean a grasp was found; `verified_hand_dof` must be `3`. Omitting the candidate file performs a pose/closure search; `--mode auto` tests three commands first and searches with six only if that search does not pass both markers.

## Source and licenses

The modified upstream base is recorded in [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json). [environment-reference.json](environment-reference.json) records the packages in the original working environment; selected installation constraints are in [isaacsim-constraints.txt](isaacsim-constraints.txt). The setup uses documented dependency overrides required by that working stack rather than installing the root package's legacy Isaac Gym dependencies.

SimToolReal's MIT license is retained in [LICENSE](LICENSE). Robot asset provenance and the NVIDIA, Wuji, and Unitree license notices are included under [simtoolreal/assets/urdf/g1_wuji/](simtoolreal/assets/urdf/g1_wuji/). Vendored rl_games retains its own license. Downloaded SONIC weights retain NVIDIA's model license. The combined URDF and adapter geometry are modified simulation assets; their wrist mounting and adapter inertia are estimates described in the provenance file.

Virtual environments, model weights, training outputs, local credentials, and caches are excluded from Git. Existing local training checkpoints are preserved outside this repository.
