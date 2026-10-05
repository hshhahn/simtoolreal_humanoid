# SimToolReal humanoid

This repository contains our modified [SimToolReal](https://github.com/tylerlum/simtoolreal) environment for a free-standing Unitree G1 with two original Wuji hands. PPO predicts a frozen SONIC 1.1 controller's latent command and three right-hand commands jointly. Both feed-forward and LSTM policies are included.

The humanoid task is marker lifting over a lowered table. This is an experimental environment and controller integration; the existing humanoid training trials have not demonstrated successful sustained grasping. A fixed-KUKA parallel-gripper LSTM baseline uses the same two-marker asset pool with the original SimToolReal scene settings. Robot assets, environment code, agent configurations, and setup scripts are included. SONIC weights are downloaded from their pinned upstream revision during humanoid setup.

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
| MLP | 1024, 512, 256, 128 units with ELU |
| LSTM variant | One 1024-unit layer before the same MLP; sequence length 16 |
| Simulation | 200 Hz physics, 50 Hz control, 128 parallel environments by default |

The decoder is frozen. PPO evaluates the continuous policy samples for its likelihood calculation; FSQ projection happens at the environment's decoder boundary. The left hand is held open. The right-hand policy starts with standard deviation 0.5, independently of the body latent exploration.

The standing reference is encoded with the released SONIC encoder and used to initialize the policy's body mean. It does not anchor the robot to the floor.

## Task and rewards

`Isaacsimenvs-G1-Wuji-Sonic-Lift-v0` uses two procedural marker asset instances, a fixed initial pose at `(-0.23, 0.32, 0.54)` metres, and an elevated target pose at `(-0.23, 0.32, 0.70)` metres. The table surface is 0.50 m high. Both wrists begin above the table.

The task retains SimToolReal's fingertip approach, lift shaping, one-shot lift bonus, object keypoint progress, goal reward, and hand velocity penalty. The humanoid body velocity penalty uses coefficient 0.002, with added fall and latent-change penalties. There is no upright reward. Falls and invalid physics terminate the affected environment.

Objects use the existing four selected bounding-box corners for the goal formulation. The marker box is `(0.141, 0.03025, 0.0271)` metres; keypoint scale 1.5 and position tolerance 0.075 produce an effective 0.1125 m threshold. Success requires ten accumulated qualifying steps. The lift reward alone does not establish a grasp and can reward tossing an object.

The complete settings are in [env_cfg.py](simtoolreal/isaacsimenvs/tasks/g1_wuji_sonic/env_cfg.py), [env.py](simtoolreal/isaacsimenvs/tasks/g1_wuji_sonic/env.py), and the inherited [SimToolReal configuration](simtoolreal/isaacsimenvs/tasks/simtoolreal/simtoolreal_env_cfg.py). The full object distribution remains available as `Isaacsimenvs-G1-Wuji-Sonic-v0`.

## Verification and training

On a GPU workstation or inside a GPU allocation:

```bash
./verify_g1_wuji_sonic.sh \
  --task Isaacsimenvs-G1-Wuji-Sonic-Lift-v0 \
  --num_envs 4 --steps 500 --headless
```

For Slurm, follow [SLURM_SETUP.md](SLURM_SETUP.md). It includes a verification job and separate MLP/LSTM training submissions. [CODEX_SETUP_PROMPT.txt](CODEX_SETUP_PROMPT.txt) is a prompt for the agent setting up the server.

The fresh local installation passed the controller checks, 13 CPU regression tests, 500 simulation steps, and two PPO iterations of each architecture with checkpoint and TensorBoard output. See [VALIDATION.txt](VALIDATION.txt); the target Slurm cluster still needs its own verification run.

To train directly on an allocated GPU, run one of these from the checkout root:

```bash
./run_g1_wuji_lift_training.sh \
  agent.params.config.max_epochs=40000 \
  'hydra.run.dir=../g1_wuji_runs/${now:%Y%m%d_%H%M%S}_mlp'

./run_g1_wuji_lift_lstm_training.sh \
  agent.params.config.max_epochs=40000 \
  'hydra.run.dir=../g1_wuji_runs/${now:%Y%m%d_%H%M%S}_lstm'
```

`max_epochs` is the total PPO iteration limit, including iterations restored from a checkpoint. Use `--checkpoint /absolute/path/model.pth` to resume with the matching architecture, or add `--checkpoint_load_mode weights` to start a new optimizer from model weights.

Each Hydra run directory contains the rl_games experiment subdirectory. Checkpoints are under `nn/`, the best checkpoint alias is `best/model.pth`, and training curves are TensorBoard event files under `summaries/`:

```bash
source ./activate_simtoolreal.sh
tensorboard --logdir ../g1_wuji_runs --host 127.0.0.1 --port 6006
```

## KUKA parallel-gripper LSTM baseline

`Isaacsimenvs-Kuka-Parallel-Gripper-Lift-v0` uses the original fixed KUKA iiwa14 arm with a parametric simulation gripper. The seven arm joints, original starting joint values, base at `(0, 0.8, 0)` metres, arm drive gains, and narrow table are retained. The table root is at `0.38 ± 0.01` m and its surface at `0.53 ± 0.01` m. Only the object pool is simplified to the two procedural marker instances.

The task inherits the original randomized object resets, delta position/orientation goals, rewards, bounding-box keypoints, termination/curriculum, action smoothing, domain randomization, and 60 Hz control / 120 Hz physics. Its reward and termination functions are the original functions. The gripper has two opposing prismatic joints, driven by one shared command, with a 0–100 mm opening. Its adapter and box/pad geometry are simulation estimates, not a specific commercial gripper model.

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

## Source and licenses

The modified upstream base is recorded in [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json). [environment-reference.json](environment-reference.json) records the packages in the original working environment; selected installation constraints are in [isaacsim-constraints.txt](isaacsim-constraints.txt). The setup uses documented dependency overrides required by that working stack rather than installing the root package's legacy Isaac Gym dependencies.

SimToolReal's MIT license is retained in [LICENSE](LICENSE). Robot asset provenance and the NVIDIA, Wuji, and Unitree license notices are included under [simtoolreal/assets/urdf/g1_wuji/](simtoolreal/assets/urdf/g1_wuji/). Vendored rl_games retains its own license. Downloaded SONIC weights retain NVIDIA's model license. The combined URDF and adapter geometry are modified simulation assets; their wrist mounting and adapter inertia are estimates described in the provenance file.

Virtual environments, model weights, training outputs, local credentials, and caches are excluded from Git. Existing local training checkpoints are preserved outside this repository.
