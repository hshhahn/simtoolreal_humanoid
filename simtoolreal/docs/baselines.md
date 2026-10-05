# Baselines

## Kinematic Retargeting Baseline

### Hand Pose Extraction

To run this baseline, we first need to perform additional processing on the data:

1. use SAM2 to extract hand masks over the RGB video frames
2. use `hamer_depth` to estimate 3D hand poses with HaMeR and then refine them with the depth images

This produces:

* `hand_mask/`
* `hand_pose_trajectory/`

Use the SimToolReal branches of the two external repos:

* SAM2: https://github.com/tylerlum/segment-anything-2-real-time/tree/SimToolReal
* HaMeR Depth: https://github.com/tylerlum/hamer_depth/tree/SimToolReal

Follow the installation instructions in those repos and create **separate `uv` environments** for SAM2 and HaMeR Depth. Do not duplicate or merge the environments.

The recommended entrypoint is `run_hand_pipeline.sh` in the SAM2 repo. Before running it, open the script and update the configuration at the top for your machine:

* `SAM2_REPO`
* `HAMER_DEPTH_REPO`
* `DEMO_DIR`
* `HAND_MASK_DIR`
* `HAND_POSE_TRAJECTORY_DIR`
* `HAND_PROMPT_ARGS`
* `HAND_TYPE`

In particular:

* `DEMO_DIR` should point to a demo directory containing `rgb/`, `depth/`, and `cam_K.txt`
* `HAND_MASK_DIR` is where SAM2 will write the hand masks
* `HAND_POSE_TRAJECTORY_DIR` is where HaMeR Depth will write the per-frame outputs
* `HAND_PROMPT_ARGS` controls how SAM2 initializes the hand mask tracking. The current default uses `--use_negative_prompt`, which lets you click the hand first and then click the arm to exclude it.
* `HAND_TYPE` defaults to `RIGHT`; change it to `LEFT` if your demo contains a left hand

Then run the pipeline from the SAM2 repo root:

```bash
cd /path/to/segment-anything-2-real-time
bash run_hand_pipeline.sh
```

At a high level, the script will:

1. run SAM2 on the original RGB frames to create `hand_mask/`
2. run HaMeR Depth on `rgb/`, `depth/`, `hand_mask/`, and `cam_K.txt`
3. save the resulting hand pose outputs into `hand_pose_trajectory/`

The script prints step banners as it runs. Depending on how `HAND_PROMPT_ARGS` is configured, SAM2 may ask you to click points on the first frame to initialize the hand mask tracking.

For the current `hamer_depth` SimToolReal workflow, the SAM-mask path does **not** require Detectron2 or GroundingDINO. It uses the SAM2 hand mask to define the hand crop and then runs HaMeR + depth refinement on that result.

For more detail, read `run_hand_pipeline.sh` directly. That script is the source of truth for the exact command sequence and environment assumptions.

### Visualize the Hand Pose Extraction and Retargeting

```
export DEMO_DIR=dextoolbench/data/hammer/claw_hammer/swing_down

python baselines/visualize_demo_with_hand.py \
--object-path assets/urdf/dex_tool_bench/hammer/claw_hammer/claw_hammer.urdf \
--object-poses-json-path $DEMO_DIR/poses.json \
--hand-poses-dir $DEMO_DIR/hand_pose_trajectory/ \
--visualize-hand-meshes \
--retarget-robot \
--save-retargeted-robot-to-file \
--rgb-path $DEMO_DIR/rgb/ \
--depth-path $DEMO_DIR/depth/ \
--cam-intrinsics-path $DEMO_DIR/cam_K.txt
```

This will generate a `retargeted_robot/<timestamp>.npz` file, which contains the retargeted robot poses.

### Replay the Retargeted Robot

You can then replay this trajectory with:

```
python deployment/replay_trajectory.py \
--file_path retargeted_robot/<timestamp>.npz
```

## Fixed Grasp Baseline

### Installation

Create a new conda environment for `pyroki`:

```
conda create --name pyroki_env python=3.10                            
conda activate pyroki_env

git clone https://github.com/chungmin99/pyroki.git
cd pyroki
pip install -e .

cd <this_repo>
pip install -e .
```

### Add Robot Spheres

Model the robot's collision geometry as a set of spheres:

```
python baselines/create_robot_spheres_interactive.py
```

### Test out TrajOpt

```
python baselines/test_trajopt_sharpa.py
```

### Run Trajopt on a Demo

```
export DEMO_DIR=dextoolbench/data/hammer/claw_hammer/swing_down
python baselines/visualize_demo_with_hand_trajopt.py \
--object-path assets/urdf/dex_tool_bench/hammer/claw_hammer/claw_hammer.urdf \
--object-poses-json-path $DEMO_DIR/poses.json \
--hand-poses-dir $DEMO_DIR/hand_pose_trajectory/ \
--visualize-hand-meshes \
--retarget-robot \
--retarget-robot-using-object-relative-pose \
--rgb-path $DEMO_DIR/rgb/ \
--depth-path $DEMO_DIR/depth/ \
--cam-intrinsics-path $DEMO_DIR/cam_K.txt
```

### Run on Real Robot

To run this on the real robot, we run the `rl_policy_node.py` and `goal_pose_node.py` to make the initial grasp, but we modify it to not update the goal pose beyond the first one.

```
python deployment/goal_pose_node.py \
--object_category hammer \
--object_name claw_hammer \
--task_name swing_down \
--success_threshold 0.0
```

```
python deployment/rl_policy_node.py \
--policy_path pretrained_policy \
--object_name claw_hammer
```

When the object has been lifted and the first goal pose has been reached, run the following to stop the policy and to save to a file `trajopt_inputs.json` that contains the inputs to TrajOpt:

```
pkill -USR1 -f rl_policy
```

Next, we run TrajOpt to generate the retargeted robot poses.

```
python baselines/run_trajopt.py
```

This will generate a `trajopt_outputs.json` file, which contains the retargeted robot poses. When this is generated, we can continue `rl_policy_node.py` to run the retargeted robot poses.
