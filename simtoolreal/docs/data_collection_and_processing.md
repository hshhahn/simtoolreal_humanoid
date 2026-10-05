# Data Collection and Processing

This document describes the full pipeline for collecting real-world object manipulation data and converting it into task trajectories for DexToolBench.

## FoundationPose Setup

Clone and set up the [FoundationPose fork](https://github.com/kushal2000/FoundationPose) in a **separate conda environment** (`foundationpose`). See its README for full installation instructions, including model weight downloads and ROS setup.

This environment is used for Steps 1, 3, and live inference.

## Step 1: Record RGB-D Video

From the FoundationPose repo, record an RGB-D video with a ZED camera:

```bash
conda activate foundationpose

python record_video.py \
    --save_dir recordings/ \
    --serial_number <ZED_SERIAL> \
    --fps 30
```

Press `Ctrl+C` to stop. This creates:

```
recordings/<timestamp>/
  ├── rgb/frame_0000.png, frame_0001.png, ...
  ├── depth/frame_0000.png, frame_0001.png, ...
  ├── cam_K.txt
  └── rgb.mp4
```

## Step 2: Extract Object Mesh (SAM 2 + SAM 3D)

Use the SimToolReal branches of the SAM2 and SAM3D repos:

* SAM2: https://github.com/tylerlum/segment-anything-2-real-time/tree/SimToolReal
* SAM3D: https://github.com/tylerlum/sam-3d-objects/tree/SimToolReal

Follow the installation instructions in those repos and create **separate `uv` virtual environments** for SAM2 and SAM3D. Do not try to share one environment across both repos.

The main entrypoint for this step is `run_mesh_pipeline.sh` in the SAM2 repo. Before running it, open that script and update the configuration at the top for your machine:

* `SAM2_REPO`
* `SAM3_REPO`
* `DEMO_DIR`
* `OUTPUT_DIR`
* any environment activation assumptions in the `sam2()` and `sam3()` helper functions if your local setup differs

In particular, make sure:

* `SAM2_REPO` points to your local `segment-anything-2-real-time` checkout
* `SAM3_REPO` points to your local `sam-3d-objects` checkout
* `DEMO_DIR` points to the recorded data directory containing `rgb/`, `depth/`, and `cam_K.txt`
* `OUTPUT_DIR` points to where you want the reconstruction and processed mesh artifacts written

Then run the pipeline from the SAM2 repo root:

```bash
cd /path/to/segment-anything-2-real-time
bash run_mesh_pipeline.sh
```

At a high level, the script will:

1. run SAM2 on the recorded RGB frames to create object masks
2. run SAM3D to reconstruct the object mesh from the RGB-D data and masks
3. render the reconstructed mesh into RGB/depth views
4. run SAM2 on those rendered views to create handle and head masks
5. run the final mesh postprocessing step to compute the canonical handle frame and export the final mesh

One small detail: the SAM2 stage writes masks for **all** frames in the original video, but the current SAM3D reconstruction step only strictly uses the **first** RGB image, **first** depth image, and **first** mask image. We still run SAM2 over the full sequence because it is useful to have all masks available.

The script prints step banners as it runs, and depending on how the prompt arguments are configured, it may ask you to click points on the first frame to initialize SAM2.

For more detail, read `run_mesh_pipeline.sh` directly. That script is the source of truth for the exact command sequence and the environment assumptions.

The final result should be a metric-scale `.obj` mesh ready for downstream use in FoundationPose and SimToolReal. The main exported artifact is:

```bash
${OUTPUT_DIR}/mesh_handle_frame/mesh_handle_frame.obj
```

## Step 3: Extract 6D Poses with FoundationPose

Run FoundationPose on the recorded video:

```bash
conda activate foundationpose

python extract_poses.py \
    --video_dir recordings/<timestamp>/ \
    --mesh_path /path/to/object.obj \
    --calibration /path/to/T_RC.txt \
    --output_path recordings/<timestamp>/poses.json \
    --debug 1
```

An interactive window opens on the first frame -- click 4 corners of a bounding box around the object for SAM segmentation. FoundationPose then tracks through all frames.

Output (`poses.json`):
```json
{
  "poses_cam": [[x, y, z, qx, qy, qz, qw], ...],
  "poses_robot": [[x, y, z, qx, qy, qz, qw], ...]
}
```

## Step 4: Process Poses into Task Trajectory

From the SimToolReal repo, process the raw poses into a DexToolBench task trajectory:

```bash
conda activate simtoolreal

python dextoolbench/process_poses.py \
    --poses_path recordings/<timestamp>/poses.json \
    --object_category hammer \
    --object_name claw_hammer \
    --task_name swing_down
```

This outputs a trajectory JSON to `dextoolbench/trajectories/<object_category>/<object_name>/<task_name>.json` with poses in world frame, ready for use in training and evaluation.

## FoundationPose During Inference

To run live tracking at inference time, install FoundationPose as described above, then run:

```bash
conda activate foundationpose
cd /path/to/FoundationPose

python live_tracking_with_ros.py \
    --mesh_path /path/to/object.obj \
    --calibration calibration/T_RC_example.txt
```

This publishes object poses to `robot_frame/current_object_pose` as a ROS `PoseStamped` topic, which is consumed by the RL Policy Node. See the main [README](../README.md) for the full deployment flowchart.
