The composed robot uses SONIC 1.1's 29-joint G1 body and the official original
Wuji Hand models on both wrists. `g1_wuji.urdf`, the meshes, actuator settings,
licenses and source provenance are produced by `scripts/build_g1_wuji_assets.py`.

The G1 visual meshes match NVIDIA's Git LFS hashes exactly. They are copied
from the pinned Unitree repository because its public checkout includes the
actual binary meshes. Body inertias and collision primitives come from
NVIDIA's training URDF. Hand inertias and joint limits come from Wuji's URDFs;
hand PD gains and reflected inertias come from Wuji's MJCF models.

The official G1 adapter mesh is converted from millimeters to meters.
The flange position and adapter density are simulation estimates recorded in
`provenance.json`. Calibrate the mounting transform and adapter inertia against
the assembled hardware before transferring a trained policy to a physical G1.

This asset contains 69 actuated physical joints. The task policy outputs 67
commands: 64 SONIC latents plus three right-hand synergies. The left hand is
held in its open pose during the initial task.

The policy predicts all 67 commands in `[-1, 1]`. Before decoding, the SONIC
coordinates are snapped to the released model's 32-level FSQ grid (step `1/16`,
range `[-1, 0.9375]`). The three hand synergy predictions remain continuous.
