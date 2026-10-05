import json

import numpy as np

from baselines.visualize_demo_with_hand_trajopt import interpolate_traj, solve_trajopt

with open("trajopt_inputs.json", "r") as f:
    trajopt_inputs = json.load(f)
T_R_Ps_using_lifted_object_pose = np.array(
    trajopt_inputs["T_R_Ps_using_lifted_object_pose"]
)
q = np.array(trajopt_inputs["q"])
TRAJECTORY_LENGTH = T_R_Ps_using_lifted_object_pose.shape[0]
assert T_R_Ps_using_lifted_object_pose.shape == (TRAJECTORY_LENGTH, 4, 4), (
    f"T_R_Ps_using_lifted_object_pose.shape: {T_R_Ps_using_lifted_object_pose.shape}, expected: ({TRAJECTORY_LENGTH}, 4, 4)"
)
assert q.shape == (29,), f"q.shape: {q.shape}, expected: (29,)"

# DOWNSAMPLE_FACTOR = 100
# Want there to be about 30 waypoints
DOWNSAMPLE_FACTOR = TRAJECTORY_LENGTH // 30
print(
    f"Original trajectory length: {TRAJECTORY_LENGTH}, DOWNSAMPLE_FACTOR: {DOWNSAMPLE_FACTOR}, downsampled trajectory length: {TRAJECTORY_LENGTH // DOWNSAMPLE_FACTOR}"
)

retargeted_qs = solve_trajopt(
    T_R_Ps=T_R_Ps_using_lifted_object_pose[::DOWNSAMPLE_FACTOR],
    q_start=q.copy(),
    dt=1 / 30,
    use_collision_avoidance=True,
)

retargeted_qs = interpolate_traj(retargeted_qs, n_steps=DOWNSAMPLE_FACTOR)
if retargeted_qs.shape[0] < TRAJECTORY_LENGTH:
    extra = TRAJECTORY_LENGTH - retargeted_qs.shape[0]
    retargeted_qs = np.concatenate(
        [retargeted_qs, retargeted_qs[-1][None].repeat(extra, axis=0)], axis=0
    )
elif retargeted_qs.shape[0] > TRAJECTORY_LENGTH:
    extra = retargeted_qs.shape[0] - TRAJECTORY_LENGTH
    retargeted_qs = retargeted_qs[:-extra]
assert retargeted_qs.shape == (TRAJECTORY_LENGTH, 29), (
    f"retargeted_qs.shape: {retargeted_qs.shape}, expected: ({TRAJECTORY_LENGTH}, 29)"
)

with open("trajopt_outputs.json", "w") as f:
    json.dump(
        {
            "retargeted_qs": retargeted_qs.tolist(),
        },
        f,
        indent=4,
    )
print("Saved trajopt outputs to trajopt_outputs.json")
