from typing import Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
from jax.typing import ArrayLike


def solve_trajopt(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_name: str,
    start_position: ArrayLike,
    start_wxyz: ArrayLike,
    end_position: ArrayLike,
    end_wxyz: ArrayLike,
    timesteps: int,
    dt: float,
) -> ArrayLike:
    if isinstance(start_position, onp.ndarray):
        np = onp
    elif isinstance(start_position, jnp.ndarray):
        np = jnp
    else:
        raise ValueError(f"Invalid type for `ArrayLike`: {type(start_position)}")

    # 1. Solve IK for the start and end poses.
    target_link_index = robot.links.names.index(target_link_name)
    start_cfg, end_cfg = solve_iks_with_collision(
        robot=robot,
        coll=robot_coll,
        world_coll_list=world_coll,
        target_link_index=target_link_index,
        target_position_0=jnp.array(start_position),
        target_wxyz_0=jnp.array(start_wxyz),
        target_position_1=jnp.array(end_position),
        target_wxyz_1=jnp.array(end_wxyz),
    )

    # 2. Initialize the trajectory through linearly interpolating the start and end poses.
    init_traj = jnp.linspace(start_cfg, end_cfg, timesteps)

    # 3. Optimize the trajectory.
    traj_vars = robot.joint_var_cls(jnp.arange(timesteps))

    robot = jax.tree.map(lambda x: x[None], robot)  # Add batch dimension.
    robot_coll = jax.tree.map(lambda x: x[None], robot_coll)  # Add batch dimension.

    # --- Soft costs ---
    costs: list[jaxls.Cost] = [
        pk.costs.rest_cost(
            traj_vars,
            traj_vars.default_factory()[None],
            jnp.array([0.01])[None],
        ),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            jnp.array([1])[None],
        ),
        pk.costs.five_point_acceleration_cost(
            robot.joint_var_cls(jnp.arange(2, timesteps - 2)),
            robot.joint_var_cls(jnp.arange(4, timesteps)),
            robot.joint_var_cls(jnp.arange(3, timesteps - 1)),
            robot.joint_var_cls(jnp.arange(1, timesteps - 3)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 4)),
            dt,
            jnp.array([0.1])[None],
        ),
        pk.costs.self_collision_cost(
            robot,
            robot_coll,
            traj_vars,
            0.02,
            5.0,
        ),
    ]

    # --- Constraints (augmented Lagrangian penalties) ---
    # Joint limits.
    costs.append(pk.costs.limit_constraint(robot, traj_vars))

    # Start / end pose constraints.
    @jaxls.Cost.factory(kind="constraint_eq_zero", name="start_pose_constraint")
    def start_pose_constraint(vals: jaxls.VarValues, var) -> jax.Array:
        return (vals[var] - start_cfg).flatten()

    @jaxls.Cost.factory(kind="constraint_eq_zero", name="end_pose_constraint")
    def end_pose_constraint(vals: jaxls.VarValues, var) -> jax.Array:
        return (vals[var] - end_cfg).flatten()

    costs.append(start_pose_constraint(robot.joint_var_cls(jnp.arange(0, 2))))
    costs.append(
        end_pose_constraint(robot.joint_var_cls(jnp.arange(timesteps - 2, timesteps)))
    )

    # Velocity limits.
    costs.append(
        pk.costs.limit_velocity_constraint(
            robot,
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            dt,
        )
    )

    # World collision avoidance using swept volumes.
    def compute_world_coll_residual(
        vals: jaxls.VarValues,
        robot: pk.Robot,
        robot_coll: pk.collision.RobotCollision,
        world_coll_obj: pk.collision.CollGeom,
        prev_traj_vars: jaxls.Var[jax.Array],
        curr_traj_vars: jaxls.Var[jax.Array],
    ):
        coll = robot_coll.get_swept_capsules(
            robot, vals[prev_traj_vars], vals[curr_traj_vars]
        )
        dist = pk.collision.collide(
            coll.reshape((-1, 1)), world_coll_obj.reshape((1, -1))
        )  # >0 means no collision
        return dist.flatten() - 0.05  # safety margin

    for world_coll_obj in world_coll:
        costs.append(
            jaxls.Cost(
                compute_world_coll_residual,
                (
                    robot,
                    robot_coll,
                    jax.tree.map(lambda x: x[None], world_coll_obj),
                    robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                    robot.joint_var_cls(jnp.arange(1, timesteps)),
                ),
                kind="constraint_geq_zero",
                name="world Collision (sweep)",
            )
        )

    # 4. Solve the optimization problem with augmented Lagrangian for constraints.
    solution = (
        jaxls.LeastSquaresProblem(
            costs=costs,
            variables=[traj_vars],
        )
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make((traj_vars.with_value(init_traj),)),
        )
    )
    return np.array(solution[traj_vars])


@jdc.jit
def solve_iks_with_collision(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: Sequence[pk.collision.CollGeom],
    target_link_index: int,
    target_position_0: jax.Array,
    target_wxyz_0: jax.Array,
    target_position_1: jax.Array,
    target_wxyz_1: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Solves the basic IK problem with collision avoidance. Returns joint configuration."""
    joint_var_0 = robot.joint_var_cls(0)
    joint_var_1 = robot.joint_var_cls(1)
    joint_vars = robot.joint_var_cls(jnp.arange(2))
    variables = [joint_vars]

    # Soft costs: pose matching, regularization, self-collision
    costs = [
        pk.costs.pose_cost(
            robot,
            joint_var_0,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz_0), target_position_0
            ),
            jnp.array(target_link_index),
            jnp.array([10.0] * 3),
            jnp.array([1.0] * 3),
        ),
        pk.costs.pose_cost(
            robot,
            joint_var_1,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz_1), target_position_1
            ),
            jnp.array(target_link_index),
            jnp.array([10.0] * 3),
            jnp.array([1.0] * 3),
        ),
        pk.costs.rest_cost(
            joint_vars,
            jnp.array(joint_vars.default_factory()[None]),
            jnp.array(0.001),
        ),
        pk.costs.self_collision_cost(
            jax.tree.map(lambda x: x[None], robot),
            jax.tree.map(lambda x: x[None], coll),
            joint_vars,
            0.02,
            5.0,
        ),
    ]

    # Small cost to encourage the start + end configs to be close to each other.
    @jaxls.Cost.factory(name="JointSimilarityCost")
    def joint_similarity_cost(vals, var_0, var_1):
        return (vals[var_0] - vals[var_1]).flatten()

    costs.append(joint_similarity_cost(joint_var_0, joint_var_1))

    # World collision as soft cost (more robust for IK initialization)
    costs.extend(
        [
            pk.costs.world_collision_cost(
                jax.tree.map(lambda x: x[None], robot),
                jax.tree.map(lambda x: x[None], coll),
                joint_vars,
                jax.tree.map(lambda x: x[None], world_coll),
                0.05,
                10.0,
            )
            for world_coll in world_coll_list
        ]
    )

    # Constraint: joint limits
    costs.append(
        pk.costs.limit_constraint(
            jax.tree.map(lambda x: x[None], robot),
            joint_vars,
        ),
    )

    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=variables)
        .analyze()
        .solve(verbose=False)
    )
    return sol[joint_var_0], sol[joint_var_1]


# --- 1. LIGHTWEIGHT IK (Used by Smart Init) ---
@jax.jit
def solve_ik_biased(
    robot: pk.Robot,
    target_link_index: int,
    target_pos: jax.Array,
    target_wxyz: jax.Array,
    bias_cfg: jax.Array,
) -> jax.Array:
    """
    Lightweight IK biased towards 'bias_cfg'.
    No collision checks are performed here to ensure high speed during initialization.
    """
    joint_var = robot.joint_var_cls(0)

    costs = [
        # Strong pose matching
        pk.costs.pose_cost(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_pos
            ),
            jnp.array(target_link_index),
            jnp.array([50.0] * 3),
            jnp.array([10.0] * 3),
        ),
        # Bias towards previous config
        pk.costs.rest_cost(joint_var, bias_cfg, jnp.array(1.0)),
        # Joint limits
        pk.costs.limit_constraint(robot, joint_var),
    ]

    sol = jaxls.LeastSquaresProblem(costs, [joint_var]).analyze().solve(verbose=False)
    return sol[joint_var]


# --- 2. JIT KERNEL FOR OPTIMIZATION ---
def _solve_sparse_kernel(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_index: int,
    start_cfg: jax.Array,
    init_traj: jax.Array,
    waypoint_pos_array: jax.Array,
    waypoint_rot_array: jax.Array,
    waypoint_indices: tuple[int, ...],
    timesteps: int,
    dt: float,
) -> jax.Array:
    """The heavy optimization solver."""
    traj_vars = robot.joint_var_cls(jnp.arange(timesteps))

    # Create batched versions for trajectory-wide costs
    robot_b = jax.tree.map(lambda x: x[None], robot)
    robot_coll_b = jax.tree.map(lambda x: x[None], robot_coll)

    costs: list[jaxls.Cost] = []

    # 1. Start Pose Constraint
    # Use unbatched 'robot' (single variable)
    @jaxls.Cost.factory(kind="constraint_eq_zero", name="start_pose_constraint")
    def start_pose_constraint(vals: jaxls.VarValues, var) -> jax.Array:
        return (vals[var] - start_cfg).flatten()

    costs.append(start_pose_constraint(robot.joint_var_cls(0)))

    # 2. Waypoints
    # Use unbatched 'robot' (single variable per cost)
    for i, t in enumerate(waypoint_indices):
        costs.append(
            pk.costs.pose_cost(
                robot,
                robot.joint_var_cls(t),
                jaxlie.SE3.from_rotation_and_translation(
                    jaxlie.SO3(waypoint_rot_array[i]), waypoint_pos_array[i]
                ),
                jnp.array(target_link_index),
                jnp.array([50.0] * 3),
                jnp.array([10.0] * 3),
            )
        )

    # 3. Smoothness & Limits
    costs.extend(
        [
            pk.costs.smoothness_cost(
                robot.joint_var_cls(jnp.arange(1, timesteps)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                jnp.array([2.0])[None],
            ),
            # Use batched 'robot_b' (variable array matches batch size)
            pk.costs.limit_constraint(robot_b, traj_vars),
            pk.costs.limit_velocity_constraint(
                robot_b,
                robot.joint_var_cls(jnp.arange(1, timesteps)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                dt,
            ),
        ]
    )

    # 4. World Collision (Sweep)
    # Use batched 'robot_b' (swept variables)
    def compute_world_coll_residual(vals, r, rc, wc, prev, curr):
        coll = rc.get_swept_capsules(r, vals[prev], vals[curr])
        dist = pk.collision.collide(coll.reshape((-1, 1)), wc.reshape((1, -1)))
        return dist.flatten() - 0.05

    for world_coll_obj in world_coll:
        costs.append(
            jaxls.Cost(
                compute_world_coll_residual,
                (
                    robot_b,
                    robot_coll_b,
                    jax.tree.map(lambda x: x[None], world_coll_obj),
                    robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                    robot.joint_var_cls(jnp.arange(1, timesteps)),
                ),
                kind="constraint_geq_zero",
                name="World Collision",
            )
        )

    # 5. Self Collision
    # Use batched 'robot_b'
    costs.append(
        pk.costs.self_collision_cost(robot_b, robot_coll_b, traj_vars, 0.02, 5.0)
    )

    solution = (
        jaxls.LeastSquaresProblem(costs, [traj_vars])
        .analyze()
        .solve(initial_vals=jaxls.VarValues.make((traj_vars.with_value(init_traj),)))
    )
    return solution[traj_vars]


# Explicitly JIT the kernel
_solve_sparse_kernel = jax.jit(
    _solve_sparse_kernel, static_argnames=["timesteps", "dt", "waypoint_indices"]
)


# --- 3. PUBLIC API ---
def solve_waypoint_trajopt(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_name: str,
    start_cfg: ArrayLike,
    waypoints: Dict[int, Tuple[ArrayLike, ArrayLike]],
    timesteps: int,
    dt: float,
) -> onp.ndarray:

    target_link_index = robot.links.names.index(target_link_name)
    start_cfg = jnp.array(start_cfg)

    # --- A. Smart Initialization (Lightweight Python Loop) ---
    print("Running Lightweight Smart Init...")
    sorted_steps = sorted(waypoints.keys())

    # Store configurations for key points
    anchors = {0: start_cfg}
    current_bias = start_cfg

    for t in sorted_steps:
        target_pos, target_wxyz = waypoints[t]

        # Solve IK without collision, biased to previous anchor
        solved_cfg = solve_ik_biased(
            robot,
            target_link_index,
            jnp.array(target_pos),
            jnp.array(target_wxyz),
            current_bias,
        )
        anchors[t] = solved_cfg
        current_bias = solved_cfg

    # Include end of trajectory in interpolation points if not explicitly set
    if sorted_steps[-1] < timesteps - 1:
        anchors[timesteps - 1] = anchors[sorted_steps[-1]]
        sorted_steps.append(timesteps - 1)

    # Linear Interpolation between anchors
    init_traj_segments = []
    prev_t = 0

    for t in sorted_steps:
        target_cfg = anchors[t]

        duration = t - prev_t
        if duration > 0:
            # linspace includes start and end; slice off start to avoid duplicates
            segment = jnp.linspace(anchors[prev_t], target_cfg, duration + 1)
            if prev_t > 0:
                segment = segment[1:]
            init_traj_segments.append(segment)
        prev_t = t

    init_traj = jnp.concatenate(init_traj_segments)

    # Handle any remaining padding due to rounding/indexing
    if len(init_traj) < timesteps:
        init_traj = jnp.concatenate(
            [init_traj, jnp.tile(init_traj[-1], (timesteps - len(init_traj), 1))]
        )
    elif len(init_traj) > timesteps:
        init_traj = init_traj[:timesteps]

    # --- B. Full Optimization (JIT Kernel) ---
    print("Running Optimization...")
    sorted_items = sorted(waypoints.items())

    # Static Structure
    indices = tuple(t for t, _ in sorted_items)

    # Dynamic Data
    pos_list = jnp.array([val[0] for _, val in sorted_items])
    rot_list = jnp.array([val[1] for _, val in sorted_items])

    traj = _solve_sparse_kernel(
        robot,
        robot_coll,
        world_coll,
        target_link_index,
        start_cfg,
        init_traj,
        pos_list,
        rot_list,
        indices,
        timesteps,
        dt,
    )

    return onp.array(traj)
