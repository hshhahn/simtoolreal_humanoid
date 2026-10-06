"""Free-standing G1 + Wuji hands, controlled jointly with SONIC 1.1."""

from pathlib import Path

from isaaclab.envs import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from ..simtoolreal.simtoolreal_env_cfg import (
    AssetsCfg,
    SimToolRealEnvCfg,
    ObsCfg,
    ResetCfg,
    RewardCfg,
    TerminationCfg,
    DomainRandomizationCfg,
    ActionCfg,
    _default_sim_cfg,
)


REPO = Path(__file__).resolve().parents[3]
MODEL_DIR = REPO.parent / "models/GEAR-SONIC/sonic_v1_1"
EXTRA_OBS_SIZES = {
    "base_position": 3,
    "base_gravity": 3,
    "base_linear_velocity": 3,
    "base_angular_velocity": 3,
    # Actor/critic receive one live frame; the decoder keeps its own 10-frame history.
    "sonic_current_proprioception": 93,
    "meta_actions": 67,
}
BIMANUAL_OBS_SIZES = {
    **EXTRA_OBS_SIZES,
    "meta_actions": 70,
    # Both sets of fingertips use the existing right-palm reference frame.
    "fingertip_pos_rel_palm": 30,
    "closest_fingertip_dist": 10,
    "left_palm_pos": 3,
    "left_palm_rot": 4,
    "left_palm_vel": 6,
}


@configclass
class SonicCfg:
    model_dir: str = str(MODEL_DIR)
    # Continuous [-1, 1] policy outputs are snapped to the native FSQ grid.
    latent_mode: str = "absolute"  # absolute | residual (around standing reference)
    residual_scale: float = 0.25
    # Opt in so saved configurations from unsmoothed runs retain their behavior.
    # Uses action.arm_moving_average and action.dof_speed_scale on the seven
    # manipulation-arm joints only; SONIC's remaining body outputs stay direct.
    smooth_right_arm_targets: bool = False
    smooth_left_arm_targets: bool = False
    control_both_hands: bool = False
    # Independent exploration for the three normalized Wuji commands. Body
    # log-std remains the PPO default (-2); no noise is added inside the decoder.
    policy_hand_std_init: float = 0.5
    policy_bounded_hand_mean: bool = True
    initial_hand_action: tuple[float, float, float] = (-0.5, -0.5, -0.5)
    initial_left_hand_action: tuple[float, float, float] = (-0.5, -0.5, -0.5)
    base_position: tuple[float, float, float] = (0.0, 0.6, 0.76)
    base_rotation: tuple[float, float, float, float] = (
        0.7071067811865476,
        0.0,
        0.0,
        -0.7071067811865476,
    )
    minimum_base_height: float = 0.45
    maximum_tilt_degrees: float = 60.0
    maximum_base_distance: float = 1.5
    fall_penalty: float = 5.0
    # Default motion penalty matches SimToolReal's physical joint velocities.
    # Retain this optional setting for reproducing older latent-penalty runs.
    latent_rate_penalty: float = 0.0
    # Speeds this far above hardware limits indicate a numerical physics failure.
    physics_velocity_limit_multiplier: float = 10.0
    physics_failure_penalty: float = 5.0


@configclass
class G1AssetsCfg(AssetsCfg):
    robot_urdf: str = str(REPO / "assets/urdf/g1_wuji/g1_wuji.urdf")
    table_urdf: str = str(REPO / "assets/urdf/g1_wuji/manipulation_table.urdf")
    robot_fix_base: bool = False
    robot_disable_gravity: bool = False
    # Keep palm and tip frames as separate bodies for observation/reward queries.
    robot_merge_fixed_joints: bool = False
    robot_capsule_collisions: bool = True
    robot_max_depenetration_velocity: float = 1.0
    robot_solver_velocity_iterations: int = 4
    # Full SimToolReal pool: 100 samples for each of 12 shape distributions.
    num_assets_per_type: int = 100
    robot_friction: float = 0.8


def g1_sim_cfg():
    sim = _default_sim_cfg()
    sim.dt = 0.005
    sim.render_interval = 4
    sim.physx.min_velocity_iteration_count = 4
    sim.physx.max_velocity_iteration_count = 4
    return sim


@configclass
class G1WujiSonicEnvCfg(SimToolRealEnvCfg):
    decimation: int = 4  # SONIC runs at exactly 50 Hz, physics at 200 Hz.
    action_space: int = 67
    observation_space: int = 432
    state_space: int = 454
    sim = g1_sim_cfg()
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, env_spacing=3.0, replicate_physics=False, clone_in_fabric=False
    )
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.6, -2.1, 1.7),
        lookat=(0.0, 0.25, 0.8),
        resolution=(960, 720),
        origin_type="env",
        env_index=0,
    )
    assets: G1AssetsCfg = G1AssetsCfg()
    sonic: SonicCfg = SonicCfg()
    obs: ObsCfg = ObsCfg(
        obs_list=ObsCfg().obs_list + tuple(EXTRA_OBS_SIZES),
        state_list=ObsCfg().state_list + tuple(EXTRA_OBS_SIZES),
    )
    action: ActionCfg = ActionCfg(hand_moving_average=0.3)
    reset: ResetCfg = ResetCfg(
        # The URDF root is the tabletop centre; its upper surface is +0.025 m.
        table_reset_z=0.475,
        table_reset_center_xy=(0.0, 0.25),
        table_reset_z_range=0.0,
        table_object_z_offset=0.10,
        reset_position_center_xy=(-0.15, 0.30),
        reset_position_noise_x=0.08,
        reset_position_noise_y=0.06,
        reset_position_noise_z=0.01,
        reset_dof_pos_random_interval_arm=0.0,
        reset_dof_pos_random_interval_fingers=0.0,
        reset_dof_vel_random_interval=0.0,
        target_volume_mins=(-0.30, 0.14, 0.55),
        target_volume_maxs=(0.30, 0.40, 0.85),
    )
    reward: RewardCfg = RewardCfg()
    termination: TerminationCfg = TerminationCfg(episode_length=500)
    # Task randomization remains active for objects. Keep timing deterministic
    # during controller bring-up; latency can be enabled after verification.
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg(
        use_action_delay=False,
        use_obs_delay=False,
        use_object_state_delay_noise=False,
        force_scale=0.0,
        torque_scale=0.0,
    )


@configclass
class G1WujiSonicLiftEnvCfg(G1WujiSonicEnvCfg):
    """Initial curriculum: two markers, right-hand pickup, vertical pose goal.

    Only the task distribution changes. Rewards, keypoint bounding boxes,
    success criteria and the floating-base controller are inherited unchanged.
    """

    assets: G1AssetsCfg = G1AssetsCfg(
        handle_head_types=("marker",), num_assets_per_type=2, shuffle_assets=False
    )
    reset: ResetCfg = G1WujiSonicEnvCfg().reset.replace(
        fixed_start_pose=(-0.23, 0.32, 0.54, 1.0, 0.0, 0.0, 0.0),
        fixed_goal_pose=(-0.23, 0.32, 0.70, 1.0, 0.0, 0.0, 0.0),
    )


@configclass
class G1WujiSonicBimanualEnvCfg(G1WujiSonicEnvCfg):
    """Full tool pool, one object per environment, either or both hands usable.

    Action order preserves the original prefix: SONIC64, right-hand3, left-hand3.
    Lift/pose goals and reward coefficients are inherited from SimToolReal.
    """

    action_space: int = 70
    observation_space: int = 457
    state_space: int = 490
    sonic: SonicCfg = SonicCfg(
        control_both_hands=True,
        smooth_right_arm_targets=True,
        smooth_left_arm_targets=True,
    )
    action: ActionCfg = ActionCfg(
        arm_moving_average=0.1, hand_moving_average=0.1, dof_speed_scale=1.5
    )
    obs: ObsCfg = ObsCfg(
        obs_list=G1WujiSonicEnvCfg().obs.obs_list + ("left_palm_pos", "left_palm_rot"),
        state_list=G1WujiSonicEnvCfg().obs.state_list
        + ("left_palm_pos", "left_palm_rot", "left_palm_vel"),
    )
    # Give each hand opportunities to approach tools on its side of the table.
    reset: ResetCfg = G1WujiSonicEnvCfg().reset.replace(
        reset_position_center_xy=(0.0, 0.30), reset_position_noise_x=0.23
    )
