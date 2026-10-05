"""Cube falling under gravity: spawn a rigid cube above the ground and verify it lands.

Run without video (fast):
    python isaacsimenvs/tests/test_cube_falling_scene.py

Run with video:
    python isaacsimenvs/tests/test_cube_falling_scene.py --video
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


VIDEO_DIR = Path(__file__).resolve().parents[1] / "videos" / "test_videos"

CUBE_SIZE = 0.2
START_Z = 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        action="store_true",
        help=f"Record mp4 to {VIDEO_DIR}/cube_falling.mp4 (forces --enable_cameras)",
    )
    parser.add_argument("--steps", type=int, default=180, help="Number of physics steps")
    parser.add_argument("--video_fps", type=int, default=30)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    args.headless = True
    if args.video:
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    app = app_launcher.app

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationCfg, SimulationContext

    PHYSICS_DT = 1.0 / 60.0
    sim = SimulationContext(SimulationCfg(dt=PHYSICS_DT))

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    cube_cfg = RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, START_Z)),
    )
    cube = RigidObject(cfg=cube_cfg)

    camera = None
    if args.video:
        from isaaclab.sensors import Camera, CameraCfg

        camera_cfg = CameraCfg(
            prim_path="/World/RecordCamera",
            update_period=0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, -1.5, 0.8),
                rot=(0.8507, 0.5257, 0.0, 0.0),
                convention="opengl",
            ),
        )
        camera = Camera(cfg=camera_cfg)

    sim.reset()

    frames = []
    dt = sim.get_physics_dt()
    capture_every = max(1, round((1.0 / args.video_fps) / dt))
    for step_i in range(args.steps):
        sim.step(render=args.video)
        cube.update(dt)
        if camera is not None and step_i % capture_every == 0:
            camera.update(capture_every * dt)
            rgb = camera.data.output["rgb"]
            if rgb is not None and rgb.shape[0] > 0:
                frames.append(rgb[0].cpu().numpy()[:, :, :3])

    final_z = cube.data.root_pos_w[0, 2].item()
    print(f"Cube final z: {final_z:.4f} (started at {START_Z}, expected ~{CUBE_SIZE / 2:.2f})")
    assert abs(final_z - (CUBE_SIZE / 2)) < 0.0001, f"Cube did not land (final_z={final_z})"

    if args.video:
        import imageio

        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        out_path = VIDEO_DIR / "cube_falling.mp4"
        imageio.mimwrite(str(out_path), frames, fps=args.video_fps)
        print(f"Wrote {len(frames)} frames to {out_path}")
    else:
        print(f"Stepped {args.steps} frames with no camera")

    print("Cube falling test OK")
    # Kit's app.close() hangs on shutdown, force-exit.
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
