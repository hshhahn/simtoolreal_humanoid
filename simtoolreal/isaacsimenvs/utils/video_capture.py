"""Inline video capture during training, logged to wandb.

Two pieces:

- `attach_record_camera(env, ...)` — spawns a `Camera` sensor, aims it using
  the env's configured `record_camera_eye` / `record_camera_target` (env-local
  frame), returns the sensor, and stores it on `env._video_camera` so observers
  can find it.

- `WandbVideoObserver` — rl_games `AlgoObserver` subclass. Every
  `video_interval` training iters, it starts accumulating frames via
  `process_infos` (fires once per env step), and when enough frames are
  buffered it encodes an mp4 and pushes it to the active wandb run.

Adds per-call render cost only while a capture window is open.
"""

from __future__ import annotations

from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from rl_games.common.algo_observer import AlgoObserver


def attach_record_camera(
    env,
    env_idx: int | None = None,
    width: int = 640,
    height: int = 480,
) -> Camera:
    """Attach a Camera sensor aimed using the env cfg's record_camera_{eye,target}.

    The env cfg must define `record_camera_eye` and `record_camera_target` as
    3-tuples in env-local coords (relative to `scene.env_origins[env_idx]`).
    Every task owns its camera framing — no articulation-tracking heuristic.

    If `env_idx` is None, picks the env whose origin is nearest to (0, 0, 0) —
    important when `num_envs` is large: the default GridCloner centers the grid
    around origin, so env_0 sits at the corner (e.g. (-126, -126, 0) for 4096
    envs at spacing=4). That's outside the default 100×100 m `GroundPlaneCfg`,
    which means env_0's camera sees no ground at all. The most central env is
    always on the ground plane.

    Must be called AFTER env.__init__() (which calls sim.reset() once).
    Triggers another `sim.reset()` so the new camera's `_timestamp` buffer
    initializes. Stores the sensor on `env._video_camera`.
    """
    if env_idx is None:
        distances = torch.norm(env.scene.env_origins, dim=1)
        env_idx = int(torch.argmin(distances).item())
        print(
            f"[attach_record_camera] auto-picked env_idx={env_idx} "
            f"(origin {env.scene.env_origins[env_idx].cpu().tolist()}, "
            f"distance {float(distances[env_idx]):.2f} m from world origin)"
        )
    camera_cfg = CameraCfg(
        prim_path="/World/RecordCamera",
        update_period=0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 10.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )
    camera = Camera(cfg=camera_cfg)
    env.sim.reset()

    env_origin = env.scene.env_origins[env_idx]
    eye_local = torch.tensor(env.cfg.record_camera_eye, device=env.device)
    target_local = torch.tensor(env.cfg.record_camera_target, device=env.device)
    eye = env_origin + eye_local
    target = env_origin + target_local
    camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))

    env._video_camera = camera
    return camera


class WandbVideoObserver(AlgoObserver):
    """Periodically log rollout mp4s to wandb.

    Hooks:
      - `after_print_stats(frame, epoch_num, total_time)` — at each iteration
        boundary; starts a new capture window every `video_interval` iters.
      - `process_infos(infos, done_indices)` — fires once per env step;
        appends a camera frame while a window is open.

    Flushes the buffer when `capture_frames` have been collected.
    """

    def __init__(
        self,
        env,
        video_interval: int = 10,
        capture_frames: int = 120,
        video_fps: int = 30,
        video_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self._env = env
        self._camera: Camera | None = getattr(env, "_video_camera", None)
        self._video_interval = video_interval
        self._capture_frames = capture_frames
        self._video_fps = video_fps
        self._video_dir = Path(video_dir) if video_dir else Path("train_videos")
        self._video_dir.mkdir(parents=True, exist_ok=True)
        self._frames: list = []
        self._capturing_bucket: int | None = None  # iter at which window opened
        self._step_count: int = 0

        # process_infos fires once per POLICY step (= physics_dt * decimation).
        # Use env.step_dt when available; otherwise fall back to physics_dt.
        step_dt = float(getattr(env, "step_dt", env.sim.get_physics_dt()))
        self._capture_every = max(1, round((1.0 / video_fps) / step_dt))
        print(
            f"[WandbVideoObserver] init: interval={video_interval} iters, "
            f"frames={capture_frames} (~{capture_frames / video_fps:.1f}s sim), "
            f"capture_every={self._capture_every} policy steps, "
            f"camera={'attached' if self._camera is not None else 'MISSING'}, "
            f"dir={self._video_dir}"
        )

    def before_init(self, base_name, config, experiment_name):
        pass

    def after_init(self, algo):
        pass

    def process_infos(self, infos, env_done_indices, **kwargs):
        if self._camera is None or self._capturing_bucket is None:
            return
        if len(self._frames) >= self._capture_frames:
            return
        # Subsample physics steps → video_fps cadence.
        self._step_count += 1
        if self._step_count % self._capture_every != 0:
            return
        try:
            step_dt = float(getattr(self._env, "step_dt", self._env.sim.get_physics_dt()))
            self._camera.update(self._capture_every * step_dt)
            rgb = self._camera.data.output.get("rgb")
            if rgb is not None and rgb.shape[0] > 0:
                self._frames.append(rgb[0].detach().cpu().numpy()[:, :, :3])
        except Exception as e:  # don't kill training over a render glitch
            print(f"[WandbVideoObserver] frame capture failed: {e}")

    def after_steps(self):
        return None

    def after_clear_stats(self):
        pass

    def after_print_stats(self, frame, epoch_num, total_time):
        if self._camera is None:
            return
        import wandb

        # Window open and full → flush.
        if self._capturing_bucket is not None and len(self._frames) >= self._capture_frames:
            self._flush(frame)
            return

        # No window open and time to start one → arm the capture loop.
        if self._capturing_bucket is None and epoch_num % self._video_interval == 0:
            self._capturing_bucket = epoch_num
            self._frames = []

    def _flush(self, train_step: int) -> None:
        import imageio
        import wandb

        bucket = self._capturing_bucket
        out = self._video_dir / f"rollout_iter{bucket:05d}.mp4"
        try:
            imageio.mimwrite(str(out), self._frames, fps=self._video_fps)
        except Exception as e:
            print(f"[WandbVideoObserver] encoding failed at iter {bucket}: {e}")
        else:
            if wandb.run is not None:
                wandb.log(
                    {"rollout_video": wandb.Video(str(out), fps=self._video_fps, format="mp4")},
                    step=train_step,
                )
                print(f"[WandbVideoObserver] logged {len(self._frames)} frames to wandb at iter {bucket}")
            else:
                print(f"[WandbVideoObserver] no active wandb run — wrote mp4 only: {out}")
        self._capturing_bucket = None
        self._frames = []
