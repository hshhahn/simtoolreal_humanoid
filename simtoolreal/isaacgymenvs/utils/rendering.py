"""Rendering helpers that do not require importing Isaac Gym."""


def render_camera_sensors_for_current_step(gym, sim, device) -> None:
    """Advance headless graphics before rendering camera sensor buffers.

    Headless training normally skips ``VecTask.render()`` when
    ``force_render=False``. Rendering camera sensors without first advancing
    graphics therefore returns a stale image, often producing an entirely
    frozen rollout video.
    """

    if str(device) != "cpu":
        gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
