from isaacgymenvs.utils.rendering import render_camera_sensors_for_current_step


class FakeGym:
    def __init__(self):
        self.calls = []

    def fetch_results(self, sim, wait):
        self.calls.append(("fetch_results", sim, wait))

    def step_graphics(self, sim):
        self.calls.append(("step_graphics", sim))

    def render_all_camera_sensors(self, sim):
        self.calls.append(("render_all_camera_sensors", sim))


def test_gpu_camera_render_advances_graphics_first():
    gym = FakeGym()

    render_camera_sensors_for_current_step(gym, "sim", "cuda:0")

    assert gym.calls == [
        ("fetch_results", "sim", True),
        ("step_graphics", "sim"),
        ("render_all_camera_sensors", "sim"),
    ]


def test_cpu_camera_render_does_not_fetch_gpu_results():
    gym = FakeGym()

    render_camera_sensors_for_current_step(gym, "sim", "cpu")

    assert gym.calls == [
        ("step_graphics", "sim"),
        ("render_all_camera_sensors", "sim"),
    ]
