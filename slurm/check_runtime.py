"""Check the pinned core packages and allocated CUDA device without starting Kit."""

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform

import torch


EXPECTED = {
    "torch": "2.7.0+cu128",
    "torchvision": "0.22.0+cu128",
    "isaacsim": "5.1.0.0",
    "isaaclab": "2.3.2.post1",
    "numpy": "1.26.0",
    "onnx": "1.23.1",
    "onnxruntime": "1.23.2",
    "tensorboard": "2.21.0",
    "rl-games": "1.6.1",
}
versions = {name: importlib.metadata.version(name) for name in EXPECTED}
assert versions == EXPECTED, f"Package versions differ: {versions}"
assert platform.python_version() == "3.11.16", platform.python_version()
repo = Path(__file__).resolve().parents[1] / "simtoolreal"
runner = importlib.util.find_spec("rl_games.torch_runner")
assert runner is not None and runner.origin is not None
assert Path(runner.origin).resolve().is_relative_to(repo / "rl_games"), runner.origin
assert torch.cuda.is_available(), "No CUDA device visible inside this allocation"
matrix = torch.randn(256, 256, device="cuda:0")
assert torch.isfinite(matrix @ matrix.T).all().item()
torch.cuda.synchronize()
print(json.dumps({
    "python": platform.python_version(),
    "packages": versions,
    "rl_games_source": runner.origin,
    "gpu": torch.cuda.get_device_name(0),
    "visible_gpu_count": torch.cuda.device_count(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "platform": platform.platform(),
    "libc": platform.libc_ver(),
}, indent=2))
