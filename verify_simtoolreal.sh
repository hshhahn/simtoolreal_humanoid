#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/activate_simtoolreal.sh"
export PYTHONUNBUFFERED=1
python - <<'PY'
import importlib.metadata
import importlib.util
import sys
from pathlib import Path

import torch

print('Python:', sys.version.split()[0])
for package in ('torch', 'torchvision', 'isaaclab', 'isaacsim', 'numpy', 'rl_games'):
    print(f'{package}: {importlib.metadata.version(package)}')
assert torch.cuda.is_available(), 'CUDA is unavailable; run on the GPU host outside the filesystem sandbox'
print('GPU:', torch.cuda.get_device_name(0))
x = torch.randn(256, 256, device='cuda')
assert torch.isfinite(x @ x.T).all().item()
torch.cuda.synchronize()
runner_spec = importlib.util.find_spec('rl_games.torch_runner')
assert runner_spec is not None and runner_spec.origin is not None
assert Path(runner_spec.origin).resolve().is_relative_to(Path.cwd() / 'rl_games')
print('CUDA computation and vendored rl_games: PASS')
PY
# Kit must run in separate processes, one at a time on this GPU.
python isaacsimenvs/tests/test_gym_register.py
python isaacsimenvs/tests/test_obs_action_spec.py \
    --num_envs 4 --num_assets_per_type 1 --steps 10
python isaacsimenvs/tests/test_pretrained_rollout.py \
    --num_envs 8 --num_assets_per_type 2 --num_steps 600
python - <<'PY'
import asyncio
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import websockets
import viser

workspace = Path.cwd().parent
log_path = workspace / '.logs/demo-http-check.log'
log_path.parent.mkdir(parents=True, exist_ok=True)
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
url = f'http://127.0.0.1:{port}/'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

async def check_websocket():
    async with websockets.connect(
        f'ws://127.0.0.1:{port}/', open_timeout=5,
        subprotocols=[f'viser-v{viser.__version__}'],
        max_size=64 * 1024 * 1024, compression=None,
    ) as connection:
        packet = await asyncio.wait_for(connection.recv(), timeout=5)
        assert isinstance(packet, bytes) and packet, 'No scene/GUI data received'

with log_path.open('w') as log:
    process = subprocess.Popen(
        [str(workspace / 'run_simtoolreal_demo.sh'), '--port', str(port)],
        cwd=workspace, stdin=subprocess.DEVNULL, stdout=log,
        stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONUNBUFFERED': '1'},
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f'Demo exited with code {process.returncode}; see {log_path}')
            # The server starts before the constructor builds its GUI and robot.
            # Wait until the main loop's banner confirms construction completed.
            if '  |     DexToolBench Interactive Policy Demo' not in log_path.read_text():
                time.sleep(0.2)
                continue
            try:
                with opener.open(url, timeout=2) as response:
                    content = response.read()
                    assert response.status == 200 and b'<html' in content.lower()
                asyncio.run(check_websocket())
                print('Interactive demo HTTP and WebSocket checks: PASS')
                break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.2)
        else:
            raise TimeoutError('Demo did not serve its web interface within 30 seconds')
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
PY
printf '\nSimToolReal verification: PASS\n'
