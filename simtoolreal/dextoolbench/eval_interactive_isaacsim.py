"""DexToolBench Interactive Policy Demo — Isaac Sim (Isaac Lab) backend.

Same viser UI as ``eval_interactive_isaacgym.py`` (reused directly); only the simulator
child differs: a plain subprocess running ``_isaacsim_interactive_worker.py``,
which boots Kit via AppLauncher and runs ``Isaacsimenvs-SimToolReal-Direct-v0``.
A plain subprocess (not multiprocessing) is required — Kit segfaults at boot
inside a multiprocessing-spawned child. The same command protocol flows over a
localhost socket Connection instead of a Pipe.

Run inside the Isaac Sim venv:

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python dextoolbench/eval_interactive_isaacsim.py \\
        --config-path pretrained_policy/config.yaml \\
        --checkpoint-path pretrained_policy/model.pth

Notes:
- Each "Load Environment" click spawns a fresh worker (Kit cannot cleanly
  tear down in-process). Kit boot takes ~1 min.
- Run only one Isaac Sim instance per GPU at a time.
"""

from __future__ import annotations

import argparse
import os
import secrets
import subprocess
import sys
from multiprocessing.connection import Listener
from pathlib import Path

_WORKER_SCRIPT = Path(__file__).resolve().parent / "_isaacsim_interactive_worker.py"


class _PopenProc:
    """Adapter giving subprocess.Popen the Process-ish API the UI expects."""

    def __init__(self, popen: subprocess.Popen):
        self._popen = popen
        self.pid = popen.pid

    def is_alive(self) -> bool:
        return self._popen.poll() is None

    def kill(self) -> None:
        self._popen.kill()

    def join(self, timeout: float | None = None) -> None:
        try:
            self._popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass


def isaacsim_worker_factory(category, object_name, task_name, table_urdf,
                            config_path, checkpoint_path):
    """Spawn the Kit worker as a plain subprocess; return (proc, conn)."""
    authkey = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    host, port = listener.address

    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    repo_root = str(_WORKER_SCRIPT.parents[1])
    env["PYTHONPATH"] = repo_root + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    popen = subprocess.Popen(
        [
            sys.executable, str(_WORKER_SCRIPT),
            "--address", f"{host}:{port}",
            "--authkey", authkey.hex(),
            "--category", category,
            "--object_name", object_name,
            "--task_name", task_name,
            "--table_urdf", table_urdf,
            "--config_path", str(config_path),
            "--checkpoint_path", str(checkpoint_path),
        ],
        env=env,
        cwd=repo_root,
    )
    # Worker connects immediately on start (before Kit boots), so this blocks
    # only briefly.
    conn = listener.accept()
    listener.close()
    return _PopenProc(popen), conn


if __name__ == "__main__":
    from dextoolbench.eval_interactive_isaacgym import InteractiveDemo

    parser = argparse.ArgumentParser(
        description="DexToolBench Interactive Policy Demo (Isaac Sim backend)",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--config-path", type=str, default="pretrained_policy/config.yaml",
        help="Path to the policy config YAML",
    )
    parser.add_argument(
        "--checkpoint-path", type=str, default="pretrained_policy/model.pth",
        help="Path to the policy checkpoint",
    )
    args = parser.parse_args()
    InteractiveDemo(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        port=args.port,
        worker_factory=isaacsim_worker_factory,
    ).run()
