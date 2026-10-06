"""torchrun worker: separate Kit caches/logs and Hydra metadata for each rank."""
from pathlib import Path
import os
import sys

root = Path(__file__).resolve().parents[1]
rank = int(os.environ["LOCAL_RANK"])
output = Path(os.environ["SIMTOOLREAL_RUN_DIR"])
scratch = Path(os.environ["SIMTOOLREAL_SCRATCH"]) / f"rank_{rank}"
for name in ("tmp", "cache", "omniverse", "kit"):
    (scratch / name).mkdir(parents=True, exist_ok=True)
os.environ.update(
    TMPDIR=str(scratch / "tmp"),
    XDG_CACHE_HOME=str(scratch / "cache"),
    OMNI_KIT_CACHE_PATH=str(scratch / "omniverse"),
)
output.mkdir(parents=True, exist_ok=True)
log = os.open(output / f"rank_{rank}.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
os.dup2(log, 1)
os.dup2(log, 2)
os.close(log)
threads = os.environ["OMP_NUM_THREADS"]
kit_args = (
    f"--portable-root {scratch / 'kit'} "
    f"--/plugins/carb.tasking.plugin/threadCount={threads}"
)
args = [
    str(root / "simtoolreal/isaacsimenvs/train.py"),
    *sys.argv[1:],
    f"--kit_args={kit_args}",
    f"hydra.run.dir={output / f'rank_{rank}'}",
]
os.execv(sys.executable, [sys.executable, *args])
