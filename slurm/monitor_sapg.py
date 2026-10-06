"""Run a Slurm training command, record resources, and honor its RAM budget."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
if not command or not os.environ.get("SLURM_JOB_ID"):
    parser.error("Supply a training command inside a Slurm allocation")
cgroup = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().strip().split(":")[-1].lstrip("/")
allocation = next((p for p in (cgroup, *cgroup.parents) if p.name.startswith("job_")), cgroup)
limit_text = (allocation / "memory.max").read_text().strip()
limit = int(limit_text) if limit_text != "max" else None
process = subprocess.Popen(command, start_new_session=True)
def stop(signum, _frame):
    if process.poll() is None:
        os.killpg(process.pid, signum)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
samples = []
peak = 0
guard = False
next_sample = 0
with args.output.open("w") as output:
    while process.poll() is None:
        memory = int((allocation / "memory.current").read_text())
        peak = max(peak, memory)
        if limit and memory > limit * 0.90:
            guard = True
            print(f"[resources] RAM reached {memory / 2**30:.1f} GiB; stopping this training step before the allocation limit.", flush=True)
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            break
        if time.monotonic() >= next_sample:
            gpu = subprocess.check_output([
                "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ], text=True)
            sample = {
                "time": time.time(), "host_memory_gib": memory / 2**30,
                "gpus": [
                    dict(zip(("index", "memory_mib", "utilization_percent"), map(int, line.split(","))))
                    for line in gpu.strip().splitlines()
                ],
            }
            output.write(json.dumps(sample) + "\n")
            output.flush()
            samples.append(sample)
            next_sample = time.monotonic() + 5
        time.sleep(1)
return_code = process.wait()
summary = {
    "exit_code": return_code, "ram_guard_triggered": guard,
    "host_peak_sampled_gib": peak / 2**30,
    "host_limit_gib": limit / 2**30 if limit else None,
    "gpu_peak_memory_mib": {
        str(i): max((g["memory_mib"] for s in samples for g in s["gpus"] if g["index"] == i), default=0)
        for i in sorted({g["index"] for s in samples for g in s["gpus"]})
    },
    "gpu_peak_utilization_percent": {
        str(i): max((g["utilization_percent"] for s in samples for g in s["gpus"] if g["index"] == i), default=0)
        for i in sorted({g["index"] for s in samples for g in s["gpus"]})
    },
}
args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("[resources] " + json.dumps(summary), flush=True)
raise SystemExit(125 if guard else return_code)
