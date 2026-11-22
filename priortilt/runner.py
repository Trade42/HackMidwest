"""Run decode_file.py chunk jobs with a bounded number of side-by-side processes and memory / heat guards.

  python -I runner.py --name p0.004 --workers 5 --jobs jobs_p0.004.txt

Each line of the jobs file is "<data file> <decoder> <chunk>". One job decodes one 500-shot chunk on a fresh
decoder and writes results/<file stem>.<decoder>.c<chunk>.npz; jobs whose output exists are skipped, so the runner
can be restarted. Guards, checked every 5 s:
  - available RAM (\\Memory\\Available MBytes) under 1.5 GiB: start no new job;
  - under 512 MiB: pause, i.e. terminate the running jobs, put them back in the queue, and wait for 1.5 GiB;
  - GPU temperature (nvidia-smi) at or over 85 C: start no new job until it is under 80 C.
Every job's record (start, end, exit code, processes running at its start and the most during it) goes to
runner_<name>.jsonl; events go to runner_<name>.log. A file named STOP_<name> next to this script makes the runner
finish its running jobs and exit; PAUSE_<name> makes it terminate and requeue them and wait.
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
KIT = os.path.join(ROOT, "repo-main", "challenge-1-ml-decoder-kit")
PT = os.path.join(ROOT, "external", "PriorTiltDecoder")
PY = os.path.join(ROOT, ".venv-cpu", "Scripts", "python.exe")
NO_NEW_MIB, PAUSE_MIB = 1536, 512
HOT_C, COOL_C = 85.0, 80.0


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def available_mib():
    s = MEMORYSTATUSEX()
    s.dwLength = ctypes.sizeof(s)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys / 2**20


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--affinity", default="0x0FFF", help="CPU mask for the runner and its jobs (inherited)")
    ap.add_argument("--results", default="results", help="folder under this script's folder for the result files")
    args = ap.parse_args()
    assert 1 <= args.workers <= 6, "at most 6 processes"
    # The training runs come first: run below normal priority and keep off the top cores. Jobs inherit both.
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p          # a 64-bit pseudo-handle, not a C int
    k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    me = k32.GetCurrentProcess()
    ok_pri = k32.SetPriorityClass(me, 0x00004000)            # BELOW_NORMAL_PRIORITY_CLASS
    ok_aff = k32.SetProcessAffinityMask(me, int(args.affinity, 16))
    if not (ok_pri and ok_aff):
        sys.exit(f"could not set priority ({ok_pri}) or affinity ({ok_aff}): error {k32.GetLastError()}")
    results = os.path.join(HERE, args.results)
    joblogs = os.path.join(HERE, "joblogs", args.results)
    os.makedirs(results, exist_ok=True)
    os.makedirs(joblogs, exist_ok=True)
    log = open(os.path.join(HERE, f"runner_{args.name}.log"), "a", buffering=1)
    rec = open(os.path.join(HERE, f"runner_{args.name}.jsonl"), "a", buffering=1)

    def say(msg):
        log.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    queue = deque()
    for line in open(args.jobs):
        if line.strip() and not line.startswith("#"):
            data, decoder, chunk = line.split()   # decoder: relay, priortilt, or priortilt_s<seed base>
            stem = os.path.splitext(os.path.basename(data))[0]
            out = os.path.join(results, f"{stem}.{decoder}.c{int(chunk):03d}.npz")
            if not os.path.exists(out):
                queue.append((data, decoder, int(chunk), out))
    say(f"start: {len(queue)} jobs to run, workers {args.workers}, pid {os.getpid()}")
    running = {}   # Popen -> job info
    hot = paused = False
    while queue or running:
        avail, temp = available_mib(), gpu_temp()
        if temp is not None:
            hot = temp >= HOT_C or (hot and temp >= COOL_C)
        stop = os.path.exists(os.path.join(HERE, f"STOP_{args.name}"))
        want_pause = os.path.exists(os.path.join(HERE, f"PAUSE_{args.name}")) or avail < PAUSE_MIB
        if want_pause and running:
            say(f"PAUSE (available {avail:.0f} MiB): terminating {len(running)} jobs and requeueing them")
            for proc, info in list(running.items()):
                proc.terminate()
                proc.wait()
                queue.appendleft(info["job"])
                rec.write(json.dumps({**info["meta"], "end": time.time(), "exit": "terminated-requeued"}) + "\n")
            running.clear()
        paused = want_pause or (paused and avail < NO_NEW_MIB)
        for proc, info in list(running.items()):
            info["meta"]["max_side_by_side"] = max(info["meta"]["max_side_by_side"], len(running))
            if proc.poll() is not None:
                meta = {**info["meta"], "end": time.time(), "exit": proc.returncode}
                rec.write(json.dumps(meta) + "\n")
                say(f"done {os.path.basename(info['job'][3])} exit {proc.returncode} in {meta['end'] - meta['start']:.0f} s")
                del running[proc]
        can_start = not (stop or paused or hot) and avail >= NO_NEW_MIB
        while can_start and queue and len(running) < args.workers:
            data, decoder, chunk, out = job = queue.popleft()
            jl = open(os.path.join(joblogs, os.path.basename(out) + ".log"), "w")
            base, _, seed = decoder.partition("_s")
            # -B: never write bytecode (the kit tracks __pycache__ in git, and -I ignores PYTHONDONTWRITEBYTECODE)
            cmd = [PY, "-I", "-B", os.path.join(HERE, "decode_file.py"), "--kit", KIT, "--pt", PT, "--data", data,
                   "--decoder", base, "--chunks", f"{chunk}:{chunk + 1}", "--out", out]
            if seed:
                cmd += ["--seed-base", seed]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            proc = subprocess.Popen(cmd, stdout=jl, stderr=subprocess.STDOUT, env=env, cwd=HERE)
            running[proc] = {"job": job, "meta": {"out": os.path.basename(out), "data": data, "decoder": decoder,
                                                  "chunk": chunk, "start": time.time(), "side_by_side_at_start": len(running) + 1,
                                                  "max_side_by_side": len(running) + 1, "available_mib_at_start": round(avail)}}
            say(f"start {os.path.basename(out)} (running {len(running)}, available {avail:.0f} MiB, gpu {temp} C)")
            avail = available_mib()
            can_start = avail >= NO_NEW_MIB
        if stop and not running:
            break
        time.sleep(5)
    say(f"exit: {len(queue)} jobs left in the queue")


if __name__ == "__main__":
    main()
