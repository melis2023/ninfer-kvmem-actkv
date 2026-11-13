"""Find the largest KVMem window that still starts at a given concurrency.

A 16 GiB card holds 12.5 GiB of weights and leaves ~2.16 GB for KV, so a batch of
`conc` lanes has to fit `(B + R) * conc` tokens of it. Two constraints collide:

  * KVMem carves one (budget + reserve) window per lane, so --kv-capacity is
    `(B + R) * conc` and the engine rejects `capacity > max_context`;
  * the draft ring is clamped to the budget (`recent_tokens = min(..., B)` in
    kvmem_window.cpp), so a budget under 8192 shrinks the drafter's window and
    costs acceptance.

This tool starts the server with each candidate and reports whether it came up,
plus the runtime reservation the engine asked for when it did not. It never
sends a request, so a probe costs one weight load (~25 s), not a decode.

usage:
  python tools/bench/run_capacity_probe.py --model M.ninfer --exe ninfer-serve.exe \
      --concurrency 1,2,4,6,8 --budget 8192,6144,4096 --reserve 2078
"""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def wait_ready(port: int, deadline: float, proc: "subprocess.Popen | None" = None) -> bool:
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        # A capacity that cannot fit dies during startup, long before the deadline;
        # wait the full window only for a process that is still alive.
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.4)
    return False


def probe(exe: Path, model: Path, port: int, budget: int, reserve: int,
          concurrency: int, max_context: int | None, dtype: str, spec: str,
          k: int, wait: float, prefill: int, extra: list[str]) -> dict:
    capacity = (budget + reserve) * concurrency
    text = max(max_context or 0, budget + reserve)
    cmd = [
        str(exe), str(model),
        "--host", "127.0.0.1", "--port", str(port), "--model-id", "probe",
        "--kvmem",
        "--kvmem-budget", str(budget),
        "--kvmem-gen-reserve", str(reserve),
        "--kvmem-host-mib", "12288",
        "--kv-capacity", str(capacity),
        "--max-concurrency", str(concurrency),
        "--kv-dtype", dtype,
        "--max-context", str(text),
        "--prefill-chunk", str(prefill),
        "--spec", spec,
        "--draft-tokens", str(k),
        "--full-draft-head",
        "--ngram-draft-tokens", "0",
        "--temperature", "0",
    ] + extra
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    started = time.time()
    try:
        ready = wait_ready(port, started + wait, proc)
    finally:
        proc.terminate()
        try:
            output = proc.communicate(timeout=30)[0].decode("utf-8", "ignore")
        except subprocess.TimeoutExpired:
            proc.kill()
            output = proc.communicate()[0].decode("utf-8", "ignore")
    need = re.search(r"reservation requires (\d+) bytes", output)
    reason = ""
    for line in output.splitlines():
        if "KVMem requires" in line or "outside the usable range" in line:
            reason = line.strip()
            break
    return {
        "budget": budget,
        "reserve": reserve,
        "concurrency": concurrency,
        "capacity": capacity,
        "max_context": text,
        "ready": ready,
        "reservation_mib": round(int(need.group(1)) / 2**20) if need else None,
        "reason": "" if ready else reason[:70],
        "seconds": round(time.time() - started, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--port", type=int, default=18271)
    ap.add_argument("--concurrency", default="1,2,4")
    ap.add_argument("--budget", default="8192,6144,4096")
    ap.add_argument("--reserve", type=int, default=2048,
                    help="generation reserve; must be 64-aligned and >= prefill-chunk")
    ap.add_argument("--max-context", type=int, default=None,
                    help="default: exactly B+R, the value the engine actually compares")
    ap.add_argument("--prefill-chunk", type=int, default=512,
                    help="KVMem requires prefill <= reserve")
    ap.add_argument("--kv-dtype", default="nvfp4")
    ap.add_argument("--spec", default="dflash")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--extra", default="",
                    help="raw flags appended to every launch, e.g. "
                         "\"--no-prefix-reuse --device-state-slots 0\"")
    ap.add_argument("--wait", type=float, default=120.0, help="seconds to wait for readiness")
    args = ap.parse_args()
    extra = args.extra.split() if args.extra else []

    rows = []
    port = args.port
    for concurrency in [int(value) for value in args.concurrency.split(",")]:
        for budget in [int(value) for value in args.budget.split(",")]:
            row = probe(Path(args.exe), Path(args.model), port, budget,
                        args.reserve, concurrency, args.max_context, args.kv_dtype,
                        args.spec, args.k, args.wait, args.prefill_chunk, extra)
            row["extra"] = args.extra
            rows.append(row)
            # A fresh port per probe: a server left behind by the previous launch
            # would answer /v1/models and turn an OOM into a false "ready".
            port += 1
            print(json.dumps(row), flush=True)
            if row["ready"]:
                # Larger budgets at the same concurrency will only ask for more.
                break

    header = (f"{'bs':>2} {'B':>6} {'R':>5} {'cap':>6} {'ctx':>6} {'ready':>5} "
              f"{'need MiB':>9}  reason")
    print(header)
    for row in rows:
        need = "-" if row["reservation_mib"] is None else f"{row['reservation_mib']}"
        print(f"{row['concurrency']:2} {row['budget']:6} {row['reserve']:5} "
              f"{row['capacity']:6} {row['max_context']:6} "
              f"{'yes' if row['ready'] else 'no':>5} {need:>9}  {row['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
