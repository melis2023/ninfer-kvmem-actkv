"""Attribute a DSpine acceptance drop to KVMem's window, not to the engine.

The question this answers: DSpine's acceptance falls 56.5% -> 44.5% -> 35.8% as the
context grows, while DFlash2's stays flat. Is that the engine (ninfer) or the KV
window approximation (KVMem)?

The method: hold everything fixed -- same binary, same artifact, same prompt, same
kv dtype, same prefill chunk, same k -- and change only `--kvmem-budget` /
`--kvmem-gen-reserve`, i.e. how much history the target can still see. If acceptance
tracks the window rather than the context length, KVMem owns the effect. If it does
not move, the engine or the KV precision owns it and the next arm to run is bf16.

The prompt is built from this repository's own CUDA sources so the run is
reproducible without shipping a corpus.

usage:
  python tools/bench/run_kvmem_budget_ab.py --model MODEL.ninfer --exe PATH/ninfer-serve.exe
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def build_prompt(root: Path, chars: int) -> str:
    files = sorted((root / "backends/ninfer/src/ops").rglob("*.cu"))
    files += sorted((root / "backends/ninfer/src/ops").rglob("*.cpp"))
    parts, total = [], 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        parts.append(f"\n// ---- {path.name} ----\n{text}")
        total += len(parts[-1])
        if total >= chars:
            break
    if total < chars:
        raise SystemExit(f"only {total} chars of source available, need {chars}")
    body = "".join(parts)[:chars]
    return (
        body
        + "\n\n// A reviewer asked for a bounds check on the kernel above.\n"
        + "// Rewrite `dspine_draw_rank` with an explicit guard and a short comment\n"
        + "// explaining why the transition table cannot be indexed out of range.\n"
    )


def wait_ready(port: int, deadline: float) -> bool:
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    return False


def read_metrics(path: Path) -> dict:
    """Walk every record and keep the deepest block carrying rounds + accepted."""
    best = {}

    def walk(node):
        if isinstance(node, dict):
            if "rounds" in node and "accepted_tokens" in node:
                if node.get("rounds", 0) and node["rounds"] > best.get("rounds", 0):
                    for key in ("rounds", "drafted_tokens", "accepted_tokens"):
                        if key in node:
                            best[key] = node[key]
                    if "accepted_per_position" in node:
                        best["accepted_per_position"] = node["accepted_per_position"]
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                walk(json.loads(line))
            except json.JSONDecodeError:
                continue
    return best


def run_arm(exe: Path, model: Path, prompt: str, arm: dict, port: int,
            out: Path, max_context: int, spec: str = "dflash",
            full_head: bool = True, concurrency: int = 1,
            prefill: int = 512, max_tokens: int = 512,
            reasoning: str = "", extra: list | None = None) -> dict:
    log = out / f"{arm['name']}.jsonl"
    stdout = out / f"{arm['name']}.out"
    if log.exists():
        log.unlink()
    # KVMem carves one (budget + reserve) window per lane, so the pool has to grow
    # with the batch. The engine rejects capacity > max_context, hence the callers
    # pass max_context = (B+R) x concurrency for these runs.
    capacity = (arm["budget"] + arm["reserve"]) * concurrency
    cmd = [
        str(exe), str(model),
        "--host", "127.0.0.1", "--port", str(port),
        "--model-id", "attr",
    ]
    if arm["budget"]:
        cmd += [
        "--kvmem",
        "--kvmem-budget", str(arm["budget"]),
        "--kvmem-gen-reserve", str(arm["reserve"]),
        "--kvmem-host-mib", "12288",
        "--kv-capacity", str(capacity),
        "--max-concurrency", str(concurrency),
        ]
    cmd += [
        "--kv-dtype", arm.get("dtype", "nvfp4"),
        "--max-context", str(max_context),
        "--prefill-chunk", str(prefill),
        "--spec", spec,
        "--draft-tokens", str(arm.get("k", 4)),
        ] + (["--full-draft-head"] if full_head else []) + [
        "--ngram-draft-tokens", "0",
        "--temperature", "0",
        "--request-log-jsonl", str(log),
    ] + (extra or [])
    with open(stdout, "wb") as handle:
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    payloads: list = [None] * concurrency
    decode_tps: list = [0.0] * concurrency

    def one(index: int) -> None:
        # Distinct tails so the lanes do not share a prompt cache entry and decode
        # genuinely in parallel rather than collapsing onto one prefix. A single lane
        # gets the prompt verbatim.
        text = prompt if concurrency == 1 else f"{prompt}\n// lane {index}\n"
        request = {
            "model": "attr",
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
        if reasoning:
            # Otherwise the server's medium default spends most of a 2K budget thinking
            # and the completion is a fragment of the artifact being timed.
            request["reasoning_effort"] = reasoning
        body = json.dumps(request).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            payloads[index] = json.loads(r.read().decode("utf-8"))

    try:
        if not wait_ready(port, time.time() + 300):
            tail = stdout.read_text(encoding="utf-8", errors="ignore")[-1500:]
            raise SystemExit(f"{arm['name']}: server never became ready\n{tail}")
        started = time.time()
        threads = [threading.Thread(target=one, args=(i,)) for i in range(concurrency)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall = time.time() - started
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    text = stdout.read_text(encoding="utf-8", errors="ignore")
    # The server's own tail line is the authoritative decode rate: it excludes
    # prefill, which the wall clock above does not. One line per lane.
    for line in text.splitlines():
        found = re.search(r"req#(\d+) done .*?decode ([\d.]+) tok/s", line)
        if found:
            index = int(found.group(1)) - 1
            if 0 <= index < concurrency:
                decode_tps[index] = float(found.group(2))
    served = [re.search(r"req#(\d+) done .*?dflash accepted ([\d,]+)/([\d,]+)", line)
              for line in text.splitlines()]
    served = [m for m in served if m and 1 <= int(m.group(1)) <= concurrency]
    metrics = read_metrics(log)
    rounds = metrics.get("rounds") or 0
    accepted = metrics.get("accepted_tokens") or 0
    drafted = metrics.get("drafted_tokens") or 0
    if served:
        # The server prints thousands separators: 806/1,296 is nine hundred accepted
        # out of twelve hundred, not 806 out of 1.
        accepted = sum(int(m.group(2).replace(",", "")) for m in served)
        drafted = sum(int(m.group(3).replace(",", "")) for m in served)
    usage = (payloads[0] or {}).get("usage") or {}
    choices = (payloads[0] or {}).get("choices") or [{}]
    content = ((choices[0] or {}).get("message") or {}).get("content") or ""
    prompt_tokens = usage.get("prompt_tokens")
    return {
        "name": arm["name"],
        "budget": arm["budget"],
        "reserve": arm["reserve"],
        "capacity": capacity,
        "concurrency": concurrency,
        "dtype": arm.get("dtype", "nvfp4"),
        "prompt_tokens": prompt_tokens,
        "rounds": rounds,
        "drafted": drafted,
        "accepted": accepted,
        "accept_rate": (accepted / drafted) if drafted else 0.0,
        # `rounds` is one lane's round count while `accepted` is summed over lanes,
        # so divide per-lane to keep the round length comparable across batch sizes.
        "tokens_per_round": (1.0 + accepted / (rounds * concurrency)) if rounds else 0.0,
        "decode_tps": sum(decode_tps),
        "decode_each": decode_tps,
        "wall": wall,
        "completion_tokens": usage.get("completion_tokens"),
        "content": content,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--port", type=int, default=18251)
    ap.add_argument("--chars", type=int, default=130000,
                    help="prompt size in characters (~3.5 chars per code token)")
    ap.add_argument("--max-context", type=int, default=40960)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--arms", default=None,
                    help="override the arm list: name:budget:reserve,name:budget:reserve")
    ap.add_argument("--spec", default="dflash")
    ap.add_argument("--no-full-head", action="store_true")
    ap.add_argument("--with-full", action="store_true",
                    help="add an arm with KVMem disabled entirely (exact attention)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="lanes decoded together; --kv-capacity becomes (B+R) x this")
    ap.add_argument("--prefill-chunk", type=int, default=512,
                    help="also a memory lever (~0.36 MiB per token) and a quality one")
    ap.add_argument("--prompt", default=None,
                    help="use this prompt instead of the one built from this repo's sources; "
                         "keeps the arm machinery for a real-task timing")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--reasoning-effort", default="",
                    help="none|minimal|low|medium|high; empty keeps the server default")
    ap.add_argument("--extra", default="",
                    help="raw flags appended to the server launch, e.g. --no-cuda-graph")
    ap.add_argument("--save-output", default=None,
                    help="write the completion of the first arm to this path")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[4]
    out = Path(args.out) if args.out else root / "build-backends/ninfer/attr"
    out.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt if args.prompt else build_prompt(root, args.chars)

    arms = [
        {"name": "tight", "budget": 12288, "reserve": 4096},
        {"name": "wide", "budget": 36480, "reserve": 16384},
    ]
    if args.with_full:
        arms.append({"name": "full", "budget": 0, "reserve": 0})
    if args.arms:
        arms = []
        for spec in args.arms.split(","):
            fields = spec.split(":")
            arm = {"name": fields[0], "budget": int(fields[1]), "reserve": int(fields[2])}
            if len(fields) > 3:
                arm["dtype"] = fields[3]
            arms.append(arm)
    rows = []
    for arm in arms:
        arm["k"] = args.k
        # The engine compares the per-lane window B+R against --max-context and lets
        # the pool be (B+R) x concurrency, so only the window has to fit.
        if arm["budget"] and arm["budget"] + arm["reserve"] > args.max_context:
            raise SystemExit(f"{arm['name']}: window {arm['budget'] + arm['reserve']} "
                             f"exceeds --max-context {args.max_context}")
        row = run_arm(Path(args.exe), Path(args.model), prompt, arm,
                      args.port, out, args.max_context,
                      args.spec, not args.no_full_head, args.concurrency,
                      args.prefill_chunk, args.max_tokens, args.reasoning_effort,
                      args.extra.split() if args.extra else [])
        rows.append(row)
        content = row.pop("content", "")
        if args.save_output and content:
            Path(args.save_output).write_text(content, encoding="utf-8")
            print(f"output -> {args.save_output} ({len(content)} chars)", flush=True)
        print(json.dumps(row), flush=True)

    (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    header = (f"{'arm':6} {'B':>6} {'R':>6} {'cap':>6} {'bs':>2} {'prompt':>7} "
              f"{'acc/round':>10} {'rate':>7} {'tok/s':>7}")
    print(header)
    for row in rows:
        print(f"{row['name']:6} {row['budget']:6} {row['reserve']:6} {row['capacity']:6} "
              f"{row['concurrency']:2} {row['prompt_tokens']:7} "
              f"{row['tokens_per_round']:10.2f} "
              f"{row['accept_rate']*100:6.1f}% {row['decode_tps']:7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
