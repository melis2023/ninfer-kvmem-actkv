#!/usr/bin/env python3
"""Typed decision endpoints, interleaved server configurations under an external SystemOne bench.

usage: decision_ab.py --bench DIR --model PATH --server NAME=PATH [--server NAME=PATH ...]
                      [--config NAME:SERVER[:FLAG,FLAG...] ...] [--conc 1,4] [--rounds 2]
                      [--out DIR] [--cpus LIST] [--flag OPTION ...]

`--flag` adds a server option (its value after a space) to every configuration, such as the model's
`--kv-capacity` or `--ngram-table`; a configuration without `--kv-dtype` runs int8.

Each round runs every configuration, in an order rotated by the round, under the bench's
`node bench.mjs --conc N` against /v1/systemone for every N, each on a server of its own, so no run
answers from another's context cache or branch memo (the bench directory needs bench.mjs and
data.mjs; it warms up with three cases of its own). Per run it records the bench's wall time,
per-case latencies and answers, and from the server's request log every read's prompt and cached
tokens, time to first token and queue time. Writes DIR/summary.json and DIR/summary.md: each
configuration's medians over the rounds, the new (uncached) prompt tokens per case, and how far its
answers move from the first configuration's.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PORT = int(os.environ.get("DECISION_AB_PORT", "18092"))
BASE = f"http://127.0.0.1:{PORT}"
COMMON = ["--host", "127.0.0.1", "--port", str(PORT), "--device", "0", "--max-context", "16384",
          "--max-concurrency", "4", "--concurrent-prefill"]
# Options only the checkpoint catalog serves, as the field report ran it; naming any selects it.
CATALOG = ["--auto-prefix-grid", "--auto-long-anchors", "--derive-session-keys"]
# The bench warms up with three triage cases before its timed cases.
WARMUP_CASES = 3
# Token counts of 1,000 and more are logged with thousands separators.
DONE = re.compile(r"req#(\d+) done \|.*?\| prompt ([\d,]+) \| output ([\d,]+) \| cache ([\d,]+)"
                  r"(?: \(([^)]*)\))?.*?\| TTFT ([\d.]+) ?(ms|s)(?:.*?\| queue ([\d.]+) ?(ms|s))?")
RUNNING: list[subprocess.Popen] = []


def stop_servers(*_) -> None:
    for process in RUNNING:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=60)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    RUNNING.clear()


def handle_signal(number, _frame) -> None:
    stop_servers()
    raise SystemExit(128 + number)


def healthy() -> bool:
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return False


def start_server(binary: Path, model: Path, flags: list[str], log: Path,
                 cpus: str | None) -> subprocess.Popen:
    kv = [] if "--kv-dtype" in flags else ["--kv-dtype", "int8"]
    command = [str(binary), str(model), *COMMON, *kv, *flags]
    if cpus:
        command = ["taskset", "-c", cpus, *command]
    log.write_text("")
    process = subprocess.Popen(command, stdout=log.open("w"), stderr=subprocess.STDOUT,
                               start_new_session=True)
    RUNNING.append(process)
    started = time.time()
    while not healthy():
        if process.poll() is not None:
            raise RuntimeError(f"server exited {process.returncode}; see {log}")
        if time.time() - started > 900:
            raise RuntimeError("server start timed out")
        time.sleep(2)
    print(f"  server up in {time.time() - started:.1f} s: {' '.join(flags) or 'defaults'}",
          flush=True)
    return process


def seconds(value: str | None, unit: str | None) -> float | None:
    if value is None:
        return None
    return float(value) / 1000 if unit == "ms" else float(value)


def reads(log_text: str) -> list[dict]:
    out = []
    for match in DONE.finditer(log_text):
        prompt, output, cache = (int(match[group].replace(",", "")) for group in (2, 3, 4))
        out.append({"prompt": prompt, "output": output, "cache": cache,
                    "cache_kind": match[5] or "", "ttft": seconds(match[6], match[7]),
                    "queue": seconds(match[8], match[9]) or 0.0})
    return out


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low, high = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def run_bench(bench: Path, run_dir: Path, conc: int) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("bench.mjs", "data.mjs"):
        shutil.copy(bench / name, run_dir / name)
    env = {**os.environ, "SEB_URL": BASE + "/v1/systemone"}
    started = time.time()
    with (run_dir / "bench.log").open("w") as out:
        rc = subprocess.run(["node", "bench.mjs", "--conc", str(conc)], cwd=run_dir, env=env,
                            stdout=out, stderr=subprocess.STDOUT).returncode
    wall = time.time() - started
    if rc != 0:
        raise RuntimeError(f"bench exited {rc}; see {run_dir / 'bench.log'}")
    results = sorted(run_dir.glob("results-*.json"))[-1]
    raw = json.loads(results.read_text())
    cases = [row for scenario in ("A", "B", "C") for row in raw[scenario]]
    errors = sum(1 for row in cases if row.get("error"))
    latencies = [row["latency"] / 1000 for row in cases if row.get("latency") is not None]
    answers = {f"{scenario}:{row['id']}": row.get("answers") for scenario in ("A", "B", "C")
               for row in raw[scenario]}
    return {"wall": wall, "cases": len(cases), "errors": errors, "latencies": latencies,
            "answers": answers}


def probabilities(answer: dict) -> dict[str, float]:
    probs = answer.get("probabilities")
    if isinstance(probs, dict):
        return {str(k): float(v) for k, v in probs.items()}
    if answer.get("type") == "noul" and "noul" in answer:
        return {"true": float(answer["noul"])}
    return {}


def winner(answer: dict):
    for key in ("choice", "score"):
        if key in answer:
            return answer[key] if key == "choice" else max(
                probabilities(answer).items(), key=lambda item: item[1])[0]
    return answer.get("noul", 0) >= 0.5


def compare(base: dict, other: dict) -> dict:
    questions = flips = 0
    worst = 0.0
    for case, answers in base.items():
        if not answers or not other.get(case):
            continue
        for qid, answer in answers.items():
            twin = other[case].get(qid)
            if twin is None:
                continue
            questions += 1
            flips += winner(answer) != winner(twin)
            a, b = probabilities(answer), probabilities(twin)
            for key in a.keys() & b.keys():
                worst = max(worst, abs(a[key] - b[key]))
    return {"questions": questions, "argmax_changes": flips, "max_probability_change": worst}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--server", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--config", action="append", required=True,
                        help="NAME:SERVER[:FLAG,FLAG...]; 'catalog' expands to the catalog options")
    parser.add_argument("--conc", default="1,4")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("decision-ab"))
    parser.add_argument("--cpus", help="taskset CPU list for every server")
    parser.add_argument("--flag", action="append", default=[],
                        help="a server option for every configuration, its value after a space")
    args = parser.parse_args()
    shared = [part for flag in args.flag for part in flag.split(" ")]
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    servers = dict(item.split("=", 1) for item in args.server)
    configs = []
    for item in args.config:
        name, server, *rest = item.split(":", 2)
        flags = []
        for flag in (rest[0].split(",") if rest and rest[0] else []):
            flags += CATALOG if flag == "catalog" else flag.split(" ")
        configs.append({"name": name, "binary": Path(servers[server]), "flags": shared + flags})
    concs = [int(c) for c in args.conc.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict[int, list[dict]]] = {c["name"]: {n: [] for n in concs} for c in configs}
    try:
        for round_index in range(args.rounds):
            order = configs[round_index % len(configs):] + configs[:round_index % len(configs)]
            for config in order:
                tag = f"r{round_index}-{config['name']}"
                print(f"round {round_index} {config['name']}", flush=True)
                for conc in concs:
                    log = args.out / f"{tag}-c{conc}.server.log"
                    start_server(config["binary"], args.model, config["flags"], log, args.cpus)
                    result = run_bench(args.bench, args.out / f"{tag}-c{conc}", conc)
                    time.sleep(1)
                    stop_servers()
                    logged = reads(log.read_text(errors="replace"))
                    # A branched read logs one request for a case's questions, a window one per
                    # question: the warm-up's share follows from the count.
                    warmup = len(logged) * WARMUP_CASES // (result["cases"] + WARMUP_CASES)
                    result["reads"] = logged[warmup:]
                    runs[config["name"]][conc].append(result)
                    new = sum(r["prompt"] - r["cache"] for r in result["reads"])
                    print(f"  conc {conc}: wall {result['wall']:.1f} s, errors {result['errors']}, "
                          f"reads {len(result['reads'])}, new prompt tokens {new}", flush=True)
    finally:
        stop_servers()
    base = configs[0]["name"]
    summary = {"configs": {}, "base": base, "common_flags": COMMON}
    lines = ["| configuration | conc | wall s (rounds) | case p50 / p90 s | new prompt tokens per case "
             "| cached share | read TTFT p50 ms | queue p50 ms | argmax changes vs " + base +
             " | max prob change |", "|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for config in configs:
        entry = {"flags": config["flags"], "binary": str(config["binary"]), "conc": {}}
        for conc in concs:
            done = runs[config["name"]][conc]
            latencies = [x for r in done for x in r["latencies"]]
            all_reads = [x for r in done for x in r["reads"]]
            measured_cases = sum(r["cases"] for r in done)
            prompt = sum(x["prompt"] for x in all_reads)
            cached = sum(x["cache"] for x in all_reads)
            changes = [compare(runs[base][conc][i]["answers"], r["answers"])
                       for i, r in enumerate(done) if i < len(runs[base][conc])]
            stats = {
                "walls": [round(r["wall"], 2) for r in done],
                "errors": sum(r["errors"] for r in done),
                "case_p50": percentile(latencies, 0.5), "case_p90": percentile(latencies, 0.9),
                "reads": len(all_reads), "prompt_tokens": prompt, "cached_tokens": cached,
                "new_tokens_per_case": (prompt - cached) / max(1, measured_cases),
                "read_ttft_p50": percentile([x["ttft"] for x in all_reads], 0.5),
                "read_queue_p50": percentile([x["queue"] for x in all_reads], 0.5),
                "cache_kinds": sorted({x["cache_kind"] for x in all_reads}),
                "against_base": changes,
            }
            entry["conc"][conc] = stats
            flips = sum(c["argmax_changes"] for c in changes)
            questions = sum(c["questions"] for c in changes)
            worst = max((c["max_probability_change"] for c in changes), default=0.0)
            lines.append(
                f"| {config['name']} | {conc} | {', '.join(f'{w:.1f}' for w in stats['walls'])} | "
                f"{stats['case_p50']:.3f} / {stats['case_p90']:.3f} | "
                f"{stats['new_tokens_per_case']:.0f} | {cached / max(1, prompt):.1%} | "
                f"{stats['read_ttft_p50'] * 1000:.0f} | {stats['read_queue_p50'] * 1000:.0f} | "
                f"{flips}/{questions} | {worst:.4f} |")
        summary["configs"][config["name"]] = entry
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    errors = sum(s["errors"] for e in summary["configs"].values() for s in e["conc"].values())
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
