#!/usr/bin/env python3
"""Interleaved A/B of ninfer-serve configurations: decode and prefill rates, time to first token,
expert-cache traffic and the answers themselves.

usage: serve_ab.py PLAN.json --out DIR

PLAN.json:
  {"configs": [{"label": "base", "server": "/path/ninfer-serve", "model": "/path/m.ninfer",
                "args": ["--expert-residency", "host"], "env": {"NAME": "value"}}, ...],
   "prompts": ["code", "prose", "zh"],        # built-in names, or keys of "prompt_files"
   "prompt_files": {"long": "/path/prompt.txt"},
   "repeat": 2,            # passes over the prompts per server start: pass 2 revisits pass 1
   "rounds": 2,            # server starts per configuration; round r starts at config r mod n
   "max_tokens": 256,
   "lock_clocks_mhz": 0}   # nonzero: nvidia-smi -lgc for the run, if the host allows it

Every request is greedy with EOS ignored and thinking off. Each configuration's server runs in its
own process group, which is ended on exit, on SIGTERM, SIGINT and SIGHUP. DIR receives one JSON per
server start and summary.json/summary.md: per configuration, pass and prompt, the mean over rounds
of every rate with its range, the change against the first configuration, and whether the answer
repeated across rounds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPTS = {
    "code": "Write a thread-safe LRU cache in Python with a per-entry TTL expiry, with docstrings "
            "and a short usage example.",
    "prose": "Explain to a curious teenager why the sky is blue during the day and red or orange "
             "at sunset, in a few friendly paragraphs.",
    "zh": "请详细介绍长城的历史：最早修建的原因、各个朝代的修建与扩展、它在军事上的作用以及今天的文化意义。",
    "math": "Solve step by step: how many positive integers less than 1000 are divisible by 7 but "
            "not by 11?",
}
PORT = int(os.environ.get("SERVE_AB_PORT", "18090"))
BASE = f"http://127.0.0.1:{PORT}"
RUNNING: list[subprocess.Popen] = []


def call(path: str, body: dict | None = None, timeout: float = 1800) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(BASE + path, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


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


def run_server(config: dict, plan: dict, texts: dict, out: Path, round_index: int) -> dict:
    label = config["label"]
    command = [config["server"], config["model"], "--host", "127.0.0.1", "--port", str(PORT),
               *config.get("args", [])]
    record = {"label": label, "round": round_index, "command": command,
              "env": config.get("env", {}), "samples": []}
    log_path = out / f"{label}-r{round_index}.server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, **config.get("env", {})},
                                   start_new_session=True)
        RUNNING.append(process)
        started = time.time()
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{label}: server exited {process.returncode}")
                try:
                    call("/health", timeout=5)
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                    if time.time() - started > 1800:
                        raise RuntimeError(f"{label}: server start timed out") from None
                    time.sleep(2)
            record["startup_seconds"] = round(time.time() - started, 1)
            for repeat in range(plan.get("repeat", 1)):
                for name in plan["prompts"]:
                    before = call("/stats")
                    begun = time.time()
                    response = call("/v1/chat/completions", {
                        "messages": [{"role": "user", "content": texts[name]}],
                        "max_tokens": plan.get("max_tokens", 256), "temperature": 0,
                        "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}})
                    wall = time.time() - begun
                    after = call("/stats")
                    timings = response.get("timings", {})
                    old, new = before.get("experts", {}), after.get("experts", {})
                    experts = {k: new[k] - old.get(k, 0) for k in new
                               if isinstance(new.get(k), (int, float))}
                    text = response["choices"][0]["message"].get("content", "")
                    sample = {"pass": repeat, "prompt": name, "wall": round(wall, 3),
                              "prompt_tokens": timings.get("prompt_n"),
                              "prefill_tok_s": timings.get("prompt_per_second"),
                              "ttft_ms": timings.get("prompt_ms"),
                              "decode_tokens": timings.get("predicted_n"),
                              "decode_tok_s": timings.get("predicted_per_second"),
                              "experts": experts,
                              "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                              "text": text}
                    record["samples"].append(sample)
                    routes = max(experts.get("routes", 0), 1)
                    print(f"{label} r{round_index} p{repeat} {name:6s} prompt "
                          f"{sample['prompt_tokens']} @ {sample['prefill_tok_s'] or 0:8.1f} tok/s "
                          f"ttft {sample['ttft_ms'] or 0:7.1f} ms | decode "
                          f"{sample['decode_tokens']} @ {sample['decode_tok_s'] or 0:7.2f} tok/s | "
                          f"hit {experts.get('hits', 0) / routes:.3f} xfer "
                          f"{experts.get('transferred_bytes', 0) / 2**20:.0f} MiB", flush=True)
        finally:
            stop_servers()
    (out / f"{label}-r{round_index}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
    return record


def summarize(plan: dict, records: list[dict]) -> dict:
    labels = [config["label"] for config in plan["configs"]]
    keys = sorted({(s["pass"], s["prompt"]) for r in records for s in r["samples"]},
                  key=lambda k: (k[0], plan["prompts"].index(k[1])))
    table = {}
    for label in labels:
        for key in keys:
            samples = [s for r in records if r["label"] == label for s in r["samples"]
                       if (s["pass"], s["prompt"]) == key]
            if not samples:
                continue
            row = {"runs": len(samples),
                   "same_answer": len({s["text_sha256"] for s in samples}) == 1}
            for metric in ("decode_tok_s", "prefill_tok_s", "ttft_ms"):
                values = [s[metric] for s in samples if s[metric] is not None]
                if values:
                    row[metric] = {"mean": statistics.fmean(values), "min": min(values),
                                   "max": max(values)}
            transferred = [s["experts"].get("transferred_bytes", 0) for s in samples]
            row["transferred_mib"] = statistics.fmean(transferred) / 2**20
            table[f"{label}|{key[0]}|{key[1]}"] = row
    return {"labels": labels, "keys": [list(k) for k in keys], "rows": table}


def markdown(summary: dict) -> str:
    labels, rows = summary["labels"], summary["rows"]
    lines = ["| pass | prompt | config | decode tok/s (range) | Δ decode | TTFT ms (range) | Δ TTFT | "
             "prefill tok/s | MiB moved | same answer |",
             "|---:|---|---|---|---:|---|---:|---:|---:|---|"]
    for pass_index, prompt in summary["keys"]:
        base = rows.get(f"{labels[0]}|{pass_index}|{prompt}")
        for label in labels:
            row = rows.get(f"{label}|{pass_index}|{prompt}")
            if row is None:
                continue
            def cell(metric):
                m = row.get(metric)
                return "-" if m is None else f"{m['mean']:.1f} ({m['min']:.1f}-{m['max']:.1f})"
            def delta(metric):
                if base is None or label == labels[0] or metric not in row or metric not in base:
                    return ""
                return f"{100 * (row[metric]['mean'] / base[metric]['mean'] - 1):+.1f}%"
            prefill = row.get("prefill_tok_s", {}).get("mean")
            lines.append(f"| {pass_index} | {prompt} | {label} | {cell('decode_tok_s')} | "
                         f"{delta('decode_tok_s')} | {cell('ttft_ms')} | {delta('ttft_ms')} | "
                         f"{'-' if prefill is None else f'{prefill:.0f}'} | "
                         f"{row['transferred_mib']:.0f} | {row['same_answer']} |")
    return "\n".join(lines) + "\n"


def lock_clocks(mhz: int) -> bool:
    if mhz <= 0:
        return False
    result = subprocess.run(["nvidia-smi", "-lgc", f"{mhz},{mhz}"], capture_output=True, text=True)
    print(json.dumps({"lock_clocks_mhz": mhz, "allowed": result.returncode == 0,
                      "message": (result.stdout + result.stderr).strip()[:200]}), flush=True)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    labels = [config["label"] for config in plan["configs"]]
    if len(set(labels)) != len(labels) or not labels:
        raise SystemExit("configuration labels must be present and unique")
    texts = dict(PROMPTS)
    for name, path in plan.get("prompt_files", {}).items():
        texts[name] = Path(path).read_text(encoding="utf-8", errors="ignore")
    missing = [name for name in plan["prompts"] if name not in texts]
    if missing:
        raise SystemExit(f"unknown prompts: {missing}")
    args.out.mkdir(parents=True, exist_ok=True)
    # A harness stop-job ends the tmux session, which sends SIGHUP: without a handler the default
    # action ends this script and leaves the server it started running.
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, handle_signal)
    locked = lock_clocks(int(plan.get("lock_clocks_mhz", 0)))
    records = []
    try:
        configs = plan["configs"]
        for round_index in range(plan.get("rounds", 1)):
            shift = round_index % len(configs)
            for config in configs[shift:] + configs[:shift]:
                records.append(run_server(config, plan, texts, args.out, round_index))
    finally:
        stop_servers()
        if locked:
            subprocess.run(["nvidia-smi", "-rgc"], capture_output=True)
    summary = summarize(plan, records)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    (args.out / "summary.md").write_text(markdown(summary))
    sys.stdout.write(markdown(summary))


if __name__ == "__main__":
    main()
