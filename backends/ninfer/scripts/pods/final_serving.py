#!/usr/bin/env python3
"""Measure fixed hybrid configurations through the public completion and stats endpoints."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path("/workspace/ninfer-work")
JOB = Path(os.environ["NINFER_JOB_DIR"])
BASE = "http://127.0.0.1:18088"
GGUF_BASELINE = sys.argv[1:] == ["--gguf-baseline"]
PLAIN_ONLY = sys.argv[1:] == ["--plain-only"]
IO_ONLY = sys.argv[1:] == ["--io-only"]
CONFIDENCE_ONLY = sys.argv[1:] == ["--confidence-only"]
SERVER = Path(os.environ.get("NINFER_SERVE_BINARY", str(ROOT / "build/apps/ninfer-serve")))
if sys.argv[1:] and not (GGUF_BASELINE or PLAIN_ONLY or IO_ONLY or CONFIDENCE_ONLY):
    raise SystemExit("usage: final_serving.py [--gguf-baseline|--plain-only|--io-only|--confidence-only]")


def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    query = urllib.request.Request(BASE + path, data=data,
                                   headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(query, timeout=240) as response:
        return json.load(response)


def run(share, drafts, io, floor=None):
    label = f"dma-{share}-k{drafts}-{io}"
    if floor is not None:
        label += f"-min-p-{floor}"
    command = [str(SERVER),
               str(ROOT / "models" / ("flash-next-q2_0-mtp.ninfer" if GGUF_BASELINE else
                                       "flash-next-native-q2-mtp.ninfer")),
               "--host", "127.0.0.1", "--port", "18088", "--device", "0",
               "--max-context", "4096", "--kv-capacity", "4096", "--kv-dtype", "int8",
               "--max-concurrency", "1", "--prefill-chunk", "512",
               "--expert-residency", "host", "--expert-cache-mib", "4096",
               "--expert-dma-share", str(share), "--expert-cpu-threads", "8",
               "--ngram-table", str(ROOT / "models/flash-next-iq4-table.ninfer"),
               "--ngram-io", io, "--ngram-ram-mib", "0", "--ngram-draft-tokens", "0"]
    if drafts:
        command += ["--spec", "mtp", "--draft-tokens", str(drafts)]
    if floor is not None:
        command += ["--draft-min-p", str(floor)]
    record = {"command": command, "label": label, "samples": []}
    with (JOB / f"{label}.server.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError(f"{label}: server exited {process.returncode}")
                try:
                    request("/health")
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(1)
            else:
                raise RuntimeError(f"{label}: server readiness timed out")
            for repeat in range(3):
                # Only the selected table's pages are evicted; no global cache changes.
                if repeat == 0:
                    pages = subprocess.check_output([
                        str(ROOT / "py311/bin/python"), "scripts/pods/ngram_cache.py",
                        str(ROOT / "models/flash-next-iq4-table.ninfer"), "--discard"], text=True)
                    record["cold_pages"] = [json.loads(line) for line in pages.splitlines()]
                before = request("/stats")
                result = request("/completion", {
                    "prompt": "The library opens at nine. The garden contains apple trees.\n" * 16 +
                              "Write a Python function to merge two sorted lists.\n",
                    "n_predict": 64, "temperature": 0, "ignore_eos": True,
                    "cache_prompt": False, "return_tokens": True})
                after = request("/stats")
                if IO_ONLY and after.get("ngram_table", {}).get("rows", 0) <= 0:
                    raise RuntimeError(f"{label}: stats omitted completed n-gram reads")
                if result["tokens_predicted"] != 64 or result["tokens_cached"] != 0:
                    raise RuntimeError(f"{label}: unexpected output length or prefix reuse")
                if repeat and result["tokens"] != record["samples"][0]["result"]["tokens"]:
                    raise RuntimeError(f"{label}: fixed configuration changed its answer")
                record["samples"].append({"repeat": repeat, "result": result,
                                          "stats_before": before, "stats_after": after})
                record["process_status"] = Path(f"/proc/{process.pid}/status").read_text()
                (JOB / f"{label}.json").write_text(json.dumps(record, indent=2) + "\n")
                print(json.dumps({"label": label, "repeat": repeat, "timings": result["timings"]}),
                      flush=True)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


configs = ((1, 4, "buffered"),) if GGUF_BASELINE else ((1, 0, "buffered"),) if PLAIN_ONLY else (
    (0, 4, "buffered"), (0.5, 4, "buffered"), (1, 4, "buffered"),
    (1, 4, "direct"), (1, 0, "buffered"))
if IO_ONLY:
    configs = ((1, 4, "buffered"), (1, 4, "direct"))
if CONFIDENCE_ONLY:
    configs = ((1, 4, "buffered", 0), (1, 4, "buffered", 0.3), (1, 4, "buffered", 1))
for config in configs:
    run(*config)
print("FINAL_SERVING_PASS", flush=True)
