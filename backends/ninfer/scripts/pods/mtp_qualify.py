#!/usr/bin/env python3
"""Check staged MTP models through serving with GPU expert arithmetic."""
import argparse
import base64
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib

ROOT = Path("/workspace/ninfer-work")
JOB = Path(os.environ["NINFER_JOB_DIR"])
BASE = "http://127.0.0.1:18088"
SERVER = ROOT / "confidence-runtime/apps/ninfer-serve"
TABLE = ROOT / "models/flash-next-iq4-table.ninfer"
PROMPTS = (
    "The library opens at nine. The garden contains apple trees.\n" * 4 +
    "Write a Python function to merge two sorted lists.\n",
    "Explain why the sky looks blue during the day. Use simple language.\n",
)


def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    query = urllib.request.Request(BASE + path, data=data,
                                   headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(query, timeout=300) as response:
        return json.load(response)


def image_fixture():
    """Return a deterministic RGB PNG for the media request check."""
    def chunk(kind, value):
        return (struct.pack(">I", len(value)) + kind + value +
                struct.pack(">I", zlib.crc32(kind + value)))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 56, 56, 8, 2, 0, 0, 0)) +
           chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * 56) * 56)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def save(record):
    (JOB / (record["label"] + ".json")).write_text(json.dumps(record, indent=2) + "\n")


def run(candidate, residency, drafts, floor, media_only=False):
    label = f"{candidate['variant']}-{residency}-k{drafts}-p{floor}" + ('-vision' if media_only else '')
    command = [str(SERVER), candidate["model_path"], "--host", "127.0.0.1", "--port", "18088",
               "--device", "0", "--max-context", "4096", "--kv-capacity", "4096",
               "--kv-dtype", "int8", "--max-concurrency", "1", "--prefill-chunk", "128",
               "--expert-residency", residency, "--expert-cache-mib", "8192",
               "--ngram-table", str(TABLE), "--ngram-io", "buffered",
               "--ngram-ram-mib", "0", "--ngram-draft-tokens", "0",
               "--vision", "--vision-max-merged", "64"]
    if drafts:
        command += ["--spec", "mtp", "--draft-tokens", str(drafts), "--draft-min-p", str(floor)]
    record = {"label": label, "command": command, "candidate": candidate,
              "residency": residency, "draft_tokens": drafts, "draft_min_p": floor,
              "samples": [], "status": "running", "page_cache": "uncontrolled"}
    save(record)
    with (JOB / (label + ".server.log")).open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            started = time.monotonic()
            while time.monotonic() - started < 240:
                if process.poll() is not None:
                    raise RuntimeError(f"{label}: server exited {process.returncode}")
                try:
                    record["health"] = request("/health")
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(1)
            else:
                raise RuntimeError(f"{label}: startup exceeded 240 seconds")
            record["startup_seconds"] = time.monotonic() - started
            record["models"] = request("/v1/models")
            print(json.dumps({"label": label, "stage": "ready"}), flush=True)
            for prompt_id, prompt in enumerate(() if media_only else PROMPTS):
                expected = None
                for repeat in range(3):
                    before = request("/stats")
                    result = request("/completion", {"prompt": prompt, "n_predict": 64,
                        "temperature": 0, "ignore_eos": True, "cache_prompt": False,
                        "return_tokens": True})
                    after = request("/stats")
                    if result["tokens_predicted"] != 64 or result["tokens_cached"] != 0:
                        raise RuntimeError(f"{label}: unexpected output length or prefix reuse")
                    if expected is not None and result["tokens"] != expected:
                        raise RuntimeError(f"{label}: fixed mode changed its output")
                    expected = result["tokens"]
                    if drafts and result["timings"].get("draft_n", 0) <= 0:
                        raise RuntimeError(f"{label}: MTP did not propose tokens")
                    record["samples"].append({"prompt": prompt_id, "repeat": repeat,
                        "result": result, "stats_before": before, "stats_after": after})
                    record["process_status"] = Path(f"/proc/{process.pid}/status").read_text()
                    record["process_io"] = Path(f"/proc/{process.pid}/io").read_text()
                    save(record)
                    print(json.dumps({"label": label, "prompt": prompt_id, "repeat": repeat,
                                      "timings": result["timings"]}), flush=True)
            if drafts and not media_only and not any(s["result"]["timings"].get("draft_n_accepted", 0) > 0
                                  for s in record["samples"]):
                raise RuntimeError(f"{label}: none of the drafts were accepted")
            # Media uses the existing plain fallback. A fresh text request must resume MTP.
            if drafts and floor == 0:
                record["vision"] = request("/v1/chat/completions", {
                    "model": record["models"]["data"][0]["id"], "temperature": 0, "max_tokens": 32,
                    "enable_thinking": False,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "Describe the image color in a full sentence and name the dominant RGB channel."},
                        {"type": "image_url", "image_url": {"url": image_fixture()}}]}]})
                if not record["vision"].get("choices"):
                    raise RuntimeError(f"{label}: media request returned no choices")
                if record['vision'].get('timings', {}).get('draft_n', 0) != 0:
                    raise RuntimeError(f'{label}: media bypassed the documented plain fallback')
                answer = record['vision']['choices'][0]['message'].get('content', '').lower()
                if 'red' not in answer:
                    raise RuntimeError(f'{label}: media answer did not identify the red fixture')
                record['text_after_vision'] = request('/completion', {
                    'prompt': PROMPTS[1], 'n_predict': 32, 'temperature': 0,
                    'ignore_eos': True, 'cache_prompt': False, 'return_tokens': True})
                if record['text_after_vision']['timings'].get('draft_n', 0) <= 0:
                    raise RuntimeError(f'{label}: the next text request did not resume MTP')
            record["status"] = "passed"
        except Exception as error:
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            save(record)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument('--media-only', action='store_true')
    args = parser.parse_args()
    variant = json.loads(args.config.read_text())["variant"]
    if variant not in ("q2", "iq3", "coder"):
        raise ValueError("unknown MTP candidate")
    candidate = json.loads((ROOT / "mtp-candidates" / variant / "download-verified.json").read_text())
    if candidate.get("artifact_sha256_verified") is not True:
        raise ValueError("candidate has not passed its SHA-256 check")
    records = []
    modes = [("host", 4, 0)] if args.media_only else [("host", 4, 0), ("disk", 4, 0)]
    if variant == "q2" and not args.media_only:
        modes += [("host", 0, 0), ("host", 4, 0.3)]
    for mode in modes:
        records.append(run(candidate, *mode, media_only=args.media_only))
    summary = {"variant": variant, "artifact_id": candidate["artifact_id"],
               "sha256": candidate["sha256"], "status": "passed",
               "source": json.loads((ROOT / "source.json").read_text()),
               "hardware": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version",
                                                     "--format=csv,noheader"], text=True),
               "checks": [r["label"] for r in records],
               "limitations": ["No physical low-RAM host or RTX 3080 20 GB qualification",
                                "OS page cache is uncontrolled; disk results are not cold-I/O measurements",
                                "Output equality is required only within each fixed execution mode",
                                "Media requests use plain decoding; fresh text requests resume MTP"]}
    name = 'vision-qualification.json' if args.media_only else 'qualification.json'
    (JOB / name).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"stage": "MTP_VISION_PASS" if args.media_only else "MTP_QUALIFICATION_PASS",
                      "variant": variant}), flush=True)


if __name__ == "__main__":
    main()
