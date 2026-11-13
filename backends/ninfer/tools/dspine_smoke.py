#!/usr/bin/env python3
"""DSpine regression smoke test (stdlib only, Windows friendly).

Two independent checks:

  cli     Greedy output of `--spec dflash` must be byte-for-byte identical to
          `--spec none`, and the drafter must accept a nonzero share of tokens.
          This catches the bridge/layout bugs that silently produce a broken
          draft while the verifier still returns the target's tokens.

  kvmem   Start ninfer-serve with a deliberately tiny KVMem budget, send a
          prompt long enough to force page pressure/reselection, and require
          a successful answer, no worker crash, and a nonzero acceptance rate.
          This catches the mandatory-page budget overflow and the
          compact/logical ring-coordinate mismatch.

Usage:
  python dspine_smoke.py --artifact <path/to/model.ninfer> --check cli
  python dspine_smoke.py --artifact <path/to/model.ninfer> --check kvmem
  python dspine_smoke.py --artifact <path/to/model.ninfer> --check all
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_BUILD = REPO / "build-backends" / "ninfer"
LONG_PARAGRAPH = (
    "In a quiet workshop beside the river, the clockmaker kept a shelf of small brass gears, "
    "each labelled with the year it was cut. Visitors often asked which one mattered most, and "
    "he would answer that a machine is only as honest as its least careful part. "
)


def log(msg: str) -> None:
    print(msg, flush=True)


def run_cli(exe: Path, artifact: str, spec: str, max_new: int, draft_tokens: int,
            extra: list[str]) -> tuple[int, bytes, str]:
    args = [str(exe), artifact, "--prompt", "Say OK.", "--max-new", str(max_new),
            "--max-context", "2048", "--kv-capacity", "2048",
            "--ngram-draft-tokens", "0", "--greedy", "--raw-output"]
    if spec == "none":
        args += ["--spec", "none"]
    else:
        args += ["--spec", spec, "--draft-tokens", str(draft_tokens), "--full-draft-head"]
    args += extra
    done = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          cwd=str(DEFAULT_BUILD))
    return done.returncode, done.stdout, done.stderr.decode("utf-8", "replace")


def acceptance_percent(text: str) -> float | None:
    m = re.search(r"dflash acceptance rate\s+([0-9.]+)%", text)
    return float(m.group(1)) if m else None


def check_cli(args: argparse.Namespace) -> bool:
    exe = args.build_dir / "apps" / "ninfer.exe"
    if not exe.is_file():
        log(f"FAIL cli: missing {exe}")
        return False
    ok = True
    base_code, base_out, base_err = run_cli(exe, args.artifact, "none", args.max_new,
                                            args.draft_tokens, args.extra)
    ds_code, ds_out, ds_err = run_cli(exe, args.artifact, "dflash", args.max_new,
                                      args.draft_tokens, args.extra)
    if base_code or ds_code:
        log(f"FAIL cli: exit codes none={base_code} dflash={ds_code}")
        log(base_err[-800:]); log(ds_err[-800:])
        return False
    if base_out != ds_out:
        log("FAIL cli: greedy output differs from --spec none baseline")
        log(f"  none  : {base_out[:200]!r}")
        log(f"  dflash: {ds_out[:200]!r}")
        ok = False
    else:
        log(f"PASS cli: output identical ({len(base_out)} bytes)")
    rate = acceptance_percent(ds_err)
    if rate is None:
        log("FAIL cli: no dflash acceptance statistics on stderr")
        ok = False
    elif rate <= 0:
        log(f"FAIL cli: acceptance rate is {rate}%")
        ok = False
    else:
        log(f"PASS cli: acceptance rate {rate}%")
    return ok


def resolve_model(port: int, requested: str | None) -> str:
    """Each artifact advertises its own id, and answering with the wrong one is a 404,
    so read it back unless the caller pinned one."""
    if requested:
        return requested
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as r:
            listing = json.loads(r.read().decode("utf-8"))
        first = (listing.get("data") or [{}])[0]
        return first.get("id") or "default"
    except Exception:  # noqa: BLE001 - a missing listing falls back to the server default
        return "default"


def check_kvmem(args: argparse.Namespace) -> bool:
    exe = args.build_dir / "apps" / "ninfer-serve.exe"
    if not exe.is_file():
        log(f"FAIL kvmem: missing {exe}")
        return False
    port = args.port
    launch = [str(exe), args.artifact, "--host", "127.0.0.1", "--port", str(port),
              "--kvmem", "--kvmem-budget", str(args.kvmem_budget),
              "--kvmem-gen-reserve", str(args.kvmem_reserve),
              "--kvmem-host-mib", "512", "--kvmem-verify-transfers",
              "--max-context", "2048", "--kv-capacity", str(args.kv_capacity),
              "--prefill-chunk", "256", "--spec", "dflash",
              "--draft-tokens", str(args.draft_tokens), "--full-draft-head"]
    launch += args.extra
    fd, log_path = tempfile.mkstemp(prefix="dspine_kvmem_", suffix=".log")
    os.close(fd)
    log(f"kvmem server log: {log_path}")
    with open(log_path, "wb") as logfile:
        proc = subprocess.Popen(launch, stdout=subprocess.DEVNULL, stderr=logfile,
                                cwd=str(args.build_dir))
        ready = False
        deadline = time.time() + 90
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                if b"listening on" in Path(log_path).read_bytes():
                    ready = True
                    break
            except OSError:
                pass
            time.sleep(0.4)
        result: dict = {}
        if ready:
            prompt = LONG_PARAGRAPH * args.repeat + "\n\nNow ignore all of the above and answer with exactly: OK."
            body = {"model": resolve_model(port, args.model),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": args.max_new, "temperature": 0, "stream": False}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                         data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=240) as response:
                    out = json.loads(response.read().decode("utf-8"))
                choice = (out.get("choices") or [{}])[0]
                result = {"ok": True, "content": choice.get("message", {}).get("content"),
                          "usage": out.get("usage")}
            except Exception as exc:  # noqa: BLE001 - report the HTTP failure verbatim
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait(timeout=10)
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    ok = True
    if not ready:
        log("FAIL kvmem: server never became ready")
        ok = False
    if not result.get("ok"):
        log(f"FAIL kvmem: request failed: {result}")
        ok = False
    else:
        log(f"PASS kvmem: request ok, content={result.get('content')!r}, usage={result.get('usage')}")
    for fatal in ("worker crash", "FATAL engine", "mandatory pressure content exceeds the budget",
                  "DFlash batch returned invalid row metadata"):
        if fatal in text:
            log(f"FAIL kvmem: server log contains {fatal!r}")
            ok = False
    if "KVMEM_PRESSURE" not in text:
        log("WARN kvmem: no pressure event was triggered; increase --repeat")
    rate = None
    m = re.search(r"dflash accepted \d+/\d+ \(([0-9.]+)%\)", text)
    if m:
        rate = float(m.group(1))
    if rate is None:
        log("FAIL kvmem: no dflash acceptance statistics")
        ok = False
    elif rate <= 0:
        log(f"FAIL kvmem: acceptance rate is {rate}%")
        ok = False
    else:
        log(f"PASS kvmem: acceptance rate {rate}%")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", default=os.environ.get("NINFER_DSPINE_ARTIFACT"),
                        help="path to the .ninfer artifact containing the dflash/DSpine component")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--check", choices=("cli", "kvmem", "all"), default="all")
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--draft-tokens", type=int, default=15)
    parser.add_argument("--kvmem-budget", type=int, default=512)
    parser.add_argument("--kvmem-reserve", type=int, default=256)
    parser.add_argument("--kv-capacity", type=int, default=768)
    parser.add_argument("--repeat", type=int, default=26,
                        help="long-prompt paragraph count (26 ~= 1402 tokens)")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--model", default=None,
                        help="model id for the OpenAI-compatible request; "
                             "default: read the served id from /v1/models")
    args, extra = parser.parse_known_args()
    args.extra = extra  # any additional engine flags, e.g. --no-cuda-graph
    if not args.artifact:
        parser.error("--artifact (or NINFER_DSPINE_ARTIFACT) is required")
    results = []
    if args.check in ("cli", "all"):
        results.append(("cli", check_cli(args)))
    if args.check in ("kvmem", "all"):
        results.append(("kvmem", check_kvmem(args)))
    for name, passed in results:
        log(f"{name}: {'PASS' if passed else 'FAIL'}")
    return 0 if all(passed for _, passed in results) else 1


if __name__ == "__main__":
    sys.exit(main())
