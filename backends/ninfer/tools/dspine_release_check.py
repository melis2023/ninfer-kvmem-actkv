#!/usr/bin/env python3
"""Release gate for DSpine: conversion audit plus both smoke checks, per artifact.

The three checks cover what throughput numbers cannot see:

  audit   the artifact's manifest still matches the released GGUF's metadata and
          tensor shapes (a wrong rank or tap list only shows up as lower acceptance)
  cli     greedy output of `--spec dflash` is byte-identical to `--spec none`
  kvmem   a tiny KVMem budget forces paging and the request still completes

`audit` needs numpy (it parses the GGUF), so run this with the same interpreter the
conversion uses. `cli` and `kvmem` start the server, so they need a free GPU.

Usage:
  python dspine_release_check.py --gguf released.gguf \
      --artifact shipping.ninfer [--artifact other.ninfer ...]
  python dspine_release_check.py ... --skip-smoke     # audit only, no GPU
  python dspine_release_check.py ... --checks cli,kvmem
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE / "dspine_audit.py"
SMOKE = HERE / "dspine_smoke.py"


def run(args: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(args, capture_output=True, text=True)
    tail = (proc.stdout or proc.stderr).strip().splitlines()
    return proc.returncode == 0, tail[-1] if tail else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, type=Path)
    ap.add_argument("--artifact", required=True, action="append", type=Path,
                    help="repeat for each artifact to gate")
    ap.add_argument("--checks", default="audit,cli,kvmem")
    ap.add_argument("--skip-smoke", action="store_true", help="alias for --checks audit")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter for the audit, which imports numpy")
    args = ap.parse_args()

    checks = ["audit"] if args.skip_smoke else [c.strip() for c in args.checks.split(",")]
    unknown = set(checks) - {"audit", "cli", "kvmem"}
    if unknown:
        raise SystemExit(f"unknown checks: {sorted(unknown)}")

    failures: list[str] = []
    for artifact in args.artifact:
        print(f"\n=== {artifact.name}")
        if not artifact.exists():
            print("  MISSING")
            failures.append(f"{artifact.name}: missing")
            continue

        if "audit" in checks:
            ok, summary = run([args.python, str(AUDIT), "--gguf", str(args.gguf),
                               "--artifact", str(artifact)])
            print(f"  audit  {'PASS' if ok else 'FAIL'}  {summary}")
            if not ok:
                failures.append(f"{artifact.name}: audit")

        for check in [c for c in checks if c != "audit"]:
            ok, summary = run([sys.executable, str(SMOKE), "--artifact", str(artifact),
                               "--check", check])
            print(f"  {check:<6} {'PASS' if ok else 'FAIL'}  {summary}")
            if not ok:
                failures.append(f"{artifact.name}: {check}")

    print()
    if failures:
        print(f"{len(failures)} failure(s):")
        for item in failures:
            print(f"  {item}")
        return 1
    print(f"all checks passed on {len(args.artifact)} artifact(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
