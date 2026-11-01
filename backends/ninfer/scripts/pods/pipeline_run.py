#!/usr/bin/env python3
"""Run the public Engine pipeline fixture with explicit devices and retain its inputs/results."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from tools.artifact.reader import Artifact

ROOT = Path("/workspace/ninfer-work")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True, choices=("0", "0,1"))
    args = parser.parse_args()
    output = Path(os.environ["NINFER_JOB_DIR"]) / "pipeline.json"
    artifact = ROOT / "models/flash-next-native-q2-mtp.ninfer"
    table = ROOT / "models/flash-next-iq4-table.ninfer"
    with Artifact(artifact) as reader:
        artifact_id = reader.artifact_id.hex()
    environment = {**os.environ, "NINFER_FLASH_NEXT_ARTIFACT": str(artifact),
                   "NINFER_FLASH_NEXT_NGRAM_TABLE": str(table),
                   "NINFER_FLASH_NEXT_DEVICES": args.devices,
                   "NINFER_FLASH_NEXT_DRAFTS": "0",
                   "NINFER_FLASH_NEXT_PIPELINE_REPORT": str(output)}
    completed = subprocess.run([str(ROOT / "build/tests/ninfer_tests"),
                                "ninfer_qwen4_exp_long_context_real"], env=environment)
    if output.exists():
        report = json.loads(output.read_text())
        report["artifact_id"] = artifact_id
        report["source"] = json.loads((ROOT / "source.json").read_text())
        report["hardware"] = subprocess.run([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total,pci.bus_id",
            "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.splitlines()
        report["exit_code"] = completed.returncode
        output.write_text(json.dumps(report, indent=2) + "\n")
    if completed.returncode:
        raise SystemExit(completed.returncode)
    if not output.exists():
        raise RuntimeError("test binary did not produce the pipeline report")
    print("PIPELINE_REPORT_READY " + str(output), flush=True)


if __name__ == "__main__":
    main()
