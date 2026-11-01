#!/usr/bin/env python3
"""Diagnose pipeline differences with the A6000's route choices on both RTX 3090s."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("/workspace/ninfer-work")
job = Path(os.environ["NINFER_JOB_DIR"])
source = json.loads((job / "inputs/device-profiles.json").read_text())
builtin = json.loads((ROOT / "src/src/runtime/engine/device_profiles.json").read_text())
entry, = [d for d in source["devices"] if d["hardware_class"] == "nvidia-rtx-a6000-sm86"]
target, = [d for d in builtin["devices"] if d["hardware_class"] == "nvidia-geforce-rtx-3090-sm86"]
forced = copy.deepcopy(entry)
forced["hardware_class"] = target["hardware_class"]
forced["multiprocessors"] = target["multiprocessors"]
forced["origin"] = "diagnostic A6000 routes on RTX 3090; not calibrated for this device"
maximum = 2147483647
for name in set(target["routes"]) | set(forced["routes"]):
    bands = forced["routes"].setdefault(name, [])
    # A6000 has no built-in profile. Empty schedules reproduce its compiled Op defaults
    # beyond measured bands instead of inheriting the RTX 3090's tuned profile there.
    if not bands or bands[-1][0] < maximum:
        bands.append([maximum, ""])
profile = job / "forced-profiles.json"
profile.write_text(json.dumps({**source, "devices": [forced]}, indent=2) + "\n")
metadata = {"purpose": "route-choice diagnostic, not RTX 3090 calibration",
            "source_hardware_class": entry["hardware_class"],
            "target_hardware_class": target["hardware_class"],
            "source_route_count": len(entry["routes"]),
            "target_route_count": len(forced["routes"])}
(job / "profile-probe.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata), flush=True)
completed = subprocess.run([sys.executable, "-m", "scripts.pods.pipeline_run", "--devices", "0,1"],
                           env={**os.environ, "NINFER_DEVICE_PROFILES": str(profile)})
raise SystemExit(completed.returncode)
