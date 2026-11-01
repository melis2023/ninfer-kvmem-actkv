#!/usr/bin/env python3
"""Delete the five superseded NInfer rentals after checking retained evidence and models."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from huggingface_hub import HfApi
from harness import rental_stopped, vast
from hf_cleanup import RETAINED

ROOT = Path(__file__).resolve().parents[2]
PODS = ROOT / ".local/pods"
TARGETS = {54722983: "pod.json", 54840067: "a6000/pod.json", 54851854: "pipeline/pod.json",
           54876622: "hybrid/pod.json", 54899355: "vnni/pod.json"}
RECEIPT = PODS / "retired-20261008.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    current = {p["id"]: p for p in vast("show", "instances")}
    evidence = PODS / "final-results/jobs"
    for relative in ("draft-confidence/exit", "final-quality/iq4/report.json", "final-quality/fp8/report.json",
                     "final-resident-logits/logits.json", "final-resident-logits/logits.json.bf16",
                     "final-hybrid-logits/logits-dma-1.json", "final-hybrid-logits/logits-dma-1.json.bf16",
                     "final-serving/dma-1-k4-buffered.json", "final-serving-rest/dma-1-k0-buffered.json"):
        if not (evidence / relative).is_file() or (evidence / relative).stat().st_size == 0:
            raise RuntimeError(f"required retained result is absent: {relative}")
    # Models with continuing practical or scientific value must still exist in the private archives.
    api = HfApi()
    models = []
    audit = {r["repo"]: r for r in json.loads((ROOT / ".local/hf-artifact-audit/audit.json").read_text())}
    for repo in RETAINED:
        artifact, = audit[repo]["artifacts"]
        info = api.model_info(repo, files_metadata=True)
        file, = [s for s in info.siblings if s.rfilename == artifact["file"]]
        if (info.private is not True or file.size != artifact["bytes"] or file.lfs is None or
                file.lfs.sha256 != artifact["file_sha256"]):
            raise RuntimeError("a retained model archive no longer matches its receipt")
        models.append({"repo": repo, "revision": info.sha, "file": file.rfilename,
                       "bytes": file.size, "sha256": file.lfs.sha256})
    if len(models) != 4:
        raise RuntimeError("expected the MTP donor, native model and two retained layer slices")
    plan = {"retained_models": models, "local_evidence": str(evidence), "rentals": []}
    for identity, relative in TARGETS.items():
        state_path = PODS / relative
        state = json.loads(state_path.read_text())
        archives = [p for p in state_path.parent.glob("results-*/results.tar.gz") if p.stat().st_size > 0]
        if state["id"] != identity or not archives:
            raise RuntimeError("rental identity or retained result archives are absent")
        live = current.get(identity)
        if live is not None and (live.get("label") != "ninfer-correctness" or not rental_stopped(
                live.get("actual_status"), live.get("intended_status"))):
            raise RuntimeError("a retirement target is active or belongs to another task")
        plan["rentals"].append({"id": identity, "state": relative,
            "local_archives": len(archives), "disk_gb": live.get("disk_space") if live else None,
            "already_absent": live is None,
            "disposition": "retain local results, source snapshots and private model archives; discard remote copies and rebuildable binaries"})
    RECEIPT.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2), flush=True)
    if not args.apply:
        return
    for item in plan["rentals"]:
        identity = item["id"]
        if not item["already_absent"]:
            vast("destroy", "instance", identity, "-y")
        # Removal from the account inventory establishes deletion, not just stopped compute.
        for attempt in range(7):
            if not any(p["id"] == identity for p in vast("show", "instances")):
                break
            time.sleep(5)
        else:
            raise RuntimeError(f"rental deletion is not yet confirmed: {identity}")
        state_path = PODS / item["state"]
        state = json.loads(state_path.read_text())
        state["destroyed"] = time.time()
        state["retirement_receipt"] = str(RECEIPT)
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        if label := state.get("guard_label"):
            subprocess.run(["launchctl", "remove", label], capture_output=True)
        item["deletion_confirmed"] = True
        RECEIPT.write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps({"rental_deleted": identity}), flush=True)


if __name__ == "__main__":
    main()
