#!/usr/bin/env python3
"""Manage this task's disposable CPU Pod; remote work uses the existing tmux harness."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

STATE = Path(__file__).resolve().parents[2] / ".local/pods/runpod-cpu"
NAME = "ninfer-mtp-01a11831-cpu"
HELPER = Path.home() / "runpod/pod-lib.sh"


def api(method, path, body=None):
    command = 'source "$1"; rest_api "$2" "$3" "${4:-}"'
    result = subprocess.run(["bash", "-c", command, "runpod-cpu", str(HELPER),
                             method, path, json.dumps(body) if body else ""],
                            capture_output=True, text=True, check=True)
    value = json.loads(result.stdout) if result.stdout.strip() else None
    if isinstance(value, dict) and (value.get("error") or value.get("errors")):
        # Provider errors can contain credentials or environment values.
        raise RuntimeError(f"RunPod {method} {path} rejected the request")
    return value


def save(value):
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "rental.json").write_text(json.dumps(value, indent=2) + "\n")


def refresh(rental):
    pod = api("GET", f"/pods/{rental['id']}")
    if pod.get("id") != rental["id"] or pod.get("name") != NAME:
        raise RuntimeError("rental identity does not match")
    for key in ("desiredStatus", "costPerHr", "publicIp", "portMappings", "vcpuCount",
                "memoryInGb", "containerDiskInGb", "networkVolumeId"):
        rental[key] = pod.get(key)
    rental["start_command_present"] = bool(pod.get("dockerStartCmd"))
    save(rental)
    if pod.get("publicIp") and (pod.get("portMappings") or {}).get("22"):
        state_path = STATE / "pod.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        state.update(provider="ssh", host=pod["publicIp"], port=pod["portMappings"]["22"],
                     runpod_id=rental["id"])
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    return rental


def arm_guard(rental):
    command = [sys.executable, str(Path(__file__).resolve()), "guard"]
    log = STATE / "guard.log"
    if sys.platform == "darwin":
        label = f"ninfer-runpod-{rental['id']}"
        subprocess.run(["launchctl", "remove", label], capture_output=True)
        subprocess.run(["launchctl", "submit", "-l", label, "-o", str(log), "-e", str(log),
                        "--", "/usr/bin/env", f"PATH={os.environ['PATH']}",
                        "/usr/bin/caffeinate", "-i", "-s", *command], check=True)
        rental["guard_label"] = label
    else:
        with log.open("a") as output:
            guard = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                     stderr=output, start_new_session=True)
        rental["guard_pid"] = guard.pid
    save(rental)


def create(dc):
    pods = api("GET", "/pods")
    if not isinstance(pods, list) or any(p.get("name") == NAME for p in pods):
        raise RuntimeError("inspect the existing task rental before creating another")
    if (STATE / "rental.json").exists():
        raise RuntimeError("a retained rental manifest already exists")
    volume = api("POST", "/networkvolumes", {"name": NAME, "size": 160, "dataCenterId": dc})
    if not volume or not volume.get("id"):
        raise RuntimeError("network volume creation was not confirmed")
    rental = {"name": NAME, "volume_id": volume["id"], "dc": dc,
              "created": time.time(), "deadline": time.time() + 7200}
    save(rental)
    pubkey = (Path.home() / ".ssh/id_ed25519.pub").read_text().strip()
    command = ("mkdir -p /root/.ssh; printf '%s\\n' " + shlex.quote(pubkey) +
               " >> /root/.ssh/authorized_keys; chmod 700 /root/.ssh; "
               "chmod 600 /root/.ssh/authorized_keys; exec /start.sh")
    pod = api("POST", "/pods", {
        "computeType": "CPU", "cpuFlavorIds": ["cpu3c"], "vcpuCount": 4,
        "cloudType": "SECURE", "dataCenterIds": [dc], "name": NAME,
        "imageName": "runpod/base:1.0.2-ubuntu2404", "supportPublicIp": True,
        "ports": ["22/tcp"], "containerDiskInGb": 20, "volumeInGb": 0,
        "networkVolumeId": volume["id"], "volumeMountPath": "/workspace",
        "dockerStartCmd": ["bash", "-c", command],
        "env": {"PUBLIC_KEY": pubkey, "SSH_PUBLIC_KEY": pubkey}})
    if not pod or not pod.get("id"):
        raise RuntimeError("Pod creation not confirmed; inspect retained volume manifest")
    rental["id"] = pod["id"]
    save(rental)
    arm_guard(rental)
    return refresh(rental)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "status", "stop", "destroy", "guard", "arm"))
    parser.add_argument("--dc", default="EU-RO-1")
    args = parser.parse_args()
    if args.action == "create":
        print(json.dumps(create(args.dc), indent=2))
        return
    rental = json.loads((STATE / "rental.json").read_text())
    if args.action == "arm":
        arm_guard(rental)
        print(json.dumps({"guard": rental.get("guard_label", rental.get("guard_pid")),
                          "deadline": rental["deadline"]}))
    elif args.action == "guard":
        print(json.dumps({"guard_started": rental["id"], "deadline": rental["deadline"]}), flush=True)
        while time.time() < rental["deadline"]:
            time.sleep(min(30, rental["deadline"] - time.time()))
            current = json.loads((STATE / "rental.json").read_text())
            if current.get("destroyed") or current.get("id") != rental.get("id"):
                return
        api("POST", f"/pods/{rental['id']}/stop")
        print(json.dumps(refresh(rental)), flush=True)
    elif args.action == "status":
        print(json.dumps(refresh(rental), indent=2))
    elif args.action == "stop":
        api("POST", f"/pods/{rental['id']}/stop")
        print(json.dumps(refresh(rental), indent=2))
    else:
        # The marker is written only after Hub upload verification and local receipt collection.
        if not (STATE / "cleanup-approved.json").exists():
            raise RuntimeError("collect receipts and verify archived artifacts before cleanup")
        refresh(rental)
        api("DELETE", f"/pods/{rental['id']}")
        if any(p.get("id") == rental["id"] for p in api("GET", "/pods")):
            raise RuntimeError("Pod deletion not yet confirmed")
        api("DELETE", f"/networkvolumes/{rental['volume_id']}")
        if any(v.get("id") == rental["volume_id"] for v in api("GET", "/networkvolumes")):
            raise RuntimeError("volume deletion not yet confirmed")
        rental["destroyed"] = time.time()
        save(rental)
        print(json.dumps({"pod_deleted": rental["id"], "volume_deleted": rental["volume_id"]}))


if __name__ == "__main__":
    main()
