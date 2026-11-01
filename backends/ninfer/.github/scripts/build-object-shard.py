"""Compile one stable subset of the release objects, preserving their full CUDA target set."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def select_objects(commands, index, count):
    if count < 1 or not 0 <= index < count:
        raise ValueError("Invalid object shard")
    outputs = set()
    for item in commands:
        output = item["output"].replace("\\", "/")
        if not output.endswith((".o", ".obj")):
            continue
        # Object paths are relative to the build tree. Host checkout paths and
        # compilation database order must not change shard ownership.
        owner = int.from_bytes(hashlib.sha256(output.encode()).digest()[:8], "big") % count
        if owner == index:
            outputs.add(output)
    return sorted(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=Path("build-release"))
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    commands = json.loads(subprocess.check_output(
        ["ninja", "-C", str(args.build), "-t", "compdb-targets",
         "ninfer", "ninfer-serve", "ninfer_bench"], text=True))
    objects = select_objects(commands, args.shard, args.count)
    if not objects:
        raise ValueError("The release object shard is empty")
    (args.build / "object-shard.json").write_text(json.dumps(objects, indent=2) + "\n")
    print(f"Compiling {len(objects)} release objects in shard {args.shard + 1}/{args.count}", flush=True)
    # Keep commands below Windows' command-line limit. Ninja still builds each
    # object's generated dependencies through the original project graph.
    batches = [[]]
    length = 0
    for output in objects:
        if length + len(output) + 3 > 24000:
            batches.append([])
            length = 0
        batches[-1].append(output)
        length += len(output) + 3
    for batch in batches:
        subprocess.run(["cmake", "--build", str(args.build), "-j", str(args.jobs),
                        "--target", *batch], check=True)


if __name__ == "__main__":
    main()
