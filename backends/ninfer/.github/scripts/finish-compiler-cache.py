"""Stop compiler work and prepare its bounded object cache for upload."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from build_telemetry import cache_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--stats", type=Path, required=True)
    args = parser.parse_args()
    if args.pid_file and args.pid_file.exists():
        try:
            os.kill(int(args.pid_file.read_text()), signal.SIGINT)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 6
        while args.pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.2)
    # Each successful compiler invocation has already committed its cache entry atomically.
    subprocess.run(["ccache", "--cleanup"], check=True, timeout=60)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(cache_stats(), indent=2) + "\n")
    subprocess.run(["ccache", "--show-stats"], check=True, timeout=20)


if __name__ == "__main__":
    main()
