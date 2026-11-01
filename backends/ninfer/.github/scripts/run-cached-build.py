"""Run a build with bounded cancellation and periodic log progress."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from build_telemetry import sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--cache-stats", type=Path)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--deadline-env", help="Environment variable containing a Unix deadline")
    parser.add_argument("--checkpoint-family", default=os.environ.get("HF_CACHE_FAMILY"))
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--checkpoint-interval", type=float, default=600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    checkpoint = None
    windows_job = None
    child = None
    checkpoint_directory = args.checkpoint_directory or Path(os.environ.get("CCACHE_DIR", ".cache/objects"))
    checkpoint_enabled = bool(args.checkpoint_family and os.environ.get("HF_TOKEN"))
    next_checkpoint = time.monotonic() + args.checkpoint_interval
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a build command is required")
    budgets = []
    if args.timeout_seconds is not None:
        budgets.append(args.timeout_seconds)
    if args.deadline_env:
        budgets.append(float(os.environ[args.deadline_env]) - time.time())
    budget = min(budgets) if budgets else float("inf")
    if budget <= 0:
        print("Build deadline already reached; leave time to save completed cache entries.", flush=True)
        return 124
    deadline = time.monotonic() + budget
    cancelled = False
    timed_out = False

    def cancel(_signum, _frame):
        nonlocal cancelled
        cancelled = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, cancel)
    if os.name == "nt":
        signal.signal(signal.SIGBREAK, cancel)
    if args.pid_file:
        args.pid_file.write_text(str(os.getpid()))
    args.log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.log.open("wb") as log:
            options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                       if os.name == "nt" else {"start_new_session": True})
            if os.name == "nt":
                from windows_build_job import WindowsBuildJob
                windows_job = WindowsBuildJob()
                command = windows_job.command(command)
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **options)
            next_report = 0
            next_sample = 0
            while child.poll() is None and not cancelled:
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                if checkpoint is not None and checkpoint.poll() is not None:
                    if checkpoint.returncode:
                        print(f"::warning::Periodic HF checkpoint failed ({checkpoint.returncode}); the previous snapshot remains available.", flush=True)
                    checkpoint = None
                    next_checkpoint = time.monotonic() + args.checkpoint_interval
                if checkpoint_enabled and checkpoint is None and time.monotonic() >= next_checkpoint:
                    checkpoint = subprocess.Popen([
                        sys.executable, str(Path(__file__).with_name("hf-build-cache.py")),
                        "save", "--directory", str(checkpoint_directory), "--family", args.checkpoint_family], **options)
                if args.cache_stats and time.monotonic() >= next_sample:
                    sample(args.cache_stats)
                    next_sample = time.monotonic() + 60
                if time.monotonic() >= next_report:
                    print(f"Build PID {child.pid}; log bytes: {args.log.stat().st_size}", flush=True)
                    with args.log.open("rb") as stream:
                        stream.seek(max(0, args.log.stat().st_size - 2048))
                        print(stream.read().decode(errors="replace").splitlines()[-2:], flush=True)
                    next_report = time.monotonic() + 15
                time.sleep(0.2)
            if cancelled or timed_out:
                reason = "Build deadline reached" if timed_out else "Build cancelled"
                print(f"{reason}; stopping compiler work before saving completed object-cache entries.", flush=True)
                try:
                    if os.name == "nt":
                        child.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        os.killpg(child.pid, signal.SIGINT)
                    child.wait(timeout=4)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
                finally:
                    # The parent can exit before its compiler children. Retain
                    # ownership of the group/job and stop it unconditionally.
                    if windows_job is not None:
                        windows_job.stop()
                    else:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    child.wait(timeout=2)
                return 124 if timed_out else 130
            return child.returncode
    finally:
        if windows_job is not None:
            windows_job.close()
        # End an unfinished transfer before the final save step takes the writer lock.
        # Its unpublished slot cannot replace the last completed checkpoint.
        if checkpoint is not None and checkpoint.poll() is None:
            checkpoint.terminate()
            try:
                checkpoint.wait(timeout=2)
            except subprocess.TimeoutExpired:
                checkpoint.kill()
                checkpoint.wait(timeout=2)
        if args.pid_file:
            args.pid_file.unlink(missing_ok=True)
        if args.log.exists():
            with args.log.open("rb") as stream:
                stream.seek(max(0, args.log.stat().st_size - 6000))
                print(stream.read().decode(errors="replace"), flush=True)


if __name__ == "__main__":
    sys.exit(main())
