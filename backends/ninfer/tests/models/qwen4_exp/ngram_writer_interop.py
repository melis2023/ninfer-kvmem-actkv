"""Check CUDA availability before loading the n-gram conversion test dependencies."""

from pathlib import Path
import subprocess
import sys


def main() -> int:
    binary = sys.argv[1]
    probe = subprocess.run([binary, "ninfer_qwen4_exp_ngram_component_test", "--check-cuda"])
    if probe.returncode:
        return probe.returncode
    writer = Path(__file__).resolve().parents[2] / "convert/test_qwen4_exp_ngram.py"
    return subprocess.run([sys.executable, "-B", str(writer), binary]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
