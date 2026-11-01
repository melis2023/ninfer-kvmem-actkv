"""The CTest launcher must skip GPU work before importing conversion dependencies."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("probe_result", [0, 77, 1])
def test_launcher_probes_before_running_conversion(tmp_path, probe_result):
    root = tmp_path / "tests"
    launcher = root / "models/qwen4_exp/ngram_writer_interop.py"
    launcher.parent.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "models/qwen4_exp/ngram_writer_interop.py"
    launcher.write_bytes(source.read_bytes())
    writer = root / "convert/test_qwen4_exp_ngram.py"
    writer.parent.mkdir()
    # The converter fails if invoked: unavailable CUDA must bypass its imports entirely.
    writer.write_text("raise SystemExit(23)\n")
    binary = tmp_path / "probe"
    binary.write_text(f"#!{sys.executable}\nimport sys\n"
                      "assert sys.argv[1:] == ['ninfer_qwen4_exp_ngram_component_test', '--check-cuda']\n"
                      f"raise SystemExit({probe_result})\n")
    binary.chmod(0o755)
    # -S removes site packages, matching the toolchain without pytest or CPU PyTorch.
    result = subprocess.run([sys.executable, "-S", str(launcher), str(binary)])
    assert result.returncode == (23 if probe_result == 0 else probe_result)
