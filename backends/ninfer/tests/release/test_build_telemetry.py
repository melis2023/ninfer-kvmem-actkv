"""Check observable object-cache and resource counters."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("build_telemetry", ROOT / ".github/scripts/build_telemetry.py")
TELEMETRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TELEMETRY)


class BuildTelemetryTest(unittest.TestCase):
    def test_cache_failures_remain_visible_in_progress(self):
        stats = dict(direct_cache_hit=42, cache_miss=3, local_storage_write=19,
                     internal_error=2, compile_failed=0, cache_size_kibibyte=1024)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(TELEMETRY.subprocess, "run", return_value=subprocess.CompletedProcess(
                    [], 0, stdout="".join(f"{key}\t{value}\n" for key, value in stats.items()))), \
                patch("builtins.print"):
            path = Path(temp) / "progress.jsonl"
            TELEMETRY.sample(path)
            report = json.loads(path.read_text())
            self.assertEqual(report["internal_error"], 2)
            self.assertEqual(report["direct_cache_hit"], 42)
            self.assertEqual(report["cache_size_kibibyte"], 1024)
            self.assertGreater(report["free_disk_bytes"], 0)

    def test_stats_timeout_is_reported_without_stopping_compilation(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(TELEMETRY.subprocess, "run", side_effect=subprocess.TimeoutExpired("ccache", 5)), \
                patch("builtins.print"):
            path = Path(temp) / "progress.jsonl"
            TELEMETRY.sample(path)
            self.assertEqual(json.loads(path.read_text())["cache_statistics_unavailable"], "TimeoutExpired")
