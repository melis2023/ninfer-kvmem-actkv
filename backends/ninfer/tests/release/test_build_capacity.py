"""Protect rented build workers from host-wide CPU counts and container RAM limits."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('capacity', Path(__file__).resolve().parents[2] /
                                             '.github/scripts/check-build-capacity.py')
CAPACITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAPACITY)


class CapacityTest(unittest.TestCase):
    def test_quotas_and_memory_bound_parallelism(self):
        self.assertEqual(CAPACITY.budget(64, 384, [30.72], 128 * CAPACITY.GIB), 31)
        self.assertEqual(CAPACITY.budget(64, 128, [92.16], 64 * CAPACITY.GIB), 28)
        self.assertEqual(CAPACITY.budget(4, 4, [], 16 * CAPACITY.GIB), 4)
        self.assertEqual(CAPACITY.budget(32, 16, [], 256 * CAPACITY.GIB), 16)

    def test_v1_and_v2_limits(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(
                CAPACITY.os, 'sysconf', side_effect=lambda name: 4096 if name == 'SC_PAGE_SIZE' else 67108864):
            root = Path(temporary)
            (root / 'cpu.max').write_text('3200000 100000')
            (root / 'memory.max').write_text(str(64 * CAPACITY.GIB))
            self.assertEqual(CAPACITY.allocation(root), ([32.0], 64 * CAPACITY.GIB))
            (root / 'cpu.max').unlink()
            (root / 'memory.max').unlink()
            (root / 'cpu,cpuacct').mkdir()
            (root / 'cpu,cpuacct/cpu.cfs_quota_us').write_text('3072000')
            (root / 'cpu,cpuacct/cpu.cfs_period_us').write_text('100000')
            (root / 'memory').mkdir()
            (root / 'memory/memory.limit_in_bytes').write_text(str(32 * CAPACITY.GIB))
            self.assertEqual(CAPACITY.allocation(root), ([30.72], 32 * CAPACITY.GIB))
