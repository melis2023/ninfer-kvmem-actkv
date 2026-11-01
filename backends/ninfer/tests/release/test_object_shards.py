"""Verify that object sharding covers the release graph exactly once."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "object_shards", Path(__file__).resolve().parents[2] / ".github/scripts/build-object-shard.py")
SHARDS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHARDS)


class ObjectShardsTest(unittest.TestCase):
    def test_complete_disjoint_partition_independent_of_order_and_separator(self):
        objects = [f"src/ops/CMakeFiles/ops.dir/kernel{i}.cu.obj" for i in range(600)]
        commands = [{"output": path} for path in objects] + [{"output": "apps/ninfer.exe"}]
        parts = [SHARDS.select_objects(commands, index, 8) for index in range(8)]
        self.assertEqual(sorted(output for part in parts for output in part), sorted(objects))
        self.assertTrue(all(parts))
        changed = [{"output": item["output"].replace("/", "\\")} for item in reversed(commands)]
        for index, part in enumerate(parts):
            self.assertEqual(SHARDS.select_objects(changed, index, 8), part)

    def test_duplicate_rules_do_not_compile_twice_and_invalid_shards_fail(self):
        self.assertEqual(SHARDS.select_objects([{"output": "x.o"}] * 2, 0, 1), ["x.o"])
        for index, count in ((-1, 8), (8, 8), (0, 0)):
            with self.assertRaises(ValueError):
                SHARDS.select_objects([], index, count)
