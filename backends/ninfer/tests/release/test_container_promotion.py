"""Exercise registry publication order and superseded-source guards with CLI fixtures."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 40


@unittest.skipUnless(shutil.which("bash"), "Requires Bash")
class ContainerPromotionTest(unittest.TestCase):
    def run_promotion(self, heads, missing=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "heads").write_text(json.dumps(heads))
            fixture = """#!/usr/bin/env python3.11
import json, os, sys
from pathlib import Path
root = Path(os.environ['FIXTURE'])
with (root / 'calls').open('a') as output:
    output.write(json.dumps([Path(sys.argv[0]).name, *sys.argv[1:]]) + '\\n')
if Path(sys.argv[0]).name == 'gh':
    heads = json.loads((root / 'heads').read_text())
    print(heads.pop(0))
    (root / 'heads').write_text(json.dumps(heads))
elif 'inspect' in sys.argv and os.environ.get('MISSING') in sys.argv[-1]:
    sys.exit(1)
"""
            for command in ("gh", "docker"):
                path = root / command
                path.write_text(fixture)
                path.chmod(0o755)
            bash = shutil.which("bash")
            result = subprocess.run([bash, str(ROOT / ".github/scripts/promote-container-images.sh")],
                                    env={**os.environ, "PATH": f"{root}:{os.environ['PATH']}",
                                         "FIXTURE": temp, "MISSING": missing or "absent-fixture",
                                         "GITHUB_REPOSITORY": "Owner/Repo", "SHA": SHA},
                                    text=True, capture_output=True)
            calls = [json.loads(line) for line in (root / "calls").read_text().splitlines()]
            return result, calls

    def test_both_images_checked_before_promotion(self):
        result, calls = self.run_promotion([SHA, SHA])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(call[3] == "inspect" for call in calls[:2]))
        creates = [call for call in calls if "create" in call]
        self.assertEqual([call[-2] for call in creates],
                         ["ghcr.io/owner/repo:cuda12.9", "ghcr.io/owner/repo:cuda13.4"])
        self.assertTrue(all(call[-1].endswith(SHA[:12]) for call in creates))

    def test_missing_second_image_prevents_all_promotion(self):
        result, calls = self.run_promotion([], missing="cuda13.4")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any("create" in call for call in calls))

    def test_superseded_source_does_not_move_tags(self):
        result, calls = self.run_promotion(["b" * 40])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any("create" in call for call in calls))

    def test_new_master_between_tags_stops_promotion(self):
        result, calls = self.run_promotion([SHA, "b" * 40])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sum("create" in call for call in calls), 1)
