"""Check cancellation, failure propagation, and source selection without a compiler."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/run-cached-build.py"
SPEC = importlib.util.spec_from_file_location("cancel_master", ROOT / ".github/scripts/cancel-master-builds.py")
CANCEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CANCEL)
SPEC = importlib.util.spec_from_file_location("wait_builds", ROOT / ".github/scripts/wait-previous-builds.py")
WAIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WAIT)


class BuildLifecycleTest(unittest.TestCase):
    def test_packaging_can_disable_inherited_cache_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "huggingface_hub.py").write_text(
                "from pathlib import Path\nPath('unexpected-cache-write').touch()\n"
                "raise RuntimeError('Packaging must not start a cache upload')\n")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--log", str(root / "package.log"),
                 "--checkpoint-family=", "--checkpoint-interval", ".01", "--", sys.executable,
                 "-c", "import time; print('archive complete', flush=True); time.sleep(.5)"],
                cwd=root, env=os.environ | {"HF_TOKEN": "test-only", "PYTHONPATH": str(root),
                                           "HF_CACHE_FAMILY": "native-windows-cuda-test"},
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("archive complete", (root / "package.log").read_text())
            self.assertFalse((root / "unexpected-cache-write").exists())

    def test_soft_timeout_stops_work_and_preserves_completed_outputs(self):
        for option in ("duration", "job-deadline"):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env = os.environ | {"TEST_BUILD_DEADLINE": str(time.time() + 2)}
                budget = (["--timeout-seconds", "1"] if option == "duration" else
                          ["--timeout-seconds", "60", "--deadline-env", "TEST_BUILD_DEADLINE"])
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), "--log", str(root / "build.log"),
                     "--pid-file", str(root / "build.pid"), *budget, "--", sys.executable, "-c",
                     "from pathlib import Path; import time; "
                     "Path('completed-object').write_bytes(b'object'); time.sleep(30); "
                     "Path('should-not-finish').touch()"],
                    cwd=root, env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 124, result.stderr)
                self.assertIn("Build deadline reached", result.stdout)
                self.assertEqual((root / "completed-object").read_bytes(), b"object")
                self.assertFalse((root / "should-not-finish").exists())
                self.assertFalse((root / "build.pid").exists())

    def test_timeout_stops_descendants_after_the_build_parent_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "grandchild.py").write_text(
                "import signal, time\nfrom pathlib import Path\n"
                "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
                "if hasattr(signal, 'SIGBREAK'): signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n"
                "Path('completed-object').write_bytes(b'object')\n"
                "while True:\n"
                " Path('compiler-progress').write_text(str(time.time_ns()))\n"
                " time.sleep(.05)\n")
            (root / "parent.py").write_text(
                "import os, signal, subprocess, sys, time\nfrom pathlib import Path\n"
                "def stop(*args): raise SystemExit(130)\n"
                "signal.signal(signal.SIGINT, stop)\n"
                "if hasattr(signal, 'SIGBREAK'): signal.signal(signal.SIGBREAK, stop)\n"
                "Path('parent-pid').write_text(str(os.getpid()))\n"
                "subprocess.Popen([sys.executable, 'grandchild.py'])\n"
                "while True: time.sleep(.1)\n")
            try:
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), "--log", str(root / "build.log"),
                     "--timeout-seconds", "2", "--", sys.executable, "parent.py"],
                    cwd=root, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 124, result.stderr)
                progress = (root / "compiler-progress").read_text()
                time.sleep(.3)
                self.assertEqual((root / "compiler-progress").read_text(), progress,
                                 "Compiler descendant survived its cancelled parent")
                self.assertEqual((root / "completed-object").read_bytes(), b"object")
            finally:
                if os.name != "nt" and (root / "parent-pid").exists():
                    try:
                        os.killpg(int((root / "parent-pid").read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_timeout_stops_an_inflight_checkpoint_without_losing_local_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "objects"
            cache.mkdir()
            (cache / "completed").write_bytes(b"object")
            # A stalled transport must not keep the writer lock or outlive the supervisor.
            (root / "huggingface_hub.py").write_text(
                "from pathlib import Path\nfrom types import SimpleNamespace\nimport time\n"
                "class HfApi:\n"
                " def bucket_info(self, *a): return SimpleNamespace(private=True)\n"
                " def get_bucket_paths_info(self, *a): return []\n"
                " def batch_bucket_files(self, *a, **kw):\n"
                "  while True:\n"
                "   Path('transfer-progress').write_text(str(time.time_ns()))\n"
                "   time.sleep(.05)\n")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--log", str(root / "build.log"),
                 "--timeout-seconds", "2", "--checkpoint-family", "native-linux-cuda-test",
                 "--checkpoint-directory", str(cache), "--checkpoint-interval", ".1",
                 "--", sys.executable, "-c", "import time; time.sleep(30)"],
                cwd=root, env=os.environ | {"HF_TOKEN": "test-only", "PYTHONPATH": str(root),
                                           "GITHUB_REF": "refs/heads/test", "GITHUB_RUN_ID": "1",
                                           "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_JOB": "linux",
                                           "GITHUB_WORKFLOW_REF": "iamwavecut/ninfer-all/.github/workflows/native-release.yml@refs/heads/test"},
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 124, result.stderr)
            progress = (root / "transfer-progress").read_text()
            time.sleep(.2)
            self.assertEqual((root / "transfer-progress").read_text(), progress)
            self.assertEqual((cache / "completed").read_bytes(), b"object")

    def test_expired_job_budget_does_not_start_a_compiler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--log", str(root / "build.log"),
                 "--deadline-env", "TEST_BUILD_DEADLINE", "--", sys.executable, "-c",
                 "from pathlib import Path; Path('started').touch()"], cwd=root,
                env=os.environ | {"TEST_BUILD_DEADLINE": str(time.time() - 1)},
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 124)
            self.assertFalse((root / "started").exists())

    def test_handoff_waits_for_cleanup_but_never_waits_for_a_newer_run(self):
        polls = []

        def api(command, **_kwargs):
            path = command[-1]
            if path.endswith("/actions/runs/10"):
                return json.dumps(dict(workflow_id=1, head_branch="master", run_number=3))
            if "/workflows/1/runs?" in path:
                return json.dumps([dict(workflow_runs=[
                    dict(id=9, run_number=2), dict(id=11, run_number=4)])])
            self.assertTrue(path.endswith("/actions/runs/9"))
            polls.append(path)
            return json.dumps(dict(status="in_progress" if len(polls) == 1 else "completed"))

        with patch.dict(os.environ, GITHUB_REPOSITORY="owner/repo", GITHUB_RUN_ID="10"), \
                patch.object(WAIT.subprocess, "check_output", side_effect=api), \
                patch.object(WAIT.time, "sleep") as sleep:
            WAIT.main()
            sleep.assert_called_once_with(15)
        self.assertEqual(len(polls), 2)

    def test_build_failure_is_not_hidden_by_logging(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "build.log"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--log", str(log), "--",
                 sys.executable, "-c", "print('compiler failed'); raise SystemExit(23)"],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 23)
            self.assertIn("compiler failed", log.read_text())

    def test_cancel_stops_build_and_preserves_completed_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "child.py"
            child.write_text(
                "import signal, time\nfrom pathlib import Path\n"
                "def stop(*args):\n    Path('stopped').touch()\n    raise SystemExit(130)\n"
                "signal.signal(signal.SIGINT, stop)\n"
                "if hasattr(signal, 'SIGBREAK'): signal.signal(signal.SIGBREAK, stop)\n"
                "Path('completed-cache-entry').write_text('completed output')\n"
                "while True: time.sleep(.1)\n")
            options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                       if os.name == "nt" else {"start_new_session": True})
            process = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--log", str(root / "build.log"),
                 "--pid-file", str(root / "build.pid"), "--", sys.executable, str(child)],
                cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options)
            try:
                deadline = time.monotonic() + 8
                while not (root / "completed-cache-entry").exists():
                    if process.poll() is not None or time.monotonic() > deadline:
                        self.fail("Fixture build did not start")
                    time.sleep(.05)
                process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
                process.communicate(timeout=7)
                self.assertEqual(process.returncode, 130)
                self.assertTrue((root / "stopped").exists())
                self.assertEqual((root / "completed-cache-entry").read_text(), "completed output")
                self.assertFalse((root / "build.pid").exists())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

    def test_master_selection_preserves_prs_other_workflows_and_new_runs(self):
        trigger = dict(head_sha="b" * 40, created_at="2026-10-20T12:00:00Z")
        run = dict(status="in_progress", head_branch="master", event="push",
                   path=".github/workflows/native-release.yml", head_sha="a" * 40,
                   created_at="2026-10-20T11:00:00Z")
        self.assertTrue(CANCEL.superseded(run, trigger))
        for changes in (dict(head_branch="build/windows-release"), dict(event="pull_request"),
                        dict(status="completed"), dict(head_sha=trigger["head_sha"]),
                        dict(path=".github/workflows/other.yml"),
                        dict(created_at="2026-10-20T12:00:01Z")):
            with self.subTest(changes=changes):
                self.assertFalse(CANCEL.superseded(run | changes, trigger))
