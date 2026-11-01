"""Check build admission with Git history and isolated registry/API responses."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('change_scopes', ROOT / '.github/scripts/change-scopes.py')
SCOPES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCOPES)


class ChangeScopesTest(unittest.TestCase):
    def test_empty_and_documentation_changes_have_no_work(self):
        for paths in ([], ['README.md', 'docs/guide.md', 'AGENTS.md'], ['src/models/README.md']):
            self.assertFalse(any(SCOPES.classify(paths).values()))

    def test_release_notes_and_pipeline_controls_do_not_compile(self):
        result = SCOPES.classify(['.github/scripts/render-native-release-notes.py',
                                  '.github/scripts/publish-native-release.sh',
                                  '.github/workflows/native-release.yml',
                                  '.github/scripts/change-scopes.py', 'tests/release/test_native_publication.py'])
        for scope in ['changed', 'native', 'docker']:
            self.assertFalse(result[scope])
        self.assertTrue(result['release_tools'])

    def test_product_sources_and_compiler_inputs_build_both(self):
        for path in ['src/ops/foo.cu', 'include/ninfer/engine.h', 'apps/launcher/main.cpp',
                     'third_party/ggml/a.cpp', 'cmake/NinferRuntime.cmake', 'CMakePresets.json',
                     '.github/scripts/configure-linux-build.sh', '.github/workflows/native-linux.yml',
                     'bench/ops/foo.cu']:
            with self.subTest(path=path):
                result = SCOPES.classify([path])
                self.assertTrue(result['native'])
                self.assertTrue(result['changed'])

    def test_tests_and_python_tools_do_not_repackage_native(self):
        result = SCOPES.classify(['tests/test_sampling.cpp'])
        self.assertTrue(result['changed'])
        self.assertFalse(result['native'])
        result = SCOPES.classify(['tools/convert/foo.py', 'tests/convert/test_foo.py'])
        self.assertTrue(result['python'])
        self.assertFalse(result['changed'])
        self.assertFalse(result['native'])

    def test_runtime_and_calculator_changes_select_their_checks(self):
        result = SCOPES.classify(['scripts/run.sh'])
        self.assertTrue(result['native'])
        self.assertTrue(result['linux_scripts'])
        self.assertFalse(result['changed'])
        result = SCOPES.classify(['docs/config-calculator.html'])
        self.assertTrue(result['calculator'])
        self.assertFalse(result['changed'])


class BuildInputsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git('init', '-q')
        self.git('config', 'user.name', 'test')
        self.git('config', 'user.email', 'test@example.invalid')
        self.git('commit', '--allow-empty', '-qm', 'base')
        self.base = self.git('rev-parse', 'HEAD').strip()
        bindir = self.root / 'bin'
        bindir.mkdir()
        for name in ['docker', 'gh']:
            path = bindir / name
            path.write_text('#!/bin/sh\nprintf "%s\\n" "$TEST_BASE"\n')
            path.chmod(0o755)
        self.env = dict(os.environ, PATH=f'{bindir}:{os.environ["PATH"]}', TEST_BASE=self.base,
                        GITHUB_EVENT_NAME='push', GITHUB_REF='refs/heads/master', GITHUB_REPOSITORY='test/ninfer')

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.root, text=True)

    def commit(self, path):
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('changed\n')
        self.git('add', path)
        self.git('commit', '-qm', path)

    def scopes(self, native=False):
        p = subprocess.run(['bash', str(ROOT / '.github/scripts/build-inputs.sh'), *(['--native'] if native else [])],
                           cwd=self.root, env=self.env, text=True, capture_output=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        return dict(line.split('=') for line in p.stdout.splitlines())

    def test_empty_pr_and_documentation_skip_all_jobs(self):
        self.git('commit', '--allow-empty', '-qm', 'empty')
        self.env.update(GITHUB_EVENT_NAME='pull_request', GITHUB_REF='refs/pull/1/merge')
        self.assertEqual(set(self.scopes().values()), {'false'})
        self.commit('docs/guide.md')
        self.env.update(GITHUB_EVENT_NAME='push', GITHUB_REF='refs/heads/master')
        for native in [False, True]:
            self.assertEqual(set(self.scopes(native).values()), {'false'})

    def test_unpublished_source_survives_a_later_documentation_push(self):
        self.commit('src/engine.cpp')
        built = self.git('rev-parse', 'HEAD').strip()
        self.commit('README.md')
        for native in [False, True]:
            self.assertEqual(self.scopes(native)['changed'], 'true')
        self.env['TEST_BASE'] = built
        for native in [False, True]:
            self.assertEqual(self.scopes(native)['native'], 'false')
            self.assertEqual(self.scopes(native)['changed'], 'false')

    def test_manual_dispatch_forces_all_and_missing_base_does_not_build(self):
        self.env['GITHUB_EVENT_NAME'] = 'workflow_dispatch'
        self.assertEqual(set(self.scopes().values()), {'true'})
        self.env.update(GITHUB_EVENT_NAME='push', TEST_BASE='')
        p = subprocess.run(['bash', str(ROOT / '.github/scripts/build-inputs.sh'), '--native'],
                           cwd=self.root, env=self.env, text=True, capture_output=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertNotIn('=true', p.stdout)
