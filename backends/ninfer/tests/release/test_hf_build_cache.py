"""Check checkpoint recovery and preservation without network access."""
import importlib.util
from contextlib import redirect_stdout
import io
import json
import os
import subprocess
import sys
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('hf_cache', Path(__file__).resolve().parents[2] / '.github/scripts/hf-build-cache.py')
CACHE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHE)


class Bucket:
    def __init__(self):
        self.files = {}
        self.interrupt = False

    def get_bucket_paths_info(self, bucket, paths):
        return [SimpleNamespace(path=p, size=len(self.files[p])) for p in paths if p in self.files]

    def batch_bucket_files(self, bucket, add):
        for local, remote in add:
            self.files[remote] = Path(local).read_bytes()
            if self.interrupt and remote.endswith('.tar'):
                self.files[remote] = b'partial upload'
                raise ConnectionError('Interrupted upload')

    def download_bucket_files(self, bucket, files, **kwargs):
        for remote, local in files:
            Path(local).write_bytes(self.files[remote])

    def list_bucket_tree(self, bucket, prefix, **kwargs):
        return [SimpleNamespace(path=p) for p in self.files if p.startswith(prefix)]


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.bucket = Bucket()
        self.ref = 'refs/heads/build/windows-release'
        self.family = 'linux-cuda-13.4.3'

    def save(self, content):
        (self.source / 'object').write_text(content)
        CACHE.save(self.bucket, self.source, self.family, self.ref, verify=True)

    def restored(self):
        target = self.root / 'restored'
        self.assertTrue(CACHE.restore(self.bucket, target, self.family, self.ref, required=True))
        return (target / 'object').read_text()

    def test_completed_snapshot_roundtrip_and_bounded_slots(self):
        for content in ('first', 'second', 'third'):
            self.save(content)
        self.assertEqual(self.restored(), 'third')
        self.assertEqual(len(self.bucket.files), 4)

    def test_new_family_reads_previous_objects_without_replacing_old_snapshots(self):
        self.save('old compiler objects')
        preserved = self.bucket.files.copy()
        target = self.root / 'new-cache'
        new_family = self.family + '-host-v2'
        self.assertTrue(CACHE.restore(self.bucket, target, new_family, self.ref,
                                     fallback_families=[self.family], required=True))
        (target / 'new-object').write_text('new compiler objects')
        CACHE.save(self.bucket, target, new_family, self.ref)
        self.assertTrue(all(self.bucket.files[key] == value for key, value in preserved.items()))
        again = self.root / 'again'
        self.assertTrue(CACHE.restore(self.bucket, again, new_family, self.ref,
                                     fallback_families=[self.family]))
        self.assertEqual((again / 'new-object').read_text(), 'new compiler objects')

    def test_corrupt_new_family_cannot_silently_use_an_old_family(self):
        self.save('previous')
        new_family = self.family + '-host-v2'
        CACHE.save(self.bucket, self.source, new_family, self.ref)
        self.bucket.files[CACHE.prefix(new_family, self.ref) + '/0.tar'] = b'corrupt'
        with self.assertRaises(ValueError):
            CACHE.restore(self.bucket, self.root / 'new-cache', new_family, self.ref,
                          fallback_families=[self.family])

    def test_interrupted_replacement_retains_last_completed_snapshot(self):
        self.save('first')
        self.save('second')
        self.bucket.interrupt = True
        with self.assertRaises(ConnectionError):
            self.save('third')
        self.assertEqual(self.restored(), 'second')

    def test_corrupt_newest_snapshot_falls_back(self):
        self.save('first')
        self.save('second')
        base = CACHE.prefix(self.family, self.ref)
        self.bucket.files[f'{base}/1.tar'] = b'corrupt'
        self.assertEqual(self.restored(), 'first')

    def test_all_corrupt_snapshots_fail_instead_of_silently_starting_cold(self):
        self.save('first')
        base = CACHE.prefix(self.family, self.ref)
        self.bucket.files[f'{base}/0.tar'] = b'corrupt'
        with self.assertRaises(ValueError):
            CACHE.restore(self.bucket, self.root / 'restored', self.family, self.ref)

    def test_empty_cache_cannot_replace_completed_checkpoint(self):
        self.save('first')
        (self.source / 'object').unlink()
        with self.assertRaises(ValueError):
            CACHE.save(self.bucket, self.source, self.family, self.ref)
        self.assertEqual(self.restored(), 'first')

    def test_other_branch_cannot_supply_objects_but_master_can(self):
        self.save('private branch')
        target = self.root / 'other'
        self.assertFalse(CACHE.restore(self.bucket, target, self.family, 'refs/heads/other'))
        CACHE.save(self.bucket, self.source, self.family, 'refs/heads/master')
        self.assertTrue(CACHE.restore(self.bucket, target, self.family, 'refs/heads/other'))

    def test_explicit_seed_bootstraps_master_without_writing_to_bucket(self):
        self.save('completed seed object')
        before = self.bucket.files.copy()
        target = self.root / 'master'
        self.assertTrue(CACHE.restore(self.bucket, target, self.family, 'refs/heads/master',
                                      required=True, seed_ref=self.ref))
        self.assertEqual((target / 'object').read_text(), 'completed seed object')
        self.assertEqual(self.bucket.files, before)

    def test_master_checkpoint_supersedes_initial_seed(self):
        self.save('initial seed')
        (self.source / 'object').write_text('current master')
        CACHE.save(self.bucket, self.source, self.family, 'refs/heads/master')
        target = self.root / 'reader'
        CACHE.restore(self.bucket, target, self.family, 'refs/heads/other', seed_ref=self.ref)
        self.assertEqual((target / 'object').read_text(), 'current master')

    def test_seed_checksums_are_required_and_pr_refs_are_rejected(self):
        self.save('initial seed')
        self.bucket.files[CACHE.prefix(self.family, self.ref) + '/0.tar'] = b'corrupt'
        with self.assertRaises(ValueError):
            CACHE.restore(self.bucket, self.root / 'reader', self.family, 'refs/heads/master',
                          required=True, seed_ref=self.ref)
        with self.assertRaises(ValueError):
            CACHE.restore(self.bucket, self.root / 'reader', self.family, 'refs/heads/master',
                          seed_ref='refs/pull/1/merge')

    def test_migration_merges_all_preserved_objects(self):
        for index in (1, 2):
            source = self.root / f'seed{index}'
            source.mkdir()
            (source / f'object{index}').write_text(str(index))
            archive = self.root / f'seed{index}.tar'
            CACHE.pack(source, archive)
            key = f'objects-v1-linux-cuda-13.4.3-{index}'
            remote = f'github-actions/migration/{key}.tar'
            metadata = dict(schema=1, key=key, archive=remote, source_ref=self.ref,
                            sha256=CACHE.digest(archive), files=1)
            self.bucket.files[remote] = archive.read_bytes()
            self.bucket.files[remote + '.json'] = json.dumps(metadata).encode()
        target = self.root / 'restored'
        CACHE.restore(self.bucket, target, self.family, self.ref,
                      legacy_prefix='objects-v1-linux-cuda-13.4.3-', required=True)
        self.assertEqual({p.name for p in target.iterdir()}, {'object1', 'object2'})

    def test_unsafe_members_are_rejected_before_extracting_any_files(self):
        for name, kind in (('../escape', tarfile.REGTYPE), ('/absolute', tarfile.REGTYPE),
                           ('C:/escape', tarfile.REGTYPE), ('link', tarfile.SYMTYPE)):
            with self.subTest(name=name):
                archive = self.root / 'unsafe.tar'
                with tarfile.open(archive, 'w') as package:
                    good = tarfile.TarInfo('good')
                    good.size = 1
                    package.addfile(good, io.BytesIO(b'x'))
                    bad = tarfile.TarInfo(name)
                    bad.type = kind
                    package.addfile(bad)
                with self.assertRaises(ValueError):
                    CACHE.unpack(archive, self.root / 'destination')
                self.assertFalse((self.root / 'destination/good').exists())

    def test_pack_excludes_mutable_statistics_and_temporary_entries(self):
        (self.source / 'tmp').mkdir()
        for name in ('object', 'stats', 'lock', 'entry.tmp', 'tmp/uncommitted'):
            (self.source / name).write_text(name)
        archive = self.root / 'test.tar'
        self.assertEqual(CACHE.pack(self.source, archive), 1)
        with tarfile.open(archive) as package:
            self.assertEqual(package.getnames(), ['object'])


class CacheOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.env = dict(GITHUB_REF='refs/heads/master', GITHUB_RUN_ID='200',
                        GITHUB_JOB='linux', GITHUB_EVENT_NAME='push',
                        GITHUB_WORKFLOW_REF='iamwavecut/ninfer-all/.github/workflows/native-release.yml@refs/heads/master')

    def test_only_native_builds_can_write_their_own_families(self):
        with patch.dict(os.environ, self.env, clear=True):
            CACHE.require_native_writer('native-linux-cuda-13.4.3', 'refs/heads/master')
            for override in (dict(GITHUB_JOB='cuda'), dict(GITHUB_EVENT_NAME='pull_request'),
                             dict(GITHUB_WORKFLOW_REF='iamwavecut/ninfer-all/.github/workflows/ci.yml@refs/heads/master')):
                with patch.dict(os.environ, override), self.assertRaises(ValueError):
                    CACHE.require_native_writer('native-linux-cuda-13.4.3', 'refs/heads/master')
            with self.assertRaises(ValueError):
                CACHE.require_native_writer('native-windows-cuda-13.4.3', 'refs/heads/master')
            with patch.dict(os.environ, GITHUB_JOB='windows'):
                CACHE.require_native_writer('vcpkg-baseline', 'refs/heads/master')

    def test_bootstrap_can_write_only_its_linux_shared_cache(self):
        env = dict(self.env, GITHUB_EVENT_NAME='workflow_dispatch',
                   GITHUB_WORKFLOW_REF='iamwavecut/ninfer-all/.github/workflows/linux-cache-bootstrap.yml@refs/heads/build/windows-release')
        with patch.dict(os.environ, env, clear=True):
            CACHE.require_native_writer('native-linux-cuda-13.4.3-shared-v1', env['GITHUB_REF'])
            for family in ('native-linux-cuda-13.4.3-host-v2', 'native-windows-cuda-13.4.3', 'vcpkg-test'):
                with self.assertRaises(ValueError):
                    CACHE.require_native_writer(family, env['GITHUB_REF'])
            with patch.dict(os.environ, GITHUB_EVENT_NAME='pull_request'), self.assertRaises(ValueError):
                CACHE.require_native_writer('native-linux-cuda-13.4.3-shared-v1', env['GITHUB_REF'])

    def test_older_rerun_cannot_replace_newer_native_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'objects'
            source.mkdir()
            (source / 'object').write_text('new build')
            bucket = Bucket()
            with patch.dict(os.environ, GITHUB_RUN_ID='200'):
                CACHE.save(bucket, source, 'native-linux-cuda-test', 'refs/heads/master')
            original = bucket.files.copy()
            (source / 'object').write_text('old rerun')
            with patch.dict(os.environ, GITHUB_RUN_ID='100'):
                self.assertFalse(CACHE.save(bucket, source, 'native-linux-cuda-test', 'refs/heads/master'))
            self.assertEqual(bucket.files, original)

    def test_optional_reader_without_secret_can_start_cold_but_writer_cannot(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, SPEC.origin, 'restore', '--directory', directory,
                       '--family', 'native-linux-cuda-test', '--source-ref', 'refs/heads/master']
            env = os.environ | self.env
            env.pop('HF_TOKEN', None)
            optional = subprocess.run(command + ['--optional'], env=env, capture_output=True, text=True)
            self.assertEqual(optional.returncode, 0, optional.stderr)
            self.assertIn('::warning::Native cache unavailable', optional.stdout)
            required = subprocess.run(command + ['--required'], env=env, capture_output=True)
            self.assertNotEqual(required.returncode, 0)

    def test_optional_reader_transport_failure_warns_without_leaking_url(self):
        class Unavailable:
            def bucket_info(self, *_args):
                raise ConnectionError('https://secret-signed-url.invalid')

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, self.env | {'HF_TOKEN': 'test-only'}), \
                patch.dict(sys.modules, huggingface_hub=SimpleNamespace(HfApi=Unavailable)), \
                patch.object(sys, 'argv', ['cache', 'restore', '--directory', directory,
                                          '--family', 'native-linux-cuda-test', '--optional']), \
                redirect_stdout(output):
            CACHE.main()
        self.assertIn('ConnectionError', output.getvalue())
        self.assertNotIn('secret-signed-url', output.getvalue())

    def test_pr_reads_target_branch_without_modifying_bucket(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'producer'
            source.mkdir()
            (source / 'object').write_text('release object')
            bucket = Bucket()
            family = 'native-linux-cuda-test'
            CACHE.save(bucket, source, family, 'refs/heads/master')
            snapshot = bucket.files.copy()
            bucket.bucket_info = lambda *_args: SimpleNamespace(private=True)
            with patch.dict(os.environ, self.env | {'GITHUB_REF': 'refs/pull/1/merge', 'HF_TOKEN': 'test-only'}), \
                    patch.dict(sys.modules, huggingface_hub=SimpleNamespace(HfApi=lambda: bucket)), \
                    patch.object(sys, 'argv', ['cache', 'restore', '--directory', str(root / 'reader'),
                                              '--family', family, '--source-ref', 'refs/heads/master']):
                CACHE.main()
            self.assertEqual((root / 'reader/object').read_text(), 'release object')
            self.assertEqual(bucket.files, snapshot)
