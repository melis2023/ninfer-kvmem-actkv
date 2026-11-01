"""Exercise publication boundaries with real archives/manifests and a fake GitHub CLI."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / '.github/scripts/publish-native-release.sh'
NOTES = SCRIPT.with_name('render-native-release-notes.py')
SHA = 'a' * 40
VARIANTS = tuple(f'{os}-x64-{cuda}' for os in ('windows', 'linux')
                 for cuda in ('cuda12.9-sm70-75-80-86-89', 'cuda13.4-sm75-80-86-89-120a', 'cuda12.9-13.4'))

GH = r'''#!/usr/bin/env python3.11
import hashlib, json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['GH_CALLS'], 'a') as log:
    log.write(json.dumps(args) + '\n')
mode = os.environ.get('MODE', '')
created_path = pathlib.Path(os.environ['GH_NOTES'] + '.created')
tag = created_path.read_text() if created_path.exists() else 'native-' + os.environ['GITHUB_SHA'][:12]
if '--notes-file' in args:
    pathlib.Path(os.environ['GH_NOTES']).write_text(pathlib.Path(args[args.index('--notes-file')+1]).read_text())
if args[0] == 'api':
    route = next(a for a in args[1:] if a.startswith('repos/'))
    if mode == 'api-failure':
        sys.exit(1)
    if route.endswith('/commits/master'):
        calls = pathlib.Path(os.environ['GH_CALLS']).read_text()
        stale = mode == 'superseded' or (mode == 'superseded-upload' and '"upload"' in calls)
        print('b' * 40 if stale else os.environ['GITHUB_SHA'])
    elif route.endswith('/git/matching-refs/tags/'):
        print('[[]]')
    elif route.endswith('/releases') and '--method' in args:
        data = json.loads(pathlib.Path(args[args.index('--input')+1]).read_text())
        created_path.write_text(data['tag_name'])
        pathlib.Path(os.environ['GH_NOTES']).write_text(data['body'])
        print(1)
    elif '/releases?per_page=100' in route:
        rows = ([dict(id=1, tag_name=tag, draft=mode != 'published', target_commitish=os.environ['GITHUB_SHA'])]
                if mode in ('draft', 'published') or created_path.exists() else [])
        print(json.dumps([rows] if '--slurp' in args else rows))
    elif '/releases/tags/' in route:
        # A draft for a tag that does not exist yet is available only by release ID.
        sys.exit(1)
    elif route.endswith('/releases/1') or route.endswith('/releases/latest'):
        if '--jq' in args:
            print(tag if args[-1] == '.tag_name' else os.environ['GITHUB_SHA'])
            sys.exit(0)
        assets = []
        for p in pathlib.Path('.').iterdir():
            if '.7z.' in p.name or '.tar.gz.' in p.name or p.name.startswith('SHA256SUMS-'):
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
                assets.append(dict(name=p.name, size=p.stat().st_size,
                                   digest='sha256:' + ('0'*64 if mode == 'bad-remote' else digest)))
        print(json.dumps(dict(id=1, tag_name=tag, target_commitish=os.environ['GITHUB_SHA'], draft=False, prerelease=False, published_at='2026-10-23T12:00:00Z', assets=assets)))
    else:
        raise SystemExit('Unexpected API call: ' + route)
elif args[:2] not in (['release', 'create'], ['release', 'upload'], ['release', 'edit']):
    raise SystemExit('Unexpected gh call: ' + repr(args))
'''


class NativePublicationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dist = self.root / 'dist'
        self.dist.mkdir()
        bindir = self.root / 'bin'
        bindir.mkdir()
        (bindir / 'gh').write_text(GH)
        (bindir / 'gh').chmod(0o755)
        self.calls = self.root / 'calls.jsonl'
        self.notes = self.root / 'notes.md'
        self.env = dict(os.environ, PATH=f'{bindir}:{os.environ["PATH"]}',
                        GITHUB_SHA=SHA, GITHUB_REPOSITORY='test/ninfer',
                        GITHUB_REF='refs/heads/master', GH_CALLS=str(self.calls), GH_NOTES=str(self.notes))
        for variant in VARIANTS:
            stem = f'ninfer-{variant}-{SHA[:12]}'
            extension = '7z' if variant.startswith('windows-') else 'tar.gz'
            volume = self.dist / f'{stem}.{extension}.001'
            volume.write_bytes(b'archive fixture ' + variant.encode())
            digest = hashlib.sha256(volume.read_bytes()).hexdigest()
            (self.dist / f'SHA256SUMS-{stem}.txt').write_text(f'{digest}  {volume.name}\n')

    def run_publish(self, mode='', success=True):
        result = subprocess.run(['bash', str(SCRIPT)], cwd=self.root,
                                env=dict(self.env, MODE=mode), text=True, capture_output=True)
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        return [json.loads(s) for s in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def test_complete_matrix_publishes_after_remote_verification(self):
        calls = self.run_publish()
        edit = next(i for i, c in enumerate(calls) if c[:2] == ['release', 'edit'])
        verification = next(i for i, c in enumerate(calls) if any(a.endswith('/releases/1') for a in c))
        self.assertLess(verification, edit)
        self.assertIn('--latest', calls[edit])
        self.assertTrue(any('--method' in c and 'POST' in c and any(a.endswith('/releases') for a in c) for c in calls))

    def test_corrupt_archive_never_contacts_github(self):
        next(self.dist.glob('*.7z.001')).write_bytes(b'corrupt')
        self.assertEqual(self.run_publish(success=False), [])

    def test_missing_variant_never_contacts_github(self):
        next(self.dist.glob('SHA256SUMS-*')).unlink()
        self.assertEqual(self.run_publish(success=False), [])

    def test_unlisted_volume_never_contacts_github(self):
        (self.dist / 'ninfer-windows-extra.7z.002').write_bytes(b'extra')
        self.assertEqual(self.run_publish(success=False), [])

    def test_superseded_source_cannot_publish(self):
        calls = self.run_publish('superseded')
        self.assertFalse(any(c[0] == 'release' for c in calls))

    def test_api_failure_does_not_create_release(self):
        calls = self.run_publish('api-failure', success=False)
        self.assertFalse(any(c[0] == 'release' for c in calls))

    def test_superseded_during_upload_keeps_draft(self):
        calls = self.run_publish('superseded-upload')
        self.assertTrue(any(c[:2] == ['release', 'upload'] for c in calls))
        self.assertFalse(any(c[:2] == ['release', 'edit'] for c in calls))

    def test_bad_remote_digest_leaves_draft(self):
        calls = self.run_publish('bad-remote', success=False)
        self.assertTrue(any(c[:2] == ['release', 'upload'] for c in calls))
        self.assertFalse(any(c[:2] == ['release', 'edit'] for c in calls))

    def test_draft_retry_reuses_release(self):
        calls = self.run_publish('draft')
        self.assertFalse(any(c[:2] == ['release', 'create'] for c in calls))
        self.assertFalse(any('--method' in c and 'POST' in c for c in calls))
        self.assertTrue(any(c[:2] == ['release', 'edit'] for c in calls))

    def test_notes_list_every_package_and_its_actual_volumes(self):
        stem = f'ninfer-linux-x64-cuda12.9-13.4-{SHA[:12]}'
        part = self.dist / f'{stem}.tar.gz.002'
        part.write_bytes(b'second volume')
        manifest = self.dist / f'SHA256SUMS-{stem}.txt'
        with manifest.open('a') as stream:
            stream.write(f'{hashlib.sha256(part.read_bytes()).hexdigest()}  {part.name}\n')
        self.run_publish()
        notes = self.notes.read_text()
        self.assertIn('Windows x64', notes)
        self.assertIn('Debian 13 x64', notes)
        tag = Path(str(self.notes) + '.created').read_text()
        self.assertRegex(tag, r'^\d{4}-\d{2}-\d{2}\.1$')
        for variant in VARIANTS:
            name = f'ninfer-{variant}-{SHA[:12]}'
            extension = '7z' if variant.startswith('windows-') else 'tar.gz'
            self.assertIn(f'[{name}.{extension}.001](https://github.com/test/ninfer/releases/download/{tag}/{name}.{extension}.001)', notes)
            self.assertIn(f'[SHA256SUMS-{name}.txt]', notes)
        self.assertIn(f'[{part.name}]', notes)
        total = len(b'archive fixture linux-x64-cuda12.9-13.4') + len(b'second volume')
        self.assertIn(f'2 parts | 0.00 GiB ({total:,} bytes)', notes)
        self.assertIn(f'cat {stem}.tar.gz.001 {stem}.tar.gz.002 | tar -xz', notes)
        self.assertIn('7z x ninfer-windows-', notes)
        self.assertIn('glibc 2.41', notes)
        self.assertIn('--cuda-backend 12', notes)
        self.assertIn('Volta', notes)
        self.assertIn('Blackwell', notes)

    def test_draft_retry_refreshes_download_guide_before_publication(self):
        calls = self.run_publish('draft')
        edit = next(c for c in calls if c[:2] == ['release', 'edit'])
        self.assertIn('--notes-file', edit)
        self.assertIn('## Downloads', self.notes.read_text())

    def test_notes_preview_uses_remote_sizes_without_archive_downloads(self):
        assets = []
        for path in self.dist.iterdir():
            assets.append(dict(name=path.name, size=1992294400 if path.name.endswith('.001') else path.stat().st_size))
        for path in self.dist.glob('*.001'):
            path.unlink()
        metadata = self.root / 'release.json'
        metadata.write_text(json.dumps({'assets': assets}))
        result = subprocess.run(['python3.11', str(NOTES), '--dist', str(self.dist),
                                 '--repository', 'test/ninfer', '--revision', SHA, '--tag', '2026-10-23.1',
                                 '--assets-json', str(metadata), '--output', str(self.notes)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('## Downloads', self.notes.read_text())
        self.assertIn('1.86 GiB (1,992,294,400 bytes)', self.notes.read_text())

    def test_notes_reject_missing_remote_volume(self):
        metadata = self.root / 'release.json'
        metadata.write_text(json.dumps({'assets': []}))
        result = subprocess.run(['python3.11', str(NOTES), '--dist', str(self.dist),
                                 '--repository', 'test/ninfer', '--revision', SHA, '--tag', '2026-10-23.1',
                                 '--assets-json', str(metadata), '--output', str(self.notes)],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.notes.exists())

    def test_notes_reject_zero_size_remote_volume(self):
        metadata = self.root / 'release.json'
        metadata.write_text(json.dumps({'assets': [dict(name=p.name, size=0) for p in self.dist.iterdir()]}))
        result = subprocess.run(['python3.11', str(NOTES), '--dist', str(self.dist),
                                 '--repository', 'test/ninfer', '--revision', SHA, '--tag', '2026-10-23.1',
                                 '--assets-json', str(metadata), '--output', str(self.notes)],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Invalid asset size:', result.stderr)
        self.assertFalse(self.notes.exists())

    def test_published_snapshot_is_not_modified(self):
        calls = self.run_publish('published')
        self.assertFalse(any(c[0] == 'release' for c in calls))
        self.assertFalse(any('--method' in c for c in calls))

    def test_other_branch_cannot_publish(self):
        self.env['GITHUB_REF'] = 'refs/heads/feature'
        self.assertEqual(self.run_publish(success=False), [])


if __name__ == '__main__':
    unittest.main()
