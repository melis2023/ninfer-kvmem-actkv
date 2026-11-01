"""Check calendar tags, retry identity and cleanup admission without GitHub mutations."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / '.github/scripts'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TAG = load('release_tag', 'native-release-tag.py')
PRUNE = load('prune_release', 'prune-native-releases.py')
SHA = 'a' * 40


class ReleaseTagTest(unittest.TestCase):
    def test_first_build_and_numbers_reserved_by_tags_and_drafts(self):
        self.assertEqual(TAG.select([], [], SHA, '2026-10-23'), ('2026-10-23.1', 0, True))
        refs = [{'ref': 'refs/tags/2026-10-23.7'}, {'ref': 'refs/tags/2026-10-22.20'}]
        drafts = [dict(tag_name='2026-10-23.8', target_commitish='b'*40, id=1, draft=True)]
        self.assertEqual(TAG.select(drafts, refs, SHA, '2026-10-23')[0], '2026-10-23.9')

    def test_retry_reuses_original_day_and_legacy_release(self):
        for tag in ['2026-10-22.3', 'native-aaaaaaaaaaaa']:
            releases = [dict(tag_name=tag, target_commitish=SHA, id=9, draft=True)]
            self.assertEqual(TAG.select(releases, [], SHA, '2026-10-23'), (tag, 9, True))

    def test_published_release_wins_over_a_draft_for_same_source(self):
        releases = [dict(tag_name='2026-10-23.2', target_commitish=SHA, id=2, draft=True),
                    dict(tag_name='2026-10-23.1', target_commitish=SHA, id=1, draft=False)]
        self.assertEqual(TAG.select(releases, [], SHA, '2026-10-23'), ('2026-10-23.1', 1, False))


class ReleaseRetentionTest(unittest.TestCase):
    def setUp(self):
        self.latest = dict(id=3, tag_name='2026-10-23.3', target_commitish=SHA,
                           draft=False, prerelease=False, published_at='2026-10-23T12:00:00Z', assets=[])
        self.rows = [self.latest, dict(self.latest, id=1, tag_name='native-bbbbbbbbbbbb', target_commitish='b'*40),
                     dict(self.latest, id=2, tag_name='2026-10-23.2', draft=True),
                     dict(self.latest, id=4, tag_name='v0.12.0'),
                     dict(self.latest, id=5, tag_name='2026-10-23.4', published_at='2026-10-23T13:00:00Z')]
        self.deleted = []
        self.comparison = 'ahead'

    def api(self, path):
        if path == 'releases/latest':
            return self.latest
        if path.startswith('releases?'):
            return self.rows
        if path.startswith('compare/'):
            return {'status': self.comparison}
        raise AssertionError(path)

    def prune(self, **kwargs):
        return PRUNE.prune('test/ninfer', '2026-10-23.3', SHA, api=self.api, delete=self.deleted.append, **kwargs)

    def test_only_older_published_native_snapshots_are_removed(self):
        self.assertEqual(self.prune(), ['native-bbbbbbbbbbbb'])
        self.assertEqual(self.deleted, [1])

    def test_moved_latest_or_changed_assets_prevent_deletion(self):
        with self.assertRaisesRegex(RuntimeError, 'assets changed'):
            self.prune(verified_assets={'assets': [{'name': 'volume', 'size': 1, 'digest': 'sha256:x'}]})
        self.latest = dict(self.latest, tag_name='2026-10-23.4')
        with self.assertRaisesRegex(RuntimeError, 'Latest changed'):
            self.prune()
        self.assertEqual(self.deleted, [])

    def test_diverged_history_is_retained(self):
        self.comparison = 'diverged'
        self.assertEqual(self.prune(), [])
        self.assertEqual(self.deleted, [])
