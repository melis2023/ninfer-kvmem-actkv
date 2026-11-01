#!/usr/bin/env python3
"""Publish the qualified profile in the existing table repository and retain its provenance."""
import hashlib
import json
import os
from pathlib import Path
import statistics

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

ROOT = Path('/workspace/ninfer-work')
JOB = Path(os.environ['NINFER_JOB_DIR'])
REPO = 'WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3'
REV = '6d7c2a69e8c5fd7813c864c6f6fa4be6f0e17cc3'
NAME = 'Qwen3.8-Flash-Next-ngram-table-IQ4_NL-ninfer-v3.ninfer'
PROFILE_NAME = 'broad-v1.hot'
BUCKET = 'WaveCut/ninfer-cache'
PREFIX = 'ngram-profile-20261008/broad-v1'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    attachment = json.loads((ROOT / 'ngram-profile/broad-v1/attachment.json').read_text())
    engine = json.loads((ROOT / 'jobs/broad-profile-engine/qualification.json').read_text())
    heldout = json.loads((ROOT / 'ngram-profile/broad-v1/heldout.json').read_text())
    manifest = json.loads((ROOT / 'ngram-corpus/broad-v1/manifest.json').read_text())
    if engine['status'] != 'passed' or engine['attachment'] != attachment:
        raise RuntimeError('the embedded profile has no matching Engine qualification')
    profile = ROOT / 'ngram-profile/broad-v1/broad-v1.hot'
    if digest(profile) != attachment['profile_sha256']:
        raise RuntimeError('the qualified profile changed')
    api = HfApi()
    if api.bucket_info(BUCKET).private is not True:
        raise RuntimeError('profile archive bucket must be private')
    api.batch_bucket_files(BUCKET, add=[(profile, PREFIX + '/' + PROFILE_NAME)])
    readback = ROOT / 'ngram-profile/profile-readback.hot'
    api.download_bucket_files(BUCKET, files=[(PREFIX + '/' + PROFILE_NAME, readback)], raise_on_missing_files=True)
    if digest(readback) != attachment['profile_sha256']:
        raise RuntimeError('archived profile failed its SHA-256 readback')
    readback.unlink()
    print('PROFILE_PRIVATE_ARCHIVE_VERIFIED', flush=True)
    before = api.model_info(REPO, files_metadata=True)
    if before.private or before.sha != REV:
        raise RuntimeError('the public table changed since the pinned audit')
    original = lambda name: Path(hf_hub_download(REPO, name, revision=REV)).read_text()
    card = original('README.md')
    old_size = 'of 28,800,142,336 bytes\n(26.82 GiB)'
    if card.count(old_size) != 1:
        raise RuntimeError('the original table size is not unambiguous')
    card = card.replace(old_size, f"of {attachment['bytes']:,} bytes\n({attachment['bytes'] / 2**30:.2f} GiB)")
    card = card.replace('Conversion command, from the repository\'s\ntree:',
                        'Original row conversion command, before profile attachment:')
    card = card.replace('Nothing reads the table at load:', 'With the default disk residency, nothing reads all table rows at load:')
    card = card.replace('`--ngram-ram` loads all of it into RAM instead.',
                        '`--ngram-residency ram` loads all of it into RAM instead.')
    table_rows = '\n'.join(f"| {budget >> 20} | {rate:.2f}% |" for budget, rate in
                            zip(heldout['budgets_bytes'], heldout['total']['hit_percent']))
    groups = [g for g in heldout['groups'] if g['tokens'] >= 2048]
    def spread(index):
        values = [g['hit_percent'][index] for g in groups]
        return f'{min(values):.2f}% / {statistics.median(values):.2f}% / {max(values):.2f}%'
    added = attachment['bytes'] - 28800142336
    section = (
        '## Optional broad n-gram profile\n\n'
        f'This update adds a {attachment["profile_bytes"]:,}-byte row ranking as an embedded '
        f'resource and as `{PROFILE_NAME}`. The container grows by {added:,} bytes '
        f'({100 * added / 28800142336:.3f}%). Every pre-existing object passed exact byte readback; '
        'the IQ4 row bytes, quantization and row digest are unchanged. This profile selects cached '
        'rows; it does not change model weights or establish answer-quality improvements.\n\n'
        'The deterministic corpus uses four pinned [Common Corpus](https://huggingface.co/datasets/PleIAs/common_corpus) '
        'shards and one pinned [github-code-clean](https://huggingface.co/datasets/codeparrot/github-code-clean) shard. '
        'It covers prose, web, press, science and code without preferred languages. Equal per-group '
        'token ceilings limit dominance, and source documents or repositories separate training '
        'from held-out text. `profile-corpus.json` records revisions, sampling and limitations. '
        'Language labels are automatic; this finite sample is not a complete language inventory '
        'or an estimate of user traffic. Exact snippet deduplication does not eliminate all near-duplicates.\n\n'
        f"The profile contains {heldout['profile_rows']:,} ranked rows from {heldout['training_tokens']:,} training tokens. "
        f"The held-out set contains {heldout['total']['tokens']:,} tokens and {len(heldout['groups'])} "
        'domain/language groups. NumPy and the native C++ profiler produced byte-identical profiles.\n\n'
        '| RAM budget, MiB | held-out row hits |\n|---:|---:|\n' + table_rows + '\n\n'
        f'Across {len(groups)} groups with at least 2,048 held-out tokens, the minimum / median / maximum '
        f'hit rates are {spread(0)} at 256 MiB and {spread(4)} at 4 GiB. '
        'Japanese books have the lowest measured coverage: 6.04% and 20.54%, respectively. '
        'All groups, including sparse ones, remain in `profile-heldout.json`.\n\n'
        'On RTX 3090, disk residency, the embedded profile and the explicit profile produced identical '
        'tokens for two prompts, two repetitions and 32 output tokens, with MTP K=4 and int8 KV. '
        'Both profile modes served rows from the configured 256 MiB RAM cache. The Engine report '
        'retains startup time, every sample and memory counters. Cache state was uncontrolled and '
        'artifact transfers overlapped these checks, so this is functional evidence, not an isolated '
        'speedup. Caching remains opt-in; earlier cache tests showed 10–11% slower first requests.\n\n'
        'Public commit `64492cb14` accepts an explicit profile:\n\n'
        '```bash\n'
        f'hf download {REPO} {PROFILE_NAME} --local-dir models\n'
        '# Add these options to the serving command above:\n'
        '--ngram-residency ram-hot --ngram-hot-profile models/broad-v1.hot --ngram-ram-mib 256\n'
        '```\n\n'
        'Automatic loading of the embedded profile was qualified on the development snapshot '
        'recorded in `profile-engine.json`; that source change is not yet published. On the public '
        'commit, supply `--ngram-hot-profile`. Default disk operation does not use the profile.\n\n')
    card = card.replace('## Credits and license', section + '## Credits and license')
    heldout['native_profile_comparison'] = 'byte-identical to native C++ output'
    heldout['engine_qualification'] = 'See profile-engine.json'
    reports = {'profile-attachment.json': attachment, 'profile-engine.json': engine,
               'profile-heldout.json': heldout, 'profile-corpus.json': manifest}
    files = {name: (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode()
             for name, value in reports.items()}
    files['README.md'] = card.encode()
    sums = [s for s in original('SHA256SUMS').splitlines() if s.split()[-1].lstrip('*') != NAME]
    sums += [attachment['sha256'] + '  ' + NAME, attachment['profile_sha256'] + '  ' + PROFILE_NAME]
    sums += [hashlib.sha256(value).hexdigest() + '  ' + name for name, value in files.items() if name.endswith('.json')]
    files['SHA256SUMS'] = ('\n'.join(sums) + '\n').encode()
    for name, data in files.items():
        (JOB / name).write_bytes(data)
    api.batch_bucket_files(BUCKET, add=[(JOB / name, PREFIX + '/' + name) for name in reports])
    operations = [CommitOperationAdd(path_in_repo=NAME, path_or_fileobj=Path(attachment['path'])),
                  CommitOperationAdd(path_in_repo=PROFILE_NAME, path_or_fileobj=profile)]
    operations += [CommitOperationAdd(path_in_repo=name, path_or_fileobj=data) for name, data in files.items()]
    commit = api.create_commit(REPO, operations=operations, parent_commit=REV,
                               commit_message='feat: add measured broad n-gram row profile')
    after = api.model_info(REPO, revision=commit.oid, files_metadata=True)
    for name, size, sha in [(NAME, attachment['bytes'], attachment['sha256']),
                           (PROFILE_NAME, attachment['profile_bytes'], attachment['profile_sha256'])]:
        stored, = [f for f in after.siblings if f.rfilename == name]
        if after.private or stored.size != size or stored.lfs.sha256 != sha:
            raise RuntimeError('published profile or table metadata differs')
    for name, data in files.items():
        if Path(hf_hub_download(REPO, name, revision=commit.oid)).read_bytes() != data:
            raise RuntimeError('published metadata failed readback')
    receipt = {'repo': REPO, 'revision': commit.oid, 'table_sha256': attachment['sha256'],
               'profile_sha256': attachment['profile_sha256'], 'public': True,
               'metadata_readback_verified': True, 'private_profile_readback_verified': True}
    (JOB / 'published.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'stage': 'PROFILE_PUBLICATION_VERIFIED', **receipt}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'error_type': type(error).__name__,
            'reason': str(error) if type(error) is RuntimeError else 'diagnostic withheld'}), flush=True)
        raise SystemExit(1) from None
