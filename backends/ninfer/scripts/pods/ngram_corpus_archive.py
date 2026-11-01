#!/usr/bin/env python3
"""Archive the completed corpus in the existing private bucket and verify a readback."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile

from huggingface_hub import HfApi

ROOT = Path('/workspace/ninfer-work')
BUCKET = 'WaveCut/ninfer-cache'
PREFIX = 'ngram-profile-20261008/broad-v1'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if sys.platform != 'linux' or not ROOT.is_dir():
        raise RuntimeError('corpus transfer must run on the remote Pod')
    job = Path(os.environ['NINFER_JOB_DIR'])
    manifest = json.loads((ROOT / 'ngram-corpus/broad-v1/manifest.json').read_text())
    if (ROOT / 'jobs/corpus-build/exit').read_text().strip() != '0':
        raise RuntimeError('corpus construction did not complete')
    archive = job / 'broad-v1.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=1) as out:
        for name in ('ngram-corpus/broad-v1', 'ngram-corpus/tokenizer.json',
                     'jobs/corpus-build/inputs', 'jobs/corpus-build/source.json'):
            out.add(ROOT / name, arcname=name)
    sha = digest(archive)
    api = HfApi()
    if api.bucket_info(BUCKET).private is not True:
        raise RuntimeError('the corpus bucket is not private')
    remote = PREFIX + '/broad-v1.tar.gz'
    api.batch_bucket_files(BUCKET, add=[(archive, remote)])
    entry, = api.get_bucket_paths_info(BUCKET, [remote])
    copy = job / 'readback.tar.gz'
    api.download_bucket_files(BUCKET, files=[(remote, copy)], raise_on_missing_files=True)
    if copy.stat().st_size != archive.stat().st_size or digest(copy) != sha:
        raise RuntimeError('corpus archive readback differs')
    receipt = {'bucket': BUCKET, 'path': remote, 'bytes': entry.size,
               'sha256': sha, 'xet_hash': entry.xet_hash, 'readback_verified': True,
               'manifest': manifest}
    receipt_path = job / 'receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')
    api.batch_bucket_files(BUCKET, add=[(receipt_path, PREFIX + '/receipt.json')])
    copy.unlink()
    # The archive remains in the bucket. The collected job contains only its recipe and receipt.
    archive.unlink()
    print(json.dumps({'stage': 'CORPUS_ARCHIVED', 'bytes': entry.size, 'sha256': sha}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'error_type': type(error).__name__, 'errno': getattr(error, 'errno', None)}), flush=True)
        raise SystemExit(1) from None
