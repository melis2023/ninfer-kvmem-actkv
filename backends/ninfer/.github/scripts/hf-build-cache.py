"""Restore and checkpoint compiler/dependency caches in the private HF bucket."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import tempfile
import time
import threading

BUCKET = 'WaveCut/ninfer-cache'
ROOT = 'github-actions/v1/iamwavecut/ninfer-all'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prefix(family, ref):
    if not re.fullmatch(r'[a-zA-Z0-9_.-]+', family):
        raise ValueError('Invalid cache family')
    return f'{ROOT}/{hashlib.sha256(ref.encode()).hexdigest()[:20]}/{family}'


def unpack(archive, directory):
    with tarfile.open(archive) as package:
        for item in package:
            path = PurePosixPath(item.name)
            if path.is_absolute() or '..' in path.parts or '\\' in item.name or ':' in item.name or \
                    not (item.isdir() or item.isfile()):
                raise ValueError('Invalid cache archive member')
        package.extractall(directory, filter='data')


def pack(directory, archive):
    count = 0
    with tarfile.open(archive, 'w') as package:
        for path in sorted(directory.rglob('*')):
            relative = path.relative_to(directory)
            if path.is_symlink() or 'tmp' in relative.parts or path.name in ('stats', 'lock') or \
                    path.name.endswith(('.lock', '.tmp')):
                continue
            try:
                with path.open('rb') as stream:
                    # Read metadata from the open inode: ccache replaces entries atomically.
                    item = package.gettarinfo(fileobj=stream, arcname=relative.as_posix())
                    if not item.isfile():
                        continue
                    package.addfile(item, stream)
                    count += 1
            except (FileNotFoundError, IsADirectoryError, PermissionError):
                continue
    return count


@contextmanager
def writer_lock(directory):
    path = directory.parent / (directory.name + '.hf-lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.write(b'0')
        stream.flush()
        limit = time.monotonic() + 180
        while True:
            try:
                if os.name == 'nt':
                    import msvcrt
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= limit:
                    raise TimeoutError('Another cache checkpoint is still running')
                time.sleep(.2)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def manifests(api, base, work):
    entries = api.get_bucket_paths_info(BUCKET, [f'{base}/{slot}.json' for slot in (0, 1)])
    found = []
    for entry in entries:
        local = work / (Path(entry.path).name)
        api.download_bucket_files(BUCKET, files=[(entry.path, local)], raise_on_missing_files=True)
        item = json.loads(local.read_text())
        slot = int(Path(entry.path).stem)
        if item['schema'] != 1 or item['archive'] != f'{base}/{slot}.tar' or \
                not re.fullmatch(r'[0-9a-f]{64}', item['sha256']):
            raise ValueError('Invalid cache manifest')
        found.append(item | {'slot': slot})
    return sorted(found, key=lambda item: item['created_ns'], reverse=True)


def save(api, directory, family, ref, verify=False):
    if not directory.is_dir():
        raise ValueError('Cache directory is missing')
    base = prefix(family, ref)
    with writer_lock(directory), tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
        work = Path(temporary)
        previous = manifests(api, base, work)
        run_id = os.environ.get('GITHUB_RUN_ID', 'local')
        if run_id.isdecimal() and any(str(item.get('run_id', '')).isdecimal() and
                                      int(item['run_id']) > int(run_id) for item in previous):
            print('A newer native run already saved this family; skipping stale writer.', flush=True)
            return False
        slot = 1 - previous[0]['slot'] if previous else 0
        archive = work / 'cache.tar'
        count = pack(directory, archive)
        if not count:
            raise ValueError('Refusing to replace a checkpoint with an empty cache')
        item = {'schema': 1, 'family': family, 'ref': ref,
                'archive': f'{base}/{slot}.tar', 'sha256': digest(archive),
                'bytes': archive.stat().st_size, 'files': count,
                'created_ns': time.time_ns(), 'run_id': os.environ.get('GITHUB_RUN_ID', 'local')}
        print(json.dumps({'state': 'uploading', 'family': family, 'bytes': item['bytes'], 'files': count}), flush=True)
        api.batch_bucket_files(BUCKET, add=[(archive, item['archive'])])
        remote, = api.get_bucket_paths_info(BUCKET, [item['archive']])
        if remote.size != item['bytes']:
            raise ValueError('Uploaded cache size differs')
        if verify:
            check = work / 'check.tar'
            api.download_bucket_files(BUCKET, files=[(item['archive'], check)], raise_on_missing_files=True)
            if digest(check) != item['sha256']:
                raise ValueError('Uploaded cache checksum differs')
        # Publish metadata only after the archive is available. Readers verify the hash and
        # fall back to the other slot if this upload or a later replacement is interrupted.
        metadata = work / 'manifest.json'
        metadata.write_text(json.dumps(item) + '\n')
        api.batch_bucket_files(BUCKET, add=[(metadata, f'{base}/{slot}.json')])
        print(json.dumps({'state': 'saved', **item}), flush=True)


def restore(api, directory, family, ref, legacy_prefix=None, required=False, seed_ref=None,
            fallback_families=()):
    if seed_ref and not seed_ref.startswith('refs/heads/'):
        raise ValueError('The seed cache must belong to a branch')
    directory.mkdir(parents=True, exist_ok=True)
    had_snapshots = False
    with tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
        work = Path(temporary)
        refs = list(dict.fromkeys([ref, 'refs/heads/master'] + ([seed_ref] if seed_ref else [])))
        for source_ref in refs:
            choices = manifests(api, prefix(family, source_ref), work)
            had_snapshots = had_snapshots or bool(choices)
            for item in choices:
                archive = work / 'cache.tar'
                try:
                    api.download_bucket_files(BUCKET, files=[(item['archive'], archive)], raise_on_missing_files=True)
                    if archive.stat().st_size != item['bytes'] or digest(archive) != item['sha256']:
                        raise ValueError('Cache checksum differs')
                    unpack(archive, directory)
                except Exception as error:
                    print(json.dumps({'state':'checkpoint-rejected', 'error':type(error).__name__}), flush=True)
                    continue
                print(json.dumps({'state':'restored', **item}), flush=True)
                return True
        if legacy_prefix:
            entries = [entry for entry in api.list_bucket_tree(BUCKET, prefix='github-actions/migration', recursive=True)
                       if entry.path.startswith('github-actions/migration/' + legacy_prefix)
                       and entry.path.endswith('.tar.json')]
            restored = 0
            for index, entry in enumerate(sorted(entries, key=lambda e: e.path)):
                local = work / f'seed-{index}.json'
                api.download_bucket_files(BUCKET, files=[(entry.path, local)], raise_on_missing_files=True)
                item = json.loads(local.read_text())
                if item['source_ref'] not in refs:
                    continue
                if item['archive'] + '.json' != entry.path or not re.fullmatch(r'[0-9a-f]{64}', item['sha256']):
                    raise ValueError('Invalid migration manifest')
                archive = work / 'seed.tar'
                api.download_bucket_files(BUCKET, files=[(item['archive'], archive)], raise_on_missing_files=True)
                if digest(archive) != item['sha256']:
                    raise ValueError('Migration cache checksum differs')
                unpack(archive, directory)
                restored += 1
                print(json.dumps({'state':'migration-restored', 'key':item['key'], 'files':item['files']}), flush=True)
            if restored:
                return True
    if had_snapshots:
        raise ValueError('No verified cache checkpoint exists')
    # Read previous families without replacing their snapshots. Compiler input hashes
    # still decide which restored entries are valid under the current toolchain.
    for fallback in dict.fromkeys(fallback_families):
        if fallback == family:
            raise ValueError('A cache family cannot fall back to itself')
        if restore(api, directory, fallback, ref, seed_ref=seed_ref):
            return True
    if required:
        raise ValueError('No verified cache checkpoint exists')
    print(json.dumps({'state':'cold', 'family':family}), flush=True)
    return False


def require_native_writer(family, ref):
    """Reject accidental writes from CI, PRs and retired recovery workflows."""
    workflow = os.environ.get('GITHUB_WORKFLOW_REF', '').split('@')[0]
    bootstrap = workflow == 'iamwavecut/ninfer-all/.github/workflows/linux-cache-bootstrap.yml'
    if ((workflow != 'iamwavecut/ninfer-all/.github/workflows/native-release.yml' and not bootstrap) or
            os.environ.get('GITHUB_EVENT_NAME') not in ('push', 'workflow_dispatch') or
            not ref.startswith('refs/heads/') or
            not os.environ.get('GITHUB_RUN_ID', '').isdecimal()):
        raise ValueError('Only a native artifact build may save shared caches')
    job = os.environ.get('GITHUB_JOB')
    if bootstrap and (job != 'linux' or not family.startswith('native-linux-cuda-') or
                      not family.endswith('-shared-v1')):
        raise ValueError('Bootstrap may write only the shared Linux artifact cache')
    if not ((job == 'linux' and family.startswith('native-linux-cuda-')) or
            (job == 'windows' and family.startswith(('native-windows-cuda-', 'vcpkg-')))):
        raise ValueError('Cache family does not belong to this native job')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('save', 'restore'))
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--family', required=True)
    parser.add_argument('--legacy-prefix')
    parser.add_argument('--fallback-family', action='append', default=[])
    parser.add_argument('--source-ref', help='Reader branch; defaults to the current branch')
    parser.add_argument('--optional', action='store_true', help='Continue without an unavailable reader cache')
    parser.add_argument('--required', action='store_true')
    parser.add_argument('--verify-upload', action='store_true')
    args = parser.parse_args()
    if args.action == 'save' and (args.optional or args.source_ref or args.fallback_family):
        parser.error('Reader options cannot be used for saves')
    if args.optional and args.required:
        parser.error('--optional and --required are mutually exclusive')
    args.directory = args.directory.expanduser()
    ref = args.source_ref or os.environ['GITHUB_REF']
    if args.action == 'save':
        require_native_writer(args.family, ref)
    try:
        if not os.environ.get('HF_TOKEN'):
            raise ValueError('HF cache credential is not configured')
        from huggingface_hub import HfApi
        api = HfApi()
        if api.bucket_info(BUCKET).private is not True:
            raise ValueError('Build cache bucket must be private')
        if args.action == 'save':
            save(api, args.directory, args.family, ref, args.verify_upload)
        else:
            restore(api, args.directory, args.family, ref, args.legacy_prefix, args.required,
                    os.environ.get('HF_BUILD_CACHE_SEED_REF'), args.fallback_family)
    except Exception as error:
        if not args.optional:
            raise
        # CI can compile without the cache. Do not print transport errors with signed URLs.
        print(f'::warning::Native cache unavailable ({type(error).__name__}); continuing with local compilation.', flush=True)


def heartbeat(done):
    while not done.wait(60):
        print('HF cache transfer or verification is still running.', flush=True)


if __name__ == '__main__':
    done = threading.Event()
    threading.Thread(target=heartbeat, args=(done,), daemon=True).start()
    try:
        main()
    except Exception as error:
        # Transport exceptions can contain signed URLs. Keep them out of public CI logs.
        print(f'::error::HF cache operation failed ({type(error).__name__})', flush=True)
        sys.exit(1)
    finally:
        done.set()
