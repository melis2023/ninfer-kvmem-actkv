#!/usr/bin/env python3.11
"""Combine two verified native archives without compiling or changing their backends."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

VARIANTS = {'12.9': ['70', '75', '80', '86', '89'],
            '13.4': ['75', '80', '86', '89', '120a']}
COMMANDS = ['ninfer', 'ninfer-serve', 'ninfer_bench', 'ninfer-calibrate',
            'ninfer-perplexity', 'ninfer-ngram-profile']


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Volumes:
    def __init__(self, paths):
        self.paths = iter(paths)
        self.stream = None

    def read(self, size):
        parts = []
        while size:
            if self.stream is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.stream = path.open('rb')
            chunk = self.stream.read(size)
            if chunk:
                parts.append(chunk)
                size -= len(chunk)
            else:
                self.stream.close()
                self.stream = None
        return b''.join(parts)

    def close(self):
        if self.stream is not None:
            self.stream.close()


def extract(source, destination, platform, cuda, revision):
    name = f"ninfer-{platform}-cuda{cuda}-sm{'-'.join(VARIANTS[cuda])}-{revision[:12]}"
    suffix = '7z' if platform == 'windows-x64' else 'tar.gz'
    manifest = source / f'SHA256SUMS-{name}.txt'
    volumes = []
    for index, line in enumerate(manifest.read_text().splitlines(), 1):
        expected = f'{name}.{suffix}.{index:03d}'
        match = re.fullmatch(r'([a-f0-9]{64})  (.+)', line)
        if not match or match[2] != expected:
            raise ValueError('Invalid archive volume order or name')
        path = source / expected
        if digest(path) != match[1]:
            raise ValueError(f'Archive checksum mismatch: {expected}')
        volumes.append(path)
    if not volumes:
        raise ValueError('Empty archive manifest')
    destination.mkdir(parents=True, exist_ok=False)
    if suffix == '7z':
        subprocess.run(['7z', 'x', str(volumes[0]), f'-o{destination}', '-y'], check=True)
    else:
        stream = Volumes(volumes)
        try:
            with tarfile.open(fileobj=stream, mode='r|gz') as archive:
                archive.extractall(destination, filter='data')
        finally:
            stream.close()
    if list(destination.iterdir()) != [destination / name]:
        raise ValueError('Unexpected archive root')
    root = destination / name
    validate(root, platform, cuda, revision)
    return root


def validate(root, platform, cuda, revision):
    info = json.loads((root / 'BUILD-INFO.json').read_text(encoding='utf-8-sig'))
    if (info.get('layout') != 'shared-runtime-v1' or info['source_revision'] != revision or
            info['modified_sources'] or info['platform'] != platform or info['cuda'] != cuda or
            info['architectures'] != VARIANTS[cuda] or info['commands'] != COMMANDS):
        raise ValueError('Backend metadata does not match this source, platform and CUDA variant')
    listed = set()
    for line in (root / 'SHA256SUMS.txt').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()) or relative in listed:
            raise ValueError('Invalid package checksum path')
        if digest(path) != expected:
            raise ValueError(f'Package checksum mismatch: {relative}')
        listed.add(relative)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if listed != actual - {'SHA256SUMS.txt'}:
        raise ValueError('Package checksum manifest does not cover all files')
    extension = '.exe' if platform == 'windows-x64' else ''
    runtime = 'ninfer-runtime.dll' if extension else 'libninfer-runtime.so'
    for relative in [*(c + extension for c in COMMANDS),
                     *(f'help/{c}.txt' for c in COMMANDS),
                     f"backends/cuda{cuda.split('.')[0]}/{runtime}",
                     f"backends/cuda{cuda.split('.')[0]}/architectures.txt"]:
        if not (root / relative).is_file():
            raise ValueError(f'Missing portable package input: {relative}')
    recorded = (root / f"backends/cuda{cuda.split('.')[0]}/architectures.txt").read_text().split()
    if recorded != VARIANTS[cuda]:
        raise ValueError('Backend architecture metadata differs from the package')
    return info


def assemble(roots, destination, platform, revision):
    if not re.fullmatch('[a-f0-9]{40}', revision):
        raise ValueError('Invalid source revision')
    info = {cuda: validate(roots[cuda], platform, cuda, revision) for cuda in VARIANTS}
    destination.mkdir(parents=True, exist_ok=False)
    # Launchers have no Toolkit dependency. Both variants implement the same private ABI.
    # Prefer the CUDA 13 copy, retaining each backend's own dependency closure.
    for path in roots['13.4'].iterdir():
        if path.name in ('SHA256SUMS.txt', 'BUILD-INFO.json', 'Dockerfile', 'packages'):
            continue
        shutil.move(str(path), destination / path.name)
    shutil.move(str(roots['12.9'] / 'backends/cuda12'), destination / 'backends/cuda12')
    # Preserve variant-specific redistribution notices without replacing the other copy.
    notices = destination / 'licenses/cuda12'
    notices.mkdir(parents=True)
    for path in roots['12.9'].glob('LICENSE*'):
        shutil.copy2(path, notices / path.name)
    if platform == 'linux-x64':
        packages = set()
        for root in roots.values():
            packages.update((root / 'packages/multi.txt').read_text().splitlines())
        (destination / 'packages').mkdir()
        (destination / 'packages/multi.txt').write_text(''.join(p + '\n' for p in sorted(packages)))
    (destination / 'BUILD-INFO.json').write_text(json.dumps({
        'layout': 'shared-runtime-v1', 'source_revision': revision, 'modified_sources': False,
        'platform': platform, 'commands': COMMANDS, 'backends': info,
        'validation': 'Assembled from verified native packages. GPU inference was not tested.'}, indent=2) + '\n')
    files = sorted(p for p in destination.rglob('*') if p.is_file())
    (destination / 'SHA256SUMS.txt').write_text(''.join(
        f'{digest(p)}  {p.relative_to(destination).as_posix()}\n' for p in files))


def archive(root, platform):
    suffix = '7z' if platform == 'windows-x64' else 'tar.gz'
    prefix = root.parent / f'{root.name}.{suffix}'
    if suffix == '7z':
        subprocess.run(['7z', 'a', '-t7z', '-mx=3', '-bsp1', '-v1900m', str(prefix), str(root)], check=True)
        subprocess.run(['7z', 't', '-bsp1', f'{prefix}.001'], check=True)
    else:
        process = subprocess.Popen(['tar', '-C', str(root.parent), '-czf', '-', root.name], stdout=subprocess.PIPE)
        try:
            index = 1
            remaining = 0
            stream = None
            try:
                while chunk := process.stdout.read(1024 * 1024):
                    if not remaining:
                        if stream:
                            stream.close()
                        stream = Path(f'{prefix}.{index:03d}').open('wb')
                        index += 1
                        remaining = 1900 * 1024**2
                    stream.write(chunk)
                    remaining -= len(chunk)
            finally:
                if stream:
                    stream.close()
            if process.wait():
                raise RuntimeError('Archive creation failed')
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()
    volumes = sorted(root.parent.glob(f'{prefix.name}.[0-9]*'))
    if suffix == 'tar.gz':
        stream = Volumes(volumes)
        try:
            with tarfile.open(fileobj=stream, mode='r|gz') as tar:
                for member in tar:
                    if member.isfile():
                        with tar.extractfile(member) as file:
                            while file.read(1024 * 1024):
                                pass
        finally:
            stream.close()
    if not volumes or any(not 0 < p.stat().st_size < 2 * 1024**3 for p in volumes):
        raise ValueError('Invalid archive volumes')
    (root.parent / f'SHA256SUMS-{root.name}.txt').write_text(''.join(
        f'{digest(p)}  {p.name}\n' for p in volumes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--platform', choices=['linux-x64', 'windows-x64'], required=True)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[a-f0-9]{40}', args.revision):
        parser.error('revision must be a full commit SHA')
    roots = {}
    for cuda in VARIANTS:
        print(f'Checking and extracting CUDA {cuda}', flush=True)
        roots[cuda] = extract(args.input, args.output / f'input-{cuda}', args.platform, cuda, args.revision)
    name = f'ninfer-{args.platform}-cuda12.9-13.4-{args.revision[:12]}'
    root = args.output / name
    assemble(roots, root, args.platform, args.revision)
    for source in roots.values():
        shutil.rmtree(source.parent)
    print(f'Archiving {name}', flush=True)
    archive(root, args.platform)


if __name__ == '__main__':
    main()
