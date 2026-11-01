#!/usr/bin/env python3.11
"""Describe verified native packages using their manifests and actual volume sizes."""
import argparse
import json
from pathlib import Path
import re


BACKENDS = (
    ('cuda12.9-13.4', 'Dual CUDA 12.9 + 13.4', 'Automatic backend selection for the GPU and driver.'),
    ('cuda12.9-sm70-75-80-86-89', 'CUDA 12.9', 'Volta/V100, or a driver that supports CUDA 12 but not CUDA 13.'),
    ('cuda13.4-sm75-80-86-89-120a', 'CUDA 13.4', 'Blackwell/RTX 50, or a CUDA 13 installation on the supported GPUs.'),
)


def size_text(size):
    return f'{size / 1024**3:.2f} GiB ({size:,} bytes)'


def load_packages(dist, revision, remote_assets=None):
    sizes = None
    if remote_assets is not None:
        sizes = {asset['name']: asset['size'] for asset in remote_assets['assets']}
    packages = []
    for platform, label in (('windows', 'Windows x64'), ('linux', 'Debian 13 x64')):
        extension = '7z' if platform == 'windows' else 'tar.gz'
        for backend, title, purpose in BACKENDS:
            stem = f'ninfer-{platform}-x64-{backend}-{revision[:12]}'
            manifest = f'SHA256SUMS-{stem}.txt'
            lines = (dist / manifest).read_text().splitlines()
            if not lines:
                raise ValueError(f'Empty manifest: {manifest}')
            volumes = []
            for index, line in enumerate(lines, 1):
                filename = f'{stem}.{extension}.{index:03d}'
                if not re.fullmatch(r'[0-9a-f]{64}  ' + re.escape(filename), line):
                    raise ValueError(f'Invalid volume entry in {manifest}: {line}')
                size = sizes[filename] if sizes is not None else (dist / filename).stat().st_size
                if not isinstance(size, int) or not 0 < size < 2 * 1024**3:
                    raise ValueError(f'Invalid asset size: {filename}')
                volumes.append((filename, size))
            packages.append(dict(platform=platform, label=label, title=title, purpose=purpose,
                                 stem=stem, manifest=manifest, volumes=volumes,
                                 size=sum(size for _, size in volumes)))
    return packages


def render_notes(repository, revision, packages, tag):
    source = f'https://github.com/{repository}/blob/{revision}'
    base = f'https://github.com/{repository}/releases/download/{tag}'

    def link(filename):
        return f'[{filename}]({base}/{filename})'

    lines = [
        f'Native builds from [{revision}](https://github.com/{repository}/commit/{revision}).',
        '', '## What to download', '',
        '- Choose **one package for your operating system**. Download every numbered part of that package and its SHA-256 manifest.',
        '- **Default choice: Dual CUDA for your operating system.** It includes both backends and selects CUDA 12 for Volta or CUDA 13 for Blackwell. On overlapping GPU targets, it prefers CUDA 13 when the driver supports it. Use `--cuda-backend 12` or `--cuda-backend 13` to require a backend.',
        '- **Single CUDA** includes one backend and has a smaller download. Choose CUDA 12.9 for Volta/V100 or a CUDA 12 driver environment. Blackwell/RTX 50 requires the CUDA 13.4 backend and a compatible driver.',
        '- CUDA runtime libraries are included. Install a compatible NVIDIA driver; no CUDA Toolkit is needed. Model weights are downloaded separately.',
        '', '## Downloads', '',
        'Sizes are compressed download totals: the sum of all parts, excluding the checksum manifest.',
        '', '| Platform | Package | Parts | Total download | Use |',
        '| --- | --- | --- | --- | --- |',
    ]
    for p in packages:
        count = len(p['volumes'])
        parts = f'{count} part' + ('s' if count != 1 else '')
        lines.append(f"| {p['label']} | {p['title']} | {parts} | {size_text(p['size'])} | {p['purpose']} |")
    lines += ['', f"All six archive alternatives total {size_text(sum(p['size'] for p in packages))}.",
              '', '### Files for each package', '',
              'Keep all parts of the selected package in one directory. Verify them against its SHA-256 manifest before extraction.', '']
    for p in packages:
        count = len(p['volumes'])
        parts = f'{count} part' + ('s' if count != 1 else '')
        lines += ['<details>', f"<summary>{p['label']} — {p['title']} ({parts}, {size_text(p['size'])})</summary>",
                  '', '| Archive part | Size |', '| --- | --- |']
        lines += [f'| {link(filename)} | {size_text(size)} |' for filename, size in p['volumes']]
        lines += ['', f"Checksums: {link(p['manifest'])}.", '']
        if p['platform'] == 'windows':
            lines += ['Extract the first part with 7-Zip. It reads the remaining parts automatically.',
                      '', '```powershell', f"7z x {p['volumes'][0][0]}", '```']
        else:
            parts = ' '.join(filename for filename, _ in p['volumes'])
            lines += ['Concatenate the parts in this order and extract the archive.', '', '```bash',
                      f"sha256sum -c {p['manifest']}", f'cat {parts} | tar -xz', '```']
        lines += ['', '</details>', '']
    lines += [
        '## Requirements and package contents', '',
        f'- **Windows x64:** install the Microsoft Visual C++ x64 runtime and a compatible NVIDIA driver. Runtime DLLs are included. [Windows instructions]({source}/docs/build-windows.md).',
        f'- **Linux x64:** built on Debian 13 with glibc 2.41. Install the system packages listed in `packages/multi.txt` after extraction. Other distributions also need the recorded library SONAMEs; newer glibc alone is insufficient. [Linux instructions]({source}/docs/release-archive-linux.md).',
        '- Each package includes six command launchers, its CUDA backend libraries, saved help, launch scripts, licences, `BUILD-INFO.json` and internal `SHA256SUMS.txt`. Keep `backends/` and `help/` beside the launchers.',
        '- Single-CUDA Linux packages include a Dockerfile that packages these binaries without compilation. The dual Linux package has no Dockerfile.',
        '', '| Backend | CUDA Toolkit used to build | GPU device images |',
        '| --- | --- | --- |',
        '| CUDA 12 | 12.9.2 | `sm_70`, `sm_75`, `sm_80`, `sm_86`, `sm_89` |',
        '| CUDA 13 | 13.4.3 | `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_120a` |',
        '', '## Validation', '',
        'CI checks the requested device images, dependency closure, command startup (`--help`), archive contents and uploaded asset sizes and SHA-256 digests. These jobs do not run GPU inference or performance tests. `BUILD-INFO.json` records the source revision, compiler, CUDA version and architectures.',
    ]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist', type=Path, required=True)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tag', required=True, help='Published release tag used in download links.')
    parser.add_argument('--assets-json', type=Path, help='Use GitHub release asset metadata for a preview without archive downloads.')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repository):
        parser.error('Invalid GitHub repository')
    if not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        parser.error('Invalid source revision')
    if not re.fullmatch(r'(?:native-[0-9a-f]{12}|[0-9]{4}-[0-9]{2}-[0-9]{2}\.[1-9][0-9]*)', args.tag):
        parser.error('Invalid native release tag')
    metadata = json.loads(args.assets_json.read_text()) if args.assets_json else None
    packages = load_packages(args.dist, args.revision, metadata)
    args.output.write_text(render_notes(args.repository, args.revision, packages, args.tag))


if __name__ == '__main__':
    main()
