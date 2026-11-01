#!/usr/bin/env python3
"""Install changed snapshot files with fresh mtimes so Ninja cannot reuse stale objects."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import tarfile


def install(root: Path):
    source = root / "src"
    source.mkdir(exist_ok=True)
    manifest = root / "files"
    previous = set(manifest.read_text().splitlines()) if manifest.exists() else set()
    names, changed = set(), []
    with tarfile.open(root / "source.tar.gz", "r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.issym()):
                raise ValueError("snapshot contains an unsupported path or file type")
            names.add(member.name)
            destination = source / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.parent.resolve().is_relative_to(source.resolve()):
                raise ValueError("snapshot parent points outside the source directory")
            temporary = destination.with_name(destination.name + f".snapshot-{os.getpid()}")
            if member.issym():
                if destination.is_symlink() and os.readlink(destination) == member.linkname:
                    continue
                temporary.symlink_to(member.linkname)
            else:
                with archive.extractfile(member) as stream:
                    data = stream.read()
                if (not destination.is_symlink() and destination.is_file() and
                        destination.read_bytes() == data):
                    if destination.stat().st_mode & 0o7777 != member.mode:
                        destination.chmod(member.mode)
                    continue
                temporary.write_bytes(data)
                temporary.chmod(member.mode)
            # Do not restore the editor's mtime: it can predate an object built from an older
            # remote snapshot while the local edit was being made.
            os.replace(temporary, destination)
            changed.append(member.name)
    removed = []
    for name in sorted(previous - names):
        destination = source / name
        if destination.is_file() or destination.is_symlink():
            destination.unlink()
            removed.append(name)
    manifest.write_text("".join(name + "\n" for name in sorted(names)))
    print(json.dumps({"snapshot_changed": changed, "snapshot_removed": removed}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    install(parser.parse_args().root)
