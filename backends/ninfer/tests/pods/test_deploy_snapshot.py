import io
import os
import tarfile
import time

from scripts.pods.deploy_snapshot import install


def snapshot(root, files):
    with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            member.mtime = 10
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(content))


def test_changed_content_arrives_newer_than_an_object_built_after_the_local_edit(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    old = source / "test.cpp"
    old.write_bytes(b"old")
    os.utime(old, (20, 20))
    object_mtime = time.time_ns() - 1_000_000_000
    snapshot(tmp_path, {"test.cpp": b"new"})
    install(tmp_path)
    assert old.read_bytes() == b"new"
    assert old.stat().st_mtime_ns > object_mtime


def test_unchanged_sources_keep_their_timestamp_and_only_removed_managed_files_are_deleted(tmp_path):
    snapshot(tmp_path, {"keep.cpp": b"same", "removed.cpp": b"old"})
    install(tmp_path)
    source = tmp_path / "src"
    kept = source / "keep.cpp"
    timestamp = kept.stat().st_mtime_ns
    (source / "unmanaged.txt").write_text("retained")
    snapshot(tmp_path, {"keep.cpp": b"same"})
    install(tmp_path)
    assert kept.stat().st_mtime_ns == timestamp
    assert not (source / "removed.cpp").exists()
    assert (source / "unmanaged.txt").read_text() == "retained"
