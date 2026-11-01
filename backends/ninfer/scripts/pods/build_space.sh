#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/ninfer-work
# Failed nvcc invocations can leave multi-GiB intermediates. Restrict cleanup to this finished
# build's time interval and CUDA's temporary-file prefix, with no compiler still running.
if [ -f "$root/jobs/build/exit" ] && [ "$(cat "$root/jobs/build/exit")" != 0 ]; then
    if ! pgrep -x 'nvcc|cicc|ptxas|fatbinary' >/dev/null; then
        "$root/py311/bin/python" - "$root/jobs/build" <<'PY'
from pathlib import Path
import sys
job = Path(sys.argv[1])
first, last = (job / "started").stat().st_mtime, (job / "finished").stat().st_mtime
count = size = 0
for path in Path("/tmp").glob("tmpxft_*"):
    if path.is_file() and not path.is_symlink():
        info = path.stat()
        if first <= info.st_mtime <= last:
            size += info.st_size
            count += 1
            path.unlink()
print(f"Removed {count} failed-build CUDA temporary files ({size} bytes)")
PY
    fi
fi
# These are download caches of this rental, never model artifacts or the upload resume state.
"$root/py311/bin/python" -c 'import shutil; from pathlib import Path; [shutil.rmtree(p, ignore_errors=True) for p in (Path("/root/.cache/uv"), Path("/root/.cache/huggingface/xet"))]'
# Preserve the finished A16 comparison binary in compressed form. It is not used by the active
# n-gram checks; gzip's CRC validates the archive before the uncompressed file is removed.
baseline="$root/baselines/native-a16-vector-20261008/ninfer_benches"
if [ -f "$baseline" ] && [ ! -f "$baseline.gz" ]; then
    temporary=$(mktemp /dev/shm/ninfer-baseline-XXXXXX.gz)
    trap 'rm -f "$temporary"' EXIT
    gzip -1 -c "$baseline" > "$temporary"
    gzip -t "$temporary"
    cp "$temporary" "$baseline.gz"
    cmp "$temporary" "$baseline.gz"
    rm "$baseline"
fi
# The baseline predates the running build; its qualification job records the correct snapshot.
cp "$root/jobs/native-a16-checks/source.json" \
    "$root/baselines/ngram-pre-cache-20261008/source.json"
df -h /workspace /dev/shm
