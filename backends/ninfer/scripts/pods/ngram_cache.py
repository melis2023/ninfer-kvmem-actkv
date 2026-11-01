#!/usr/bin/env python3
"""Inspect or discard only the selected artifact's n-gram file pages on Linux."""
import argparse
import ctypes
import json
import mmap
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("artifact", type=Path)
parser.add_argument("--discard", action="store_true")
args = parser.parse_args()
libc = ctypes.CDLL(None, use_errno=True)
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
libc.mincore.restype = ctypes.c_int
page = mmap.PAGESIZE

with Artifact(args.artifact) as artifact:
    parts = binding_parts(artifact.directory.bindings["ngram/table"], artifact.by_id, "ngram/table")
    if len(parts) != 1 or parts[0][1] != 0:
        raise ValueError("expected one stored n-gram object")
    table = artifact.object(parts[0][0])
    first, last, cursor = table.offset, table.offset + table.bytes, 0
    for index, file in enumerate(artifact.directory.files):
        begin, end = max(first, cursor), min(last, cursor + file.payload_bytes)
        if begin < end:
            path = args.artifact if index == 0 else args.artifact.parent / file.path
            offset = (artifact.payload_offset if index == 0 else page) + begin - cursor
            aligned = offset // page * page
            length = end - begin + offset - aligned
            fd = os.open(path, os.O_RDONLY)
            try:
                if args.discard:
                    os.posix_fadvise(fd, offset, end - begin, os.POSIX_FADV_DONTNEED)
                with mmap.mmap(fd, length, offset=aligned, access=mmap.ACCESS_COPY) as mapped:
                    address = ctypes.addressof(ctypes.c_char.from_buffer(mapped))
                    count = (length + page - 1) // page
                    vector = (ctypes.c_ubyte * count)()
                    if libc.mincore(address, length, vector):
                        raise OSError(ctypes.get_errno(), "mincore")
                    resident = sum(value & 1 for value in vector)
                    print(json.dumps({"file": str(path), "table_bytes": end - begin,
                                      "pages": count, "resident_pages": resident,
                                      "discard_requested": args.discard}), flush=True)
                    if args.discard and resident > max(2, count // 1000):
                        raise RuntimeError("n-gram pages remained resident after the discard hint")
            finally:
                os.close(fd)
        cursor += file.payload_bytes
