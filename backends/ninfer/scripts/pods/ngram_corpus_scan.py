#!/usr/bin/env python3
"""Inspect bounded, pinned corpus shards on the CPU Pod without exporting their texts."""
from collections import Counter
import json
import os
from pathlib import Path
import sys

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

SOURCES = (
    ("PleIAs/common_corpus", "307910e4c5d040d6f318e6edf2a2b97849155771",
     "common_corpus_1/subset_100_1.parquet"),
    ("codeparrot/github-code-clean", "c48d40f9e70f0196f8236901ee35807f7d6c44c0",
     "data/train-00000-of-00880.parquet"),
)


def main():
    root = Path("/workspace/ninfer-work/ngram-corpus")
    if sys.platform != "linux" or not root.parent.is_dir():
        raise RuntimeError("corpus downloads must run on the CPU Pod")
    job = Path(os.environ["NINFER_JOB_DIR"])
    reports = []
    for repo, revision, filename in SOURCES:
        print(json.dumps({"stage": "download_corpus_shard", "repo": repo, "file": filename}), flush=True)
        path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                                   local_dir=root / repo.split("/")[0]))
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        report = {"repo": repo, "revision": revision, "file": filename, "local_path": str(path),
                  "bytes": path.stat().st_size, "rows": parquet.metadata.num_rows,
                  "fields": names, "schema": str(parquet.schema_arrow)}
        counters = {name: Counter() for name in names if name.lower() in
                    ("language", "lang", "collection", "open type", "open_type", "license")}
        columns = list(counters)
        for batch in parquet.iter_batches(batch_size=512, columns=columns):
            for row in batch.to_pylist():
                for name in counters:
                    counters[name][str(row[name])] += 1
        report["groups"] = {name: dict(counts.most_common()) for name, counts in counters.items()}
        # Metadata examples identify grouping keys. Do not put document text in job output.
        examples = next(parquet.iter_batches(batch_size=4,
            columns=[n for n in names if n.lower() not in ("text", "content", "code")])).to_pylist()
        report["metadata_examples"] = examples
        reports.append(report)
        (job / "corpus-scan.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"stage": "corpus_scanned", "repo": repo, "rows": report["rows"],
                          "fields": names, "groups": report["groups"]}), flush=True)


if __name__ == "__main__":
    main()
