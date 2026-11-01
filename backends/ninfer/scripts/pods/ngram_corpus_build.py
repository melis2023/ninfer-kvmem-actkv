#!/usr/bin/env python3
"""Build a bounded corpus across observed language and domain groups on the CPU Pod."""
import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from scripts.pods.mtp_release import RemoteDonor

ROOT = Path("/workspace/ninfer-work/ngram-corpus")
COMMON = "PleIAs/common_corpus"
COMMON_REV = "307910e4c5d040d6f318e6edf2a2b97849155771"
CODE = "codeparrot/github-code-clean"
CODE_REV = "c48d40f9e70f0196f8236901ee35807f7d6c44c0"
BASE = "WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3"
SEED = "ninfer-ngram-broad-20261008"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def group(row, code):
    language = str(row.get("language") or "Unknown").strip()
    language = {"GO": "Go", "Portugueuse": "Portuguese", "Spanish; Castilian": "Spanish"}.get(language, language)
    if code or row.get("open_type") == "Open Source":
        domain = "code"
    elif row.get("language_type") == "Spoken":
        domain = "speech"
    elif row.get("open_type") == "Open Culture":
        domain = "press" if re.search(r"news|zeitung|anno", str(row.get("collection")), re.I) else "books-and-culture"
    else:
        domain = {"Open Government": "government-and-law", "Open Science": "science",
                  "Open Web": "web-and-discussion", "Semantic data": "structured-text"}.get(
                      row.get("open_type"), "other")
    return domain, language


def identities(row, code):
    if code:
        origin = row["repo_name"]
        return origin + "/" + row["path"], "repo:" + origin, "repo:" + origin
    identity = str(row.get("identifier") or row.get("title") or "")
    creator = str(row.get("creator") or "").strip()
    url = urlparse(identity)
    if row.get("open_type") == "Open Source" and url.netloc.endswith("github.com"):
        repo = "/".join(url.path.strip("/").split("/")[:2])
        return identity, "repo:" + repo, "repo:" + repo
    # A book/document is never split across construction and evaluation.
    unit = "document:" + identity
    if creator and creator.lower() not in ("none", "unknown", "null"):
        authority = urlparse(creator).netloc or creator
    else:
        authority = url.netloc or identity
    origin = str(row.get("collection")) + ":" + authority
    return identity, unit, origin


def load_tokenizer(audit):
    catalog = {r["repo"]: r for r in json.loads(audit.read_text())}
    record, = catalog[BASE]["artifacts"]
    model = RemoteDonor(catalog[BASE], record, ROOT / "tokenizer-cache")
    object_id = model.directory.components["text"]["resources"]["tokenizer.json"]
    data = b"".join(model.iter_object(object_id))
    path = ROOT / "tokenizer.json"
    path.write_bytes(data)
    return Tokenizer.from_file(str(path)), hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux" or not ROOT.parent.is_dir():
        raise RuntimeError("corpus assembly must run on the CPU Pod")
    output = ROOT / "broad-v1"
    if output.exists():
        raise RuntimeError("corpus output already exists; inspect before rebuilding")
    output.mkdir(parents=True)
    job = Path(os.environ["NINFER_JOB_DIR"])
    tokenizer, tokenizer_sha = load_tokenizer(args.audit)
    sources = [(COMMON, COMMON_REV, f"common_corpus_{part}/subset_100_1.parquet", False)
               for part in (1, 3, 5, 9)]
    sources += [(CODE, CODE_REV, "data/train-00000-of-00880.parquet", True)]
    pools = defaultdict(list)
    observed = Counter()
    paths = []
    # Retain at most 96 deterministic metadata candidates per group and partition.
    for source_index, (repo, revision, name, code) in enumerate(sources):
        print(json.dumps({"stage": "corpus_metadata", "repo": repo, "file": name}), flush=True)
        path = Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision,
                                   local_dir=ROOT / repo.split("/")[0]))
        paths.append(path)
        parquet = pq.ParquetFile(path)
        columns = [n for n in parquet.schema_arrow.names if n not in ("text", "code")]
        row_index = 0
        for batch in parquet.iter_batches(batch_size=256, columns=columns):
            for row in batch.to_pylist():
                index = row_index
                row_index += 1
                if code and re.search(r"(^|/)(node_modules|vendor|third_party)/|\.min\.(js|css)$", row["path"]):
                    continue
                domain, language = group(row, code)
                if not code and domain == "code" and not urlparse(str(row.get("identifier"))).netloc.endswith("github.com"):
                    # Use repository-identified code so train/evaluation cannot share its files.
                    continue
                identity, unit, origin = identities(row, code)
                if not identity or language in ("Unknown", "Multilingual", "Ignore List", "Raw token data"):
                    continue
                partition = "heldout" if int(digest(SEED + unit)[:8], 16) % 5 == 0 else "train"
                key = domain, language, partition
                observed[key] += 1
                score = int(digest(SEED + repo + identity), 16)
                item = (-score, source_index, index, identity, unit, origin, row.get("collection", "GitHub"),
                        row.get("license"), row.get("title", row.get("path")))
                heap = pools[key]
                if len(heap) < 96:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
    selected = defaultdict(dict)
    for key, heap in pools.items():
        for item in sorted(heap, reverse=True):
            selected[item[1]][item[2]] = (key, item)
    candidates = defaultdict(list)
    # Read texts in bounded Arrow batches. Keep only the selected snippets.
    for source_index, path in enumerate(paths):
        code = sources[source_index][3]
        column = "code" if code else "text"
        row_index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=[column]):
            for row in batch.to_pylist():
                picked = selected[source_index].get(row_index)
                row_index += 1
                if picked is None:
                    continue
                key, item = picked
                text = row[column] or ""
                if len(text) < 128 or text.count("\ufffd") > len(text) // 100:
                    continue
                # Select a contiguous interior segment, so book prefaces do not dominate.
                if len(text) > 24000:
                    start = int(digest(item[3])[:8], 16) % (len(text) - 24000)
                    text = text[start:start + 24000]
                    newline = text.find("\n")
                    if 0 <= newline < 1000:
                        text = text[newline + 1:]
                ids = tokenizer.encode(text, add_special_tokens=False).ids[:1024]
                if len(ids) < 64:
                    continue
                snippet = tokenizer.decode(ids, skip_special_tokens=False)
                # The C++ profiler tokenizes this exact string again.
                tokens = len(tokenizer.encode(snippet, add_special_tokens=False).ids)
                candidates[key].append((item, snippet, tokens, digest(" ".join(snippet.split()))))
        print(json.dumps({"stage": "corpus_text_selected", "source": source_index}), flush=True)
    records = []
    seen = set()
    units = {"train": set(), "heldout": set()}
    partitions = {p: (output / (p + ".jsonl")).open("w") for p in ("train", "heldout")}
    group_counts = {}
    try:
        for key, values in sorted(candidates.items()):
            domain, language, partition = key
            slug = digest(domain + "/" + language)[:12]
            group_file = output / f"{partition}-{slug}.jsonl"
            origin_count, collection_count = Counter(), Counter()
            tokens = documents = 0
            quota = 16384 if partition == "train" else 4096
            with group_file.open("w") as group_output:
                for item, text, size, text_sha in sorted(values, key=lambda x: x[0], reverse=True):
                    if text_sha in seen or origin_count[item[5]] >= 4:
                        continue
                    if tokens >= quota:
                        break
                    seen.add(text_sha)
                    units[partition].add(item[4])
                    origin_count[item[5]] += 1
                    collection_count[item[6]] += 1
                    source = sources[item[1]]
                    doc = {"text": text, "domain": domain, "language": language,
                           "source": {"repo": source[0], "revision": source[1], "file": source[2],
                                      "row": item[2], "identifier": item[3], "split_unit": item[4],
                                      "origin": item[5], "collection": item[6], "license": item[7]},
                           "tokens": size, "text_sha256": text_sha}
                    line = json.dumps(doc, ensure_ascii=False) + "\n"
                    group_output.write(line)
                    partitions[partition].write(line)
                    tokens += size
                    documents += 1
            entry = {"domain": domain, "language": language, "partition": partition,
                     "file": group_file.name, "tokens": tokens, "documents": documents,
                     "origins": len(origin_count), "collections": dict(collection_count),
                     "observed_documents": observed[key], "target_tokens": quota}
            records.append(entry)
            group_counts[key] = documents
    finally:
        for stream in partitions.values():
            stream.close()
    if units["train"] & units["heldout"]:
        raise RuntimeError("a source document or repository entered both partitions")
    manifest = {"seed": SEED, "tokenizer_sha256": tokenizer_sha, "sources": sources,
                "groups": records, "partition_unit_overlap": 0,
                "tokens": {p: sum(r["tokens"] for r in records if r["partition"] == p)
                           for p in partitions},
                "limitations": ["Coverage sample from five pinned shards, not all available text",
                    "Source language and domain labels are not manually verified",
                    "Identical normalized snippets are removed; near-duplicate detection is not exhaustive",
                    "Sparse groups remain visible and do not establish language-wide coverage",
                    "Equal per-group token ceilings are a declared sampling choice, not observed user traffic"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (job / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"stage": "CORPUS_BUILT", "groups": len(records), "tokens": manifest["tokens"],
                      "path": str(output)}), flush=True)


if __name__ == "__main__":
    main()
