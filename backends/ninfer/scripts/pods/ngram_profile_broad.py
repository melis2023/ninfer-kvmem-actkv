#!/usr/bin/env python3
"""Build the broad row-frequency profile and report every held-out source group."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tarfile

from huggingface_hub import HfApi
import numpy as np
from tokenizers import Tokenizer

ROOT = Path('/workspace/ninfer-work')
JOB = Path(os.environ['NINFER_JOB_DIR'])


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows_for(ids, config, eos):
    """Evaluate the stored three-token XOR hash, cutting history at EOS."""
    ids = np.asarray(ids, dtype=np.uint64)
    previous = np.full((3, len(ids)), eos, dtype=np.uint64)
    previous[0] = ids
    if len(ids) > 1:
        previous[1, 1:] = ids[:-1]
    if len(ids) > 2:
        previous[2, 2:] = ids[:-2]
    previous[2, previous[1] == eos] = eos
    values = previous * np.asarray(config['multipliers'], dtype=np.uint64)[:, None]
    two = values[0] ^ values[1]
    three = two ^ values[2]
    mixed = np.concatenate((np.repeat(two[:, None], 8, axis=1),
                            np.repeat(three[:, None], 8, axis=1)), axis=1)
    return (mixed % np.asarray(config['head_vocab'], dtype=np.uint64) +
            np.asarray(config['head_offset'], dtype=np.uint64)).reshape(-1)


def fingerprint(config):
    values = [config['ngram_size'], config['heads_per_ngram']]
    for key in ('multipliers', 'head_vocab', 'head_offset'):
        values.extend([len(config[key]), *config[key]])
    values.append(config['rows'])
    value = 0xcbf29ce484222325
    for byte in struct.pack(f'<{len(values)}Q', *values):
        value = ((value ^ byte) * 0x100000001b3) & 0xffffffffffffffff
    return value


def main():
    if sys.platform != 'linux' or not ROOT.is_dir():
        raise RuntimeError('corpus work must run on the remote Pod')
    receipt = json.loads((JOB / 'inputs/archive-receipt.json').read_text())
    api = HfApi()
    if api.bucket_info(receipt['bucket']).private is not True:
        raise RuntimeError('corpus bucket is not private')
    archive = ROOT / 'broad-corpus.tar.gz'
    api.download_bucket_files(receipt['bucket'], files=[(receipt['path'], archive)],
                              raise_on_missing_files=True)
    if sha(archive) != receipt['sha256']:
        raise RuntimeError('corpus archive differs from the verified CPU output')
    with tarfile.open(archive) as package:
        for member in package:
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or not (member.isdir() or member.isfile()):
                raise RuntimeError('unexpected corpus archive member')
            if path.parts[:2] == ('jobs', 'corpus-build'):
                member.name = str(Path('ngram-profile/corpus-recipe').joinpath(*path.parts[2:]))
            elif path.parts[0] != 'ngram-corpus':
                raise RuntimeError('unexpected corpus archive root')
            package.extract(member, ROOT, filter='data')
    archive.unlink()
    corpus = ROOT / 'ngram-corpus/broad-v1'
    manifest = json.loads((corpus / 'manifest.json').read_text())
    metadata = json.loads((JOB / 'inputs/conversion.json').read_text())
    tokenizer_path = ROOT / 'ngram-corpus/tokenizer.json'
    if sha(tokenizer_path) != metadata['object_sha256']['resource/text/tokenizer.json']:
        raise RuntimeError('corpus tokenizer differs from the unchanged model tokenizer')
    config = metadata['components']['ngram']['config']
    eos = metadata['components']['text']['config']['eos_token_id']
    if config['ngram_size'] != 3 or config['heads_per_ngram'] != 8:
        raise RuntimeError('this recipe requires the published three-token, sixteen-head hash')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    output = ROOT / 'ngram-profile/broad-v1'
    output.mkdir(parents=True, exist_ok=True)
    counts = np.zeros(config['rows'], dtype=np.uint32)
    train_tokens = 0
    with (corpus / 'train.jsonl').open() as stream:
        for line in stream:
            doc = json.loads(line)
            ids = tokenizer.encode(doc['text'], add_special_tokens=False).ids
            if len(ids) != doc['tokens']:
                raise RuntimeError('training token count differs from the corpus receipt')
            train_tokens += len(ids)
            np.add.at(counts, rows_for(ids, config, eos), 1)
    if train_tokens != manifest['tokens']['train']:
        raise RuntimeError('training corpus token total differs')
    seen = np.flatnonzero(counts)
    order = np.lexsort((seen, -counts[seen].astype(np.int64)))
    ranked = seen[order].astype('<u4')
    del counts, seen, order
    profile = output / 'broad-v1.hot'
    with profile.open('wb') as stream:
        stream.write(b'NFNGHOT1' + struct.pack('<4Q', config['rows'], fingerprint(config),
                                             train_tokens, len(ranked)))
        stream.write(ranked.tobytes())
    print(json.dumps({'stage': 'PROFILE_BUILT', 'tokens': train_tokens,
                      'profile_rows': len(ranked), 'bytes': profile.stat().st_size}), flush=True)
    rank = np.full(config['rows'], np.iinfo(np.uint32).max, dtype=np.uint32)
    rank[ranked] = np.arange(len(ranked), dtype=np.uint32)
    index_bytes = (config['rows'] + 63) // 64 * 8 + (config['rows'] + 511) // 512 * 4
    budgets = [mib << 20 for mib in (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)]
    capacity = np.asarray([min((size - index_bytes) // 90, len(ranked)) for size in budgets])
    def empty():
        return {'documents': 0, 'tokens': 0, 'reads': 0, 'hits': np.zeros(len(budgets), dtype=np.uint64)}
    groups = defaultdict(empty)
    total = empty()
    with (corpus / 'heldout.jsonl').open() as stream:
        for line in stream:
            doc = json.loads(line)
            ids = tokenizer.encode(doc['text'], add_special_tokens=False).ids
            if len(ids) != doc['tokens']:
                raise RuntimeError('held-out token count differs from the corpus receipt')
            rows = rows_for(ids, config, eos)
            hits = (rank[rows, None] < capacity[None, :]).sum(axis=0, dtype=np.uint64)
            for record in (total, groups[(doc['domain'], doc['language'])]):
                record['documents'] += 1
                record['tokens'] += len(ids)
                record['reads'] += len(rows)
                record['hits'] += hits
    if total['tokens'] != manifest['tokens']['heldout']:
        raise RuntimeError('held-out corpus token total differs')
    def serialize(record):
        return {**record, 'hits': record['hits'].tolist(),
                'hit_percent': (100 * record['hits'] / max(1, record['reads'])).tolist()}
    report = {'profile_sha256': sha(profile), 'profile_bytes': profile.stat().st_size,
              'profile_rows': len(ranked), 'training_tokens': train_tokens,
              'budgets_bytes': budgets, 'stored_row_bytes': 90, 'index_bytes': index_bytes,
              'total': serialize(total), 'groups': [dict(domain=k[0], language_label=k[1], **serialize(v))
                                                   for k, v in sorted(groups.items())],
              'limitations': manifest['limitations'], 'native_profile_comparison': 'pending',
              'engine_qualification': 'pending'}
    (output / 'heldout.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    (JOB / 'heldout.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'stage': 'PROFILE_HELDOUT_COMPLETE', 'total': serialize(total),
                      'groups': len(groups)}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'error_type': type(error).__name__, 'errno': getattr(error, 'errno', None),
            'reason': str(error) if type(error) is RuntimeError else 'diagnostic withheld'}), flush=True)
        raise SystemExit(1) from None
