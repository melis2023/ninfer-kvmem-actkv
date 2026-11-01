#!/usr/bin/env python3
"""Prepare or publish a qualified MTP replacement in its existing public repository."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from scripts.pods.mtp_release import BASES

ROOT = Path('/workspace/ninfer-work')
MINIMUM = '64492cb14d246fa04a5241c42e771dcfdff40753'
DONOR_REV = '766911a6b7369840a91dbcd95f9f997acaab6cd6'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=list(BASES), required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux' or not ROOT.is_dir():
        raise RuntimeError('model publication must run on the remote GPU Pod')
    variant, repo = args.variant, BASES[args.variant]
    candidate = json.loads((ROOT / 'mtp-candidates' / variant / 'download-verified.json').read_text())
    qualification = json.loads((args.qualification / 'qualification.json').read_text())
    media_job = ROOT / 'jobs' / ('mtp-vision-' + variant)
    media_qualification = json.loads((media_job / 'vision-qualification.json').read_text())
    if (candidate.get('artifact_sha256_verified') is not True or qualification['status'] != 'passed' or
            qualification['sha256'] != candidate['sha256'] or qualification['variant'] != variant):
        raise RuntimeError('this candidate has no matching successful GPU qualification')
    if media_qualification['status'] != 'passed' or media_qualification['sha256'] != candidate['sha256']:
        raise RuntimeError('this candidate has no matching Vision preservation qualification')
    media_records = [json.loads((media_job / (label + '.json')).read_text())
                     for label in media_qualification['checks']]
    if not media_records or any(r['status'] != 'passed' or
        r['vision'].get('timings', {}).get('draft_n', 0) != 0 or
        r['text_after_vision']['timings'].get('draft_n', 0) <= 0 for r in media_records):
        raise RuntimeError('the media fallback or subsequent text MTP check failed')
    audit = {r['repo']: r for r in json.loads(args.audit.read_text())}[repo]
    base, = audit['artifacts']
    model = Path(candidate['model_path'])
    if model.stat().st_size != candidate['bytes']:
        raise RuntimeError('candidate file size changed after qualification')
    api = HfApi()
    before = api.model_info(repo, files_metadata=True)
    if before.private or before.sha != audit['revision']:
        raise RuntimeError('the public repository changed since the pinned audit')
    output = Path(os.environ['NINFER_JOB_DIR']) / 'publication'
    output.mkdir(exist_ok=True)
    original = lambda name: Path(hf_hub_download(repo, name, revision=before.sha)).read_bytes()
    card = original('README.md').decode()
    card = card.replace('tags:\n', 'tags:\n- mtp\n', 1)
    card = card.replace('82f120dece4de7e411fc8815fb0d3cc34706fa3e', MINIMUM).replace('`82f120de`', '`64492cb14`')
    card, count = re.subn(r'MTP drafting is not available \(no GSQ-RCO release carries the MTP layer\)\.',
        'This file includes the MTP block. Enable drafting with `--spec mtp --draft-tokens 4`.', card)
    if count != 1:
        raise RuntimeError('the original MTP availability statement was not found exactly once')
    card, count = re.subn(r'of [\d,]+ bytes\n\([\d.]+ GiB\)',
        f"of {candidate['bytes']:,} bytes\n({candidate['bytes'] / 2**30:.2f} GiB)", card, count=1)
    if count != 1:
        raise RuntimeError('the original model size was not found')
    card = card.replace('Conversion\ncommand, from the repository\'s tree:',
                        'Original text/Vision conversion\ncommand, before the MTP attachment:')
    reports = []
    rows = []
    for label in qualification['checks']:
        record = json.loads((args.qualification / (label + '.json')).read_text())
        if record['status'] != 'passed' or len(record['samples']) != 6:
            raise RuntimeError('a required request set is incomplete')
        for prompt in (0, 1):
            values = [s['result']['timings']['predicted_per_second'] for s in record['samples']
                      if s['prompt'] == prompt]
            rows.append(f"| {prompt + 1} | {record['residency']} | {record['draft_tokens']} | {record['draft_min_p']} | "
                        f"{statistics.median(values):.2f} | {min(values):.2f}–{max(values):.2f} |")
        reports.append({'label': label, 'residency': record['residency'],
            'draft_tokens': record['draft_tokens'], 'draft_min_p': record['draft_min_p'],
            'startup_seconds': record['startup_seconds'], 'command': record['command'],
            'samples': record['samples'], 'vision': record.get('vision'),
            'process_status': record.get('process_status'), 'process_io': record.get('process_io')})
    section = (
        '## MTP attachment and qualification\n\n'
        f"The October 8, 2026 update adds 28 MTP objects ({2791415296:,} bytes) and 1,567 bindings. "
        'Every previous text weight, Vision weight, tokenizer resource and IQ4 table descriptor '
        'passed byte-preservation checks. The separate 28.80 GB IQ4 table is unchanged.\n\n'
        f'MTP comes from the [Unsloth shared-Q8_0 release](https://huggingface.co/unsloth/'
        f'Qwen3.8-Flash-Next-GGUF/tree/{DONOR_REV}), through the pinned donor recorded in '
        '`mtp-attachment.json`. The original text/Vision conversion report remains in the repository. '
        'The base quantization labels describe the text model, not the added MTP block.\n\n'
        'The public Engine passed host-to-GPU and disk-to-GPU MTP requests on one RTX 3090, '
        'with Vision enabled, int8 KV, 4,096 context capacity, 128-token prefill chunks and an '
        '8 GiB device expert cache. All expert arithmetic ran on the GPU. Each mode used two '
        'fixed prompts, three greedy repetitions and 64 generated tokens per request, with prefix '
        'reuse disabled. Each fixed mode repeated exactly and accepted MTP drafts. Image requests '
        'identified the red test image with thinking disabled. Flash-Next disables MTP for media '
        'requests and uses plain decoding. A fresh text request resumed MTP after each image check. '
        'An initial check incorrectly required MTP for media and failed. Source inspection '
        'confirmed this existing limitation; the engine behavior was not changed.\n\n'
        'Samples include the first request. Cache contents are not reset between repetitions. '
        'Other artifacts were being prepared on the same host during this campaign, so these '
        'timings do not isolate the performance effect of MTP.\n\n'
        'The measurements below cover only those short requests. The OS page cache was not '
        'controlled. They do not establish cold-disk speed, low-RAM operation or RTX 3080 support. '
        'Different draft widths or modes can produce different tokens because their arithmetic '
        'rounds differently. Prior quality results below were measured without this MTP attachment.\n\n'
        'Prompt 1 requests a Python merge function after a repeated prose prefix. Prompt 2 requests '
        'a simple explanation of the blue sky. Each row summarizes the three repetitions of that prompt.\n\n'
        '| prompt | expert placement | draft tokens | minimum draft probability | median decode tok/s | range tok/s |\n'
        '|---:|---|---:|---:|---:|---:|\n' + '\n'.join(rows) + '\n\n'
        'The probability-floor experiment uses the tested development snapshot. The normal MTP '
        'command above uses the default floor. `mtp-qualification.json` records every sample and '
        'the tested source snapshot; the minimum commit above identifies artifact support.\n\n')
    if card.count('## Quality') != 1:
        raise RuntimeError('the original quality section is not unambiguous')
    card = card.replace('## Quality', section + '## Quality')
    notice = original('NOTICE').decode().rstrip() + (
        '\n\nMTP attachment (October 8, 2026): shared-Q8_0 MTP weights from '
        f'unsloth/Qwen3.8-Flash-Next-GGUF, revision {DONOR_REV}. '
        'The Qwen Community License 1.0 continues to apply.\n')
    checksums = original('SHA256SUMS').decode().splitlines()
    checksums = [line for line in checksums if line.split() and line.split()[-1].lstrip('*') != base['file']]
    checksums.append(candidate['sha256'] + '  ' + base['file'])
    report = {'variant': variant, 'model_sha256': candidate['sha256'],
        'artifact_id': candidate['artifact_id'], 'status': 'passed',
        'hardware': qualification['hardware'], 'source': qualification['source'],
        'limitations': sorted(set(qualification['limitations'] + media_qualification['limitations'])),
        'configurations': reports, 'vision_preservation_checks': media_records}
    attachment = json.loads(Path(str(model) + '.conversion.json').read_text())
    attachment['file'] = base['file']
    attachment['gpu_qualification'] = 'See mtp-qualification.json for the qualified scope and source snapshot'
    files = {'README.md': card.encode(), 'NOTICE': notice.encode(),
        'SHA256SUMS': ('\n'.join(checksums) + '\n').encode(),
        'mtp-attachment.json': (json.dumps(attachment, indent=2) + '\n').encode(),
        'mtp-qualification.json': (json.dumps(report, indent=2) + '\n').encode()}
    for name in ('mtp-attachment.json', 'mtp-qualification.json'):
        checksums.append(hashlib.sha256(files[name]).hexdigest() + '  ' + name)
    files['SHA256SUMS'] = ('\n'.join(checksums) + '\n').encode()
    for name, data in files.items():
        (output / name).write_bytes(data)
    if not args.apply:
        print(json.dumps({'stage': 'PUBLICATION_PREPARED', 'repo': repo, 'path': str(output)}), flush=True)
        return
    operations = [CommitOperationAdd(path_in_repo=base['file'], path_or_fileobj=model)]
    operations += [CommitOperationAdd(path_in_repo=name, path_or_fileobj=data) for name, data in files.items()]
    commit = api.create_commit(repo, operations=operations, parent_commit=before.sha,
                               commit_message='feat: add qualified MTP to the existing artifact')
    after = api.model_info(repo, revision=commit.oid, files_metadata=True)
    stored, = [f for f in after.siblings if f.rfilename == base['file']]
    if after.private or stored.size != candidate['bytes'] or stored.lfs.sha256 != candidate['sha256']:
        raise RuntimeError('published artifact metadata differs from the qualified candidate')
    for name, data in files.items():
        if Path(hf_hub_download(repo, name, revision=commit.oid)).read_bytes() != data:
            raise RuntimeError('published metadata failed readback')
    receipt = {'repo': repo, 'revision': commit.oid, 'file': base['file'],
               'sha256': stored.lfs.sha256, 'bytes': stored.size, 'public': True,
               'metadata_readback_verified': True}
    (Path(os.environ['NINFER_JOB_DIR']) / 'published.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'stage': 'PUBLICATION_VERIFIED', **receipt}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'error_type': type(error).__name__, 'errno': getattr(error, 'errno', None),
            'reason': str(error) if type(error) is RuntimeError else 'diagnostic withheld'}), flush=True)
        raise SystemExit(1) from None
