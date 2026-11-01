#!/usr/bin/env python3
"""Append the measured hot-row profile while preserving every IQ4 table byte."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec
from tools.artifact.writer import ArtifactWriter
from attach_mtp import spec, verify_copy
from ngram_profile_broad import fingerprint

ROOT = Path('/workspace/ninfer-work')
JOB = Path(os.environ['NINFER_JOB_DIR'])
PROFILE = ROOT / 'ngram-profile/broad-v1/broad-v1.hot'
OUTPUT = ROOT / 'ngram-profile/broad-v1/flash-next-iq4-profiled.ninfer'


class ProfileSource:
    def iter_object(self, identity):
        with PROFILE.open('rb') as stream:
            while data := stream.read(8 << 20):
                yield data


def main():
    if (ROOT / 'jobs/broad-profile-native/exit').read_text().strip() != '0':
        raise RuntimeError('the native profiler comparison did not pass')
    heldout = json.loads((ROOT / 'ngram-profile/broad-v1/heldout.json').read_text())
    profile_data = PROFILE.read_bytes()
    if hashlib.sha256(profile_data).hexdigest() != heldout['profile_sha256']:
        raise RuntimeError('the evaluated profile changed')
    with Artifact(ROOT / 'models/flash-next-iq4-table.ninfer') as base:
        config = base.directory.components['ngram']['config']
        if profile_data[:8] != b'NFNGHOT1':
            raise RuntimeError('invalid profile header')
        rows, signature, tokens, count = struct.unpack_from('<4Q', profile_data, 8)
        if (rows != config['rows'] or signature != fingerprint(config) or
                tokens != heldout['training_tokens'] or len(profile_data) != 40 + count * 4):
            raise RuntimeError('profile geometry or provenance differs from the table')
        ids = np.frombuffer(profile_data, dtype='<u4', offset=40)
        if (len(ids) and int(ids.max()) >= rows) or len(np.unique(ids)) != count:
            raise RuntimeError('profile contains duplicate or out-of-range rows')
        components = deepcopy(base.directory.components)
        resource = 'resource/ngram/hot_profile'
        if components['ngram'].get('resources', {}).get('hot_profile'):
            raise RuntimeError('the source table already has a profile')
        components['ngram'].setdefault('resources', {})['hot_profile'] = resource
        specs = [spec(obj, obj.id) for obj in base.objects] + [ResourceSpec(resource, len(profile_data))]
        with ArtifactWriter(OUTPUT, specs, components=components, bindings=base.directory.bindings,
                uses=base.directory.uses, metadata=base.directory.metadata,
                provenance={**base.directory.provenance, 'hot_profile': {
                    'parent_artifact_id': base.artifact_id.hex(), 'sha256': heldout['profile_sha256'],
                    'train_tokens': tokens, 'corpus_bucket': 'WaveCut/ninfer-cache',
                    'corpus_path': 'ngram-profile-20261008/broad-v1/broad-v1.tar.gz'}}) as writer:
            for obj in base.objects:
                writer.write_object(obj.id, base.iter_object(obj.id))
            writer.write_object(resource, [profile_data])
        with Artifact(OUTPUT) as result:
            sources = {obj.id: (base, obj.id) for obj in base.objects}
            sources[resource] = (ProfileSource(), resource)
            hashes, sha = verify_copy(result, OUTPUT, sources)
            if result.directory.components != components or result.directory.bindings != base.directory.bindings:
                raise RuntimeError('the table metadata changed during attachment')
            receipt = {'path': str(OUTPUT), 'bytes': OUTPUT.stat().st_size, 'sha256': sha,
                'artifact_id': result.artifact_id.hex(), 'profile_sha256': heldout['profile_sha256'],
                'profile_bytes': len(profile_data), 'table_sha256': config['table_sha256'],
                'base_objects_verified': len(base.objects), 'object_sha256': hashes}
    (JOB / 'attachment.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (OUTPUT.parent / 'attachment.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'stage': 'PROFILE_ATTACHMENT_VERIFIED', **receipt}), flush=True)


if __name__ == '__main__':
    main()
