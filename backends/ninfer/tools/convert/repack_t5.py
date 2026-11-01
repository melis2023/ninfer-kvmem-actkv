"""Repack a native PTQ1 artifact to T5, preserving other objects byte for byte.

Run: python -m tools.convert.repack_t5 INPUT.ninfer OUTPUT.ninfer
The output is a separate artifact; the input is never modified.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from ..artifact.reader import Artifact
from ..artifact.writer import ArtifactWriter
from ..artifact.schema import TensorObject, TensorSpec, ResourceSpec
from ..artifact.layouts import row_split_geometry
from ..artifact.tensor_output import TensorOutput
from ..artifact.codecs.ptq1 import unpack as unpack_ptq1
from ..artifact.codecs.t5 import pack, unpack


def repack(source: Path, destination: Path) -> dict:
    with Artifact(source) as original:
        targets = {o.id for o in original.objects
                   if isinstance(o, TensorObject) and o.format == 'ptq1_g128_fp16'}
        if not targets:
            raise ValueError('input has no native PTQ1 objects')
        specs = [TensorSpec(o.id, o.shape, 't5_g128_fp16' if o.id in targets else o.format,
                            o.layout, o.divisors) if isinstance(o, TensorObject)
                 else ResourceSpec(o.id, o.bytes, o.encoding) for o in original.objects]
        directory = original.directory
        uses = deepcopy(list(directory.uses))
        for use in uses:
            binding = directory.bindings[use['parameter']]
            parts = binding.get('parts', [binding])
            objects = {part['object'] for part in parts}
            if objects & targets:
                if not objects <= targets:
                    raise ValueError('one parameter mixes PTQ1 and another representation')
                use['activation_policy'] = 'AllowA8'
        report = {'source': str(source.resolve()), 'destination': str(destination.resolve()),
                  'source_artifact_id': original.artifact_id.hex(), 'repacked': [], 'copied': []}
        provenance = deepcopy(directory.provenance)
        provenance['t5_repack'] = {'source_artifact_id': original.artifact_id.hex(),
            'kernel_source': 'JGamboa/ninfer-4090-windows@e905787f067de89ef6ef28c52d65a2ce91d425ca',
            'weights': 'exact PTQ1 codes and FP16 scale bits; A8 execution permission'}
        with ArtifactWriter(destination, specs, components=directory.components,
                            bindings=directory.bindings, uses=uses, metadata=directory.metadata,
                            provenance=provenance) as writer:
            for obj in original.objects:
                if obj.id not in targets:
                    digest = hashlib.sha256()
                    def chunks():
                        for chunk in original.iter_object(obj.id):
                            digest.update(chunk)
                            yield chunk
                    writer.write_object(obj.id, chunks())
                    report['copied'].append({'id': obj.id, 'bytes': obj.bytes, 'sha256': digest.hexdigest()})
                    continue
                n, k = obj.shape
                geometry = row_split_geometry(obj.format, obj.shape)
                output = TensorOutput(writer, obj.id)
                logical_hash = hashlib.sha256()
                scale_hash = hashlib.sha256()
                # Bound temporary tensors even for the vocabulary head.
                rows_per_chunk = max(1, (4 << 20) // k)
                for begin in range(0, n, rows_per_chunk):
                    rows = min(rows_per_chunk, n - begin)
                    def plane(offset, pitch):
                        return torch.frombuffer(bytearray(original.read_range(
                            obj.offset + offset + begin * pitch, rows * pitch)), dtype=torch.uint8)
                    low = plane(geometry.base_offset, geometry.base_row_bytes)
                    high = plane(geometry.high_offset, geometry.high_row_bytes)
                    scales = plane(geometry.scale_offset, geometry.scale_row_bytes).view(torch.float16)
                    codes = unpack_ptq1(low, high)
                    if not torch.equal(unpack(pack(codes)), codes):
                        raise ValueError(f'{obj.id}: exact T5 round trip failed')
                    logical_hash.update(codes.numpy().tobytes())
                    scale_hash.update(scales.view(torch.uint8).numpy().tobytes())
                    output.write_codes(begin, codes.reshape(rows, k//128, 128),
                                       scales.reshape(rows, k//128))
                report['repacked'].append({'id': obj.id, 'shape': list(obj.shape),
                    'codes_sha256': logical_hash.hexdigest(), 'scale_sha256': scale_hash.hexdigest()})
                print(f'T5 {obj.id} {n}x{k}', flush=True)
        # Re-read the persisted artifact, including unchanged MTP, norms and resources.
        with Artifact(destination) as converted:
            for item in report['copied']:
                digest = hashlib.sha256()
                for chunk in converted.iter_object(item['id']): digest.update(chunk)
                if digest.hexdigest() != item['sha256']:
                    raise ValueError(f"{item['id']}: unchanged object mismatch")
            for item in report['repacked']:
                obj = converted.object(item['id'])
                n,k = obj.shape
                geometry = row_split_geometry(obj.format, obj.shape)
                code_hash, scale_hash = hashlib.sha256(), hashlib.sha256()
                for begin in range(0,n,max(1,(4<<20)//k)):
                    rows=min(max(1,(4<<20)//k),n-begin)
                    raw=converted.read_range(obj.offset+begin*geometry.base_row_bytes, rows*geometry.base_row_bytes)
                    codes=unpack(torch.frombuffer(bytearray(raw),dtype=torch.uint8))
                    code_hash.update(codes.numpy().tobytes())
                    scale_hash.update(converted.read_range(obj.offset+geometry.scale_offset+begin*geometry.scale_row_bytes,rows*geometry.scale_row_bytes))
                if (code_hash.hexdigest()!=item['codes_sha256'] or scale_hash.hexdigest()!=item['scale_sha256']):
                    raise ValueError(f"{item['id']}: persisted codes/scales mismatch")
        report['verified'] = True
        destination.with_suffix(destination.suffix+'.repack.json').write_text(
            json.dumps(report, indent=2), encoding='utf-8')
        return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('destination',type=Path)
    args=p.parse_args()
    torch.set_num_threads(4)
    report=repack(args.source,args.destination)
    print(f"Verified {len(report['repacked'])} T5 objects and {len(report['copied'])} copied objects")


if __name__=='__main__':
    main()
