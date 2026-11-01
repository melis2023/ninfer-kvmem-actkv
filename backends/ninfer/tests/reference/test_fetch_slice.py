from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from unittest.mock import patch

from tools.reference import fetch_slice


def safetensors(tensors):
    header, payload = {}, bytearray()
    for name, data in tensors.items():
        header[name] = {"dtype": "U8", "shape": [len(data)],
                        "data_offsets": [len(payload), len(payload) + len(data)]}
        payload.extend(data)
    encoded = json.dumps(header).encode()
    return struct.pack("<Q", len(encoded)) + encoded + payload


class FetchSliceTest(unittest.TestCase):
    def test_parallel_shards_preserve_ranges_and_complete_manifest(self):
        payloads = {"mtp.a": b"first", "mtp.b": b"second"}
        weight_map = {name: f"part-{i}.safetensors" for i, name in enumerate(payloads)}
        files = {weight_map[name]: safetensors({name: value, "unselected": b"skip"})
                 for name, value in payloads.items()}
        rendezvous = threading.Barrier(2)

        def request(url, start=None, end=None):
            if url.endswith("config.json"):
                return b'{}'
            if url.endswith("index.json"):
                return json.dumps({"weight_map": weight_map}).encode()
            return files[url.rsplit("/", 1)[-1]][start:end]

        def chunks(url, start, end, chunk_bytes):
            rendezvous.wait(timeout=5)
            yield request(url, start, end)

        with (tempfile.TemporaryDirectory() as directory,
              patch.object(fetch_slice, "_request", request),
              patch.object(fetch_slice, "_range_chunks", chunks)):
            out = Path(directory)
            fetch_slice.fetch("owner/model", "a" * 40, out, [], ["mtp."], workers=2)
            manifest = json.loads((out / "fetch-manifest.json").read_text())
            index = json.loads((out / "model.safetensors.index.json").read_text())
            self.assertEqual(index["weight_map"], weight_map)
            for name, expected in payloads.items():
                raw = (out / weight_map[name]).read_bytes()
                size = struct.unpack("<Q", raw[:8])[0]
                self.assertEqual(raw[8 + size:], expected)
                self.assertEqual(manifest["files"][weight_map[name]]["sha256"],
                                 hashlib.sha256(raw).hexdigest())

    def test_mtp_subset_streams_exact_ranges_and_records_hashes(self):
        tensors = {"mtp.a": bytes(range(20)), "model.large": b"x" * 10000,
                   "mtp.b": bytes(range(16))}
        shard = safetensors(tensors)
        requests = []

        def request(url, start=None, end=None):
            if url.endswith("config.json"):
                return b'{"model_type":"qwen4_exp"}'
            if url.endswith("index.json"):
                return json.dumps({"weight_map": {name: "part.safetensors" for name in tensors}}).encode()
            requests.append((start, end))
            return shard[start:end]

        def chunks(url, start, end, chunk_bytes):
            for pos in range(start, end, chunk_bytes):
                yield request(url, pos, min(pos + chunk_bytes, end))

        with (tempfile.TemporaryDirectory() as directory,
              patch.object(fetch_slice, "_request", request),
              patch.object(fetch_slice, "_range_chunks", chunks)):
            out = Path(directory)
            names = fetch_slice.fetch("owner/model", "a" * 40, out, [], ["mtp."], chunk_bytes=5)
            self.assertEqual(names, ["mtp.a", "mtp.b"])
            result = (out / "part.safetensors").read_bytes()
            header_bytes = struct.unpack("<Q", result[:8])[0]
            header = json.loads(result[8:8 + header_bytes])
            payload = result[8 + header_bytes:]
            self.assertEqual(set(header), set(names))
            for name in names:
                begin, end = header[name]["data_offsets"]
                self.assertEqual(payload[begin:end], tensors[name])
            self.assertEqual(sum(end - start for start, end in requests[2:]), 36)
            self.assertTrue(all(end - start <= 5 for start, end in requests[2:]))
            manifest = json.loads((out / "fetch-manifest.json").read_text())
            self.assertEqual(manifest["revision"], "a" * 40)
            record = manifest["files"]["part.safetensors"]
            self.assertEqual(record["sha256"], hashlib.sha256(result).hexdigest())
            for name in names:
                self.assertEqual(record["tensors"][name]["sha256"],
                                 hashlib.sha256(tensors[name]).hexdigest())
            with self.assertRaises(FileExistsError):
                fetch_slice.fetch("owner/model", "a" * 40, out, [], ["mtp."])
            self.assertEqual((out / "part.safetensors").read_bytes(), result)

    def test_ignored_range_is_refused_before_reading_a_full_shard(self):
        response = io.BytesIO(b"a huge ignored full shard")
        response.status, response.headers = 200, {}
        with patch.object(fetch_slice.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "did not honor"):
                fetch_slice._request("https://huggingface.co/test", 10, 20)

    def test_truncated_range_is_refused(self):
        response = io.BytesIO(b"short")
        response.status, response.headers = 206, {"Content-Range": "bytes 10-19/30"}
        with patch.object(fetch_slice.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "expected 10 bytes, got 5"):
                fetch_slice._request("https://huggingface.co/test", 10, 20)

    def test_tensor_stream_uses_one_http_request_and_bounded_reads(self):
        response = io.BytesIO(b"0123456789")
        response.status, response.headers = 206, {"Content-Range": "bytes 100-109/200"}
        with patch.object(fetch_slice.urllib.request, "urlopen", return_value=response) as opened:
            chunks = list(fetch_slice._range_chunks("https://huggingface.co/test", 100, 110, 4))
            self.assertEqual(chunks, [b"0123", b"4567", b"89"])
            opened.assert_called_once()
            self.assertEqual(opened.call_args.args[0].get_header("Range"), "bytes=100-109")

    def test_failed_tensor_copy_does_not_publish_a_safetensors_file(self):
        shard = object.__new__(fetch_slice.Shard)
        shard.url, shard.data_start = "unused", 100
        shard.header = {"mtp.a": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]}}
        def failed_chunks(*args):
            yield b"1234"
            raise RuntimeError("truncated")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subset.safetensors"
            with patch.object(fetch_slice, "_range_chunks", failed_chunks):
                with self.assertRaisesRegex(RuntimeError, "truncated"):
                    fetch_slice._write_safetensors(path, shard, ["mtp.a"], 4)
            self.assertFalse(path.exists())
            self.assertTrue(path.with_suffix(".safetensors.partial").exists())


if __name__ == "__main__":
    unittest.main()
