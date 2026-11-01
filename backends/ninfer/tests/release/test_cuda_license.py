"""Check licence extraction and preservation when downloaded content is invalid."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SCRIPT = Path(__file__).resolve().parents[2] / '.github/scripts/fetch-cuda-license.py'
spec = importlib.util.spec_from_file_location('cuda_license', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CudaLicenseTest(unittest.TestCase):
    def test_verified_archive_extracts_exact_licence(self):
        self.exercise(b'licence fixture', False, True)

    def test_checksum_failure_preserves_existing_file(self):
        self.exercise(b'licence fixture', True, False)

    def test_empty_licence_preserves_existing_file(self):
        self.exercise(b'', False, False)

    def exercise(self, licence, corrupt, success):
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w') as archive:
            archive.writestr('cuda_cudart-archive/LICENSE', licence)
        data = data.getvalue()
        metadata = {'cuda_cudart': {'windows-x86_64': {
            'relative_path': 'cuda_cudart/windows-x86_64/runtime.zip',
            'sha256': '0' * 64 if corrupt else hashlib.sha256(data).hexdigest()}}}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'EULA.txt'
            output.write_bytes(b'existing licence')
            with patch.object(module.urllib.request, 'urlopen', side_effect=[
                    io.BytesIO(json.dumps(metadata).encode()), io.BytesIO(data)]):
                if success:
                    module.fetch('13.4.3', output)
                    self.assertEqual(output.read_bytes(), licence)
                else:
                    with self.assertRaises(ValueError):
                        module.fetch('13.4.3', output)
                    self.assertEqual(output.read_bytes(), b'existing licence')


if __name__ == '__main__':
    unittest.main()
