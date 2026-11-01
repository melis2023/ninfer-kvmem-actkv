#!/usr/bin/env python3
"""Check embedded and explicit broad profiles against uncached Engine output."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error

from mtp_qualify import request, PROMPTS

ROOT = Path('/workspace/ninfer-work')
JOB = Path(os.environ['NINFER_JOB_DIR'])


def main():
    attachment = json.loads((ROOT / 'ngram-profile/broad-v1/attachment.json').read_text())
    reference = json.loads((ROOT / 'jobs/mtp-qualify-q2/q2-host-k4-p0.json').read_text())
    expected, records = {}, []
    for mode in ('disk', 'embedded', 'explicit'):
        command = reference['command'].copy()
        command[command.index('--ngram-table') + 1] = attachment['path']
        if mode != 'disk':
            command += ['--ngram-residency', 'ram-hot']
            command[command.index('--ngram-ram-mib') + 1] = '256'
        if mode == 'explicit':
            command[command.index('--ngram-table') + 1] = str(ROOT / 'models/flash-next-iq4-table.ninfer')
            command += ['--ngram-hot-profile', str(ROOT / 'ngram-profile/broad-v1/broad-v1.hot')]
        record = {'mode': mode, 'command': command, 'samples': []}
        with (JOB / (mode + '.server.log')).open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                start = time.monotonic()
                while time.monotonic() - start < 300:
                    if process.poll() is not None:
                        raise RuntimeError('profile Engine exited during startup')
                    try:
                        request('/health')
                        break
                    except (urllib.error.URLError, TimeoutError):
                        time.sleep(1)
                else:
                    raise RuntimeError('profile Engine startup exceeded five minutes')
                record['startup_seconds'] = time.monotonic() - start
                for prompt, text in enumerate(PROMPTS):
                    for repeat in range(2):
                        before = request('/stats')
                        result = request('/completion', {'prompt': text, 'n_predict': 32,
                            'temperature': 0, 'ignore_eos': True, 'cache_prompt': False, 'return_tokens': True})
                        after = request('/stats')
                        expected.setdefault(prompt, result['tokens'])
                        if (result['tokens'] != expected[prompt] or result['tokens_predicted'] != 32 or
                                result['tokens_cached'] != 0 or result['timings'].get('draft_n', 0) <= 0):
                            raise RuntimeError('row residency changed tokens or disabled MTP')
                        resident = after['ngram_table']['resident_rows'] - before['ngram_table']['resident_rows']
                        if mode != 'disk' and resident <= 0:
                            raise RuntimeError('the hot profile served no resident rows')
                        if mode == 'disk' and resident != 0:
                            raise RuntimeError('the baseline unexpectedly used resident rows')
                        record['samples'].append({'prompt': prompt, 'repeat': repeat, 'result': result,
                            'ngram_before': before['ngram_table'], 'ngram_after': after['ngram_table']})
                record['process_status'] = Path(f'/proc/{process.pid}/status').read_text()
                record['status'] = 'passed'
            finally:
                (JOB / (mode + '.json')).write_text(json.dumps(record, indent=2) + '\n')
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        records.append(record)
        print(json.dumps({'stage': 'PROFILE_ENGINE_MODE_PASS', 'mode': mode}), flush=True)
    report = {'status': 'passed', 'attachment': attachment, 'modes': records,
        'hardware': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True),
        'source': json.loads((ROOT / 'source.json').read_text()),
        'limitations': ['OS page cache was uncontrolled', 'Concurrent artifact transfers prevent isolated performance attribution',
            'Two short prompts with 32 generated tokens and two repetitions per configuration',
            'The cache remains opt-in; this check establishes preserved output and actual row hits']}
    (JOB / 'qualification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PROFILE_ENGINE_PASS', flush=True)


if __name__ == '__main__':
    main()
