#!/usr/bin/env python3.11
"""Remove older published native releases only after the new latest is verified."""
import argparse
import json
import re
import subprocess
from pathlib import Path


def native_tag(tag):
    return bool(re.fullmatch(r'(?:native-[0-9a-f]{12}|[0-9]{4}-[0-9]{2}-[0-9]{2}\.[1-9][0-9]*)', tag))


def prune(repository, tag, revision, api=None, delete=None, verified_assets=None):
    if api is None:
        def api(path):
            return json.loads(subprocess.check_output(['gh', 'api', f'repos/{repository}/{path}'], text=True))
    if delete is None:
        def delete(release_id):
            subprocess.run(['gh', 'api', '--method', 'DELETE', f'repos/{repository}/releases/{release_id}', '--silent'], check=True)
            result = subprocess.run(['gh', 'api', '--include', f'repos/{repository}/releases/{release_id}'], capture_output=True, text=True)
            if result.returncode == 0 or not (re.search(r'^HTTP/\S+ 404\b', result.stdout, re.M) or 'HTTP 404' in result.stderr):
                raise RuntimeError(f'Release deletion not confirmed: {release_id}')

    def current():
        latest = api('releases/latest')
        if (latest['tag_name'] != tag or latest['target_commitish'] != revision
                or latest['draft'] or latest['prerelease']):
            raise RuntimeError('Latest changed; old releases retained')
        if verified_assets is not None:
            def inventory(assets):
                return {a['name']: (a['size'], a['digest']) for a in assets}
            if inventory(latest['assets']) != inventory(verified_assets['assets']):
                raise RuntimeError('Latest assets changed; old releases retained')
        return latest

    latest = current()
    removed = []
    page = 1
    candidates = []
    while rows := api(f'releases?per_page=100&page={page}'):
        candidates.extend(rows)
        if len(rows) < 100:
            break
        page += 1
    for release in candidates:
        if (release['id'] == latest['id'] or release['draft'] or release['prerelease']
                or not native_tag(release['tag_name'])
                or release['published_at'] > latest['published_at']
                or not re.fullmatch(r'[0-9a-f]{40}', release['target_commitish'])):
            continue
        comparison = api(f"compare/{release['target_commitish']}...{revision}")
        if comparison['status'] not in ('ahead', 'identical'):
            continue
        current()
        delete(release['id'])
        removed.append(release['tag_name'])
        print(f"Removed older native release {release['tag_name']}; Git tag retained.", flush=True)
    return removed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--verified-assets', type=Path, required=True)
    args = parser.parse_args()
    prune(args.repository, args.tag, args.revision, verified_assets=json.loads(args.verified_assets.read_text()))
