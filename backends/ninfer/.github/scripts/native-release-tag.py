#!/usr/bin/env python3.11
"""Allocate a UTC date/build tag, reusing a release for the same source revision."""
import argparse
from datetime import datetime, timezone
import json
import re
from pathlib import Path


def flatten(pages):
    return [item for page in pages for item in page] if pages and isinstance(pages[0], list) else pages


def select(releases, refs, revision, day):
    same = [r for r in releases if r.get('target_commitish') == revision
            and re.fullmatch(r'(?:native-[0-9a-f]{12}|[0-9]{4}-[0-9]{2}-[0-9]{2}\.[1-9][0-9]*)', r['tag_name'])]
    if same:
        release = sorted(same, key=lambda r: (r['draft'], -r['id']))[0]
        return release['tag_name'], release['id'], release['draft']
    tags = [r['tag_name'] for r in releases] + [r['ref'].removeprefix('refs/tags/') for r in refs]
    numbers = [int(m[1]) for tag in tags if (m := re.fullmatch(re.escape(day) + r'\.([1-9][0-9]*)', tag))]
    return f'{day}.{max(numbers, default=0)+1}', 0, True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--releases', type=Path, required=True)
    parser.add_argument('--refs', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    day = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    tag, release_id, draft = select(flatten(json.loads(args.releases.read_text())),
                                    flatten(json.loads(args.refs.read_text())), args.revision, day)
    print(tag, release_id, str(draft).lower())
