#!/usr/bin/env bash
# Publish an immutable source snapshot only after all Windows and Linux packages pass their checks.
set -euo pipefail
: "${GITHUB_REPOSITORY:?}" "${GITHUB_SHA:?}" "${GITHUB_REF:?}"
[[ "$GITHUB_REF" == refs/heads/master ]] || { echo 'Publication requires master' >&2; exit 1; }
[[ "$GITHUB_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid source revision' >&2; exit 1; }
repo="$GITHUB_REPOSITORY"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd dist
shopt -s nullglob
checksums=(SHA256SUMS-ninfer-*.txt)
[[ ${#checksums[@]} == 6 ]] || { echo 'Expected all six single-CUDA and universal checksum manifests' >&2; exit 1; }
# Validate the exact expected variants and this source revision, not merely four arbitrary files.
for variant in {windows,linux}-x64-cuda12.9-sm70-75-80-86-89 {windows,linux}-x64-cuda13.4-sm75-80-86-89-120a {windows,linux}-x64-cuda12.9-13.4; do
  manifests=(SHA256SUMS-ninfer-"$variant-${GITHUB_SHA:0:12}.txt")
  [[ ${#manifests[@]} == 1 && -f "${manifests[0]}" ]] || { echo "Missing package: $variant" >&2; exit 1; }
done
assets=(ninfer-windows-*.7z.* ninfer-linux-*.tar.gz.*)
[[ ${#assets[@]} -ge 4 ]] || { echo 'Missing archive volumes' >&2; exit 1; }
# Refuse extra, unverified volumes and manifests with paths outside this directory.
python3.11 - "${checksums[@]}" <<'PY'
import hashlib
import pathlib
import re
import sys
listed = set()
for filename in sys.argv[1:]:
    lines = pathlib.Path(filename).read_text().splitlines()
    if not lines:
        raise SystemExit(f'Empty manifest: {filename}')
    stem = filename.removeprefix('SHA256SUMS-').removesuffix('.txt')
    extension = '7z' if stem.startswith('ninfer-windows-') else 'tar.gz'
    for index, line in enumerate(lines, 1):
        match = re.fullmatch(r'([0-9a-f]{64})  (ninfer-[A-Za-z0-9._-]+\.(?:7z|tar\.gz)\.[0-9]{3,})', line)
        if not match or match[2] != f'{stem}.{extension}.{index:03d}':
            raise SystemExit(f'Invalid volume entry: {line}')
        path = pathlib.Path(match[2])
        if not 0 < path.stat().st_size < 2 * 1024**3:
            raise SystemExit(f'Invalid asset size: {path}')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != match[1]:
                raise SystemExit(f'Checksum mismatch: {path}')
        listed.add(path.name)
if listed != {p.name for pattern in ('*.7z.*', '*.tar.gz.*') for p in pathlib.Path('.').glob(pattern)}:
    raise SystemExit('Archive volumes differ from the checksum manifests')
PY
head="$(gh api "repos/$repo/commits/master" --jq .sha)"
if [[ "$head" != "$GITHUB_SHA" ]]; then
  echo 'Source was superseded on master; packages remain in Actions artifacts.'
  exit 0
fi
# Resolve retries by full source revision; reserve daily numbers across releases and Git tags.
gh api --paginate --slurp "repos/$repo/releases?per_page=100" > releases.json
gh api --paginate --slurp "repos/$repo/git/matching-refs/tags/" > release-refs.json
read -r tag release_id draft < <(python3.11 "$script_dir/native-release-tag.py" \
  --releases releases.json --refs release-refs.json --revision "$GITHUB_SHA")
published=false
if [[ "$release_id" != 0 ]]; then
  [[ "$(gh api "repos/$repo/releases/$release_id" --jq .target_commitish)" == "$GITHUB_SHA" ]] || {
    echo 'Existing release points to a different source revision' >&2; exit 1;
  }
  [[ "$draft" == true ]] || published=true
fi
notes="$(mktemp)"
trap 'rm -f "$notes"' EXIT
python3.11 "$script_dir/render-native-release-notes.py" --dist . \
  --repository "$repo" --revision "$GITHUB_SHA" --tag "$tag" --output "$notes"
if [[ "$release_id" == 0 ]]; then
  python3.11 - "$tag" "$GITHUB_SHA" "$notes" <<'PYCREATE'
import json, pathlib, sys
pathlib.Path('create-release.json').write_text(json.dumps(dict(
    tag_name=sys.argv[1], target_commitish=sys.argv[2], draft=True,
    name='NInfer ' + sys.argv[1], body=pathlib.Path(sys.argv[3]).read_text())))
PYCREATE
  # The creation response supplies the ID; no read-after-write list lookup is needed.
  release_id="$(gh api --method POST "repos/$repo/releases" --input create-release.json --jq .id)"
fi
[[ "$release_id" =~ ^[0-9]+$ ]] || { echo 'Could not resolve the draft release ID' >&2; exit 1; }
# A retry can repair only this commit's draft. Published releases are immutable here.
if [[ "$published" != true ]]; then
  gh release upload "$tag" -R "$repo" "${assets[@]}" "${checksums[@]}" --clobber
fi
# Check the remote asset digests before making the draft visible or moving latest.
# A draft's tag may not exist until publication, so the tag endpoint can return 404.
gh api "repos/$repo/releases/$release_id" > assets.json
python3.11 - <<'PY'
import hashlib
import json
import pathlib
assets = {a['name']: a for a in json.loads(pathlib.Path('assets.json').read_text())['assets']}
paths = [*pathlib.Path('.').glob('*.7z.*'), *pathlib.Path('.').glob('*.tar.gz.*'), *pathlib.Path('.').glob('SHA256SUMS-*.txt')]
if set(assets) != {p.name for p in paths}:
    raise SystemExit('Remote assets differ from the prepared release')
for path in paths:
    with path.open('rb') as stream:
        digest = 'sha256:' + hashlib.file_digest(stream, 'sha256').hexdigest()
    asset = assets[path.name]
    if asset.get('digest') != digest or asset['size'] != path.stat().st_size:
        raise SystemExit(f'Remote asset verification failed: {path}')
PY
head="$(gh api "repos/$repo/commits/master" --jq .sha)"
if [[ "$head" != "$GITHUB_SHA" ]]; then
  echo "Source was superseded during upload; $tag remains a draft."
  exit 0
fi
if [[ "$published" != true ]]; then
  gh release edit "$tag" -R "$repo" --draft=false --latest --notes-file "$notes"
elif [[ "$(gh api "repos/$repo/releases/latest" --jq .tag_name)" != "$tag" ]]; then
  echo "Published release $tag is no longer latest; leaving it unchanged."
  exit 0
fi
[[ "$(gh api "repos/$repo/releases/latest" --jq .tag_name)" == "$tag" ]] || {
  echo 'Latest release verification failed' >&2; exit 1;
}
echo "Published https://github.com/$repo/releases/tag/$tag"
python3.11 "$script_dir/prune-native-releases.py" --repository "$repo" --tag "$tag" --revision "$GITHUB_SHA" --verified-assets assets.json
