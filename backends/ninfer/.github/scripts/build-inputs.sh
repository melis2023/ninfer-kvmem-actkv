#!/usr/bin/env bash
# Compare with the published revision so cancelled builds cannot hide pending source changes.
# --native uses the public native release; CI uses the published CUDA image.
# Manual dispatch forces all scopes. Automatic runs classify paths without starting a compiler.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
force() {
  printf '%s\n' "$1" >&2
  python3.11 "$script_dir/change-scopes.py" --force </dev/null
  exit 0
}
[[ "${GITHUB_EVENT_NAME:-}" != workflow_dispatch ]] || force 'Manual dispatch: all checks and builds requested.'
base=''
case "${GITHUB_EVENT_NAME:-}:${GITHUB_REF:-}" in
  pull_request:*) base="$(git rev-parse --verify --quiet 'HEAD^1')" || base='' ;;
  push:refs/heads/master)
    if [[ "${1:-}" == --native ]]; then
      base="$(gh api "repos/$GITHUB_REPOSITORY/releases/latest" --jq .target_commitish)" || base=''
    else
      repository="$(printf '%s' "$GITHUB_REPOSITORY" | tr '[:upper:]' '[:lower:]')"
      base="$(docker buildx imagetools inspect "ghcr.io/$repository:cuda13.4" \
        --format '{{ index .Image.Config.Labels "org.opencontainers.image.revision" }}')" || base=''
    fi ;;
  push:*) base="$(git rev-parse --verify --quiet 'HEAD^1')" || base='' ;;
esac
[[ "$base" =~ ^[0-9a-f]{40}$ ]] || {
  echo 'Cannot resolve an immutable comparison base; no automatic build admitted. Use manual dispatch for bootstrap.' >&2
  exit 1
}
if ! git cat-file -e "$base^{commit}" 2>/dev/null \
    && ! git fetch --quiet --no-tags --depth=1 origin "$base"; then
  echo "Base $base cannot be fetched; no automatic build admitted." >&2
  exit 1
fi
printf 'Comparing %s with HEAD\n' "$base" >&2
git diff --name-only --no-renames -z "$base" HEAD | python3.11 "$script_dir/change-scopes.py"
