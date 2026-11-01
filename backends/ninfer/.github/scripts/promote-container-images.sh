#!/usr/bin/env bash
# Both commit-tagged CUDA images must exist before moving either floating tag.
set -euo pipefail
image="ghcr.io/$(printf '%s' "$GITHUB_REPOSITORY" | tr '[:upper:]' '[:lower:]')"
for cuda in 12.9 13.4; do
  docker buildx imagetools inspect "$image:cuda$cuda-${SHA:0:12}" >/dev/null
done
for cuda in 12.9 13.4; do
  head="$(gh api "repos/$GITHUB_REPOSITORY/commits/master" --jq .sha)"
  if [[ "$head" != "$SHA" ]]; then
    echo 'Source is no longer the head of master; keeping commit tags only.'
    exit 0
  fi
  docker buildx imagetools create --tag "$image:cuda$cuda" "$image:cuda$cuda-${SHA:0:12}"
done
