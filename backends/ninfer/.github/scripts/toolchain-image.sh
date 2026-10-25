#!/usr/bin/env bash
# Makes the Dockerfile's toolchain stage available locally as ninfer-toolchain:ci and prints its
# registry reference. The stage is published as ghcr.io/<owner>/<repo>-toolchain under a tag that
# hashes its definition (the global ARGs and the stage), so it is built once per change: pulled when
# the registry has that tag, built here otherwise, and then pushed with PUSH=1 for later runs, which
# also deletes every other version of the package (GH_TOKEN must be able to).
#
#   PUSH=0|1 .github/scripts/toolchain-image.sh      (after docker login to ghcr.io)
set -euo pipefail
package="${GITHUB_REPOSITORY#*/}"; package="${package,,}-toolchain"
tag="$(sed -n '/^ARG CUDA_VERSION/,/^FROM toolchain AS build/p' Dockerfile | sha256sum | cut -c1-16)"
ref="ghcr.io/${GITHUB_REPOSITORY_OWNER,,}/$package:$tag"

delete_other_versions() {
  local kind=users versions
  [[ "$(gh api "repos/$GITHUB_REPOSITORY" --jq .owner.type)" == Organization ]] && kind=orgs
  local api="/$kind/$GITHUB_REPOSITORY_OWNER/packages/container/$package/versions"
  versions="$(gh api --paginate "$api?per_page=100" \
    --jq ".[] | select(.metadata.container.tags | index(\"$tag\") | not) | .id")"
  for id in $versions; do
    gh api -X DELETE "$api/$id" --silent && printf 'deleted toolchain version %s\n' "$id" >&2
  done
}

if docker pull --quiet "$ref" > /dev/null 2>&1; then
  printf 'pulled %s\n' "$ref" >&2
else
  printf 'building %s\n' "$ref" >&2
  docker build --target toolchain --tag "$ref" \
    --label "org.opencontainers.image.source=$GITHUB_SERVER_URL/$GITHUB_REPOSITORY" . >&2
  if [[ "${PUSH:-0}" == 1 ]]; then
    if docker push --quiet "$ref" >&2; then
      delete_other_versions
    else
      printf 'could not push %s; using the local build\n' "$ref" >&2
    fi
  fi
fi
docker tag "$ref" ninfer-toolchain:ci
printf '%s\n' "$ref"
