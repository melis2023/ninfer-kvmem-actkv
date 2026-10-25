#!/usr/bin/env bash
# Deletes every version of the repository's container image except `latest` and the newest other
# tagged image, each with the manifests its index lists (the platform image and its attestation),
# so the registry holds the current image and the one before it. The publishing workflow runs it
# after every push; DRY_RUN=1 only lists what would go.
#
#   GH_TOKEN=<token that can read and delete the package> docker/prune-images.sh <owner/repo>
set -euo pipefail
(( $# == 1 )) || { printf 'usage: prune-images.sh <owner/repo>\n' >&2; exit 2; }
repo="${1,,}"
owner="${repo%%/*}" package="${repo#*/}"
kind=users
[[ "$(gh api "repos/$repo" --jq .owner.type)" == Organization ]] && kind=orgs
api="/$kind/$owner/packages/container/$package/versions"

versions="$(gh api --paginate "$api?per_page=100" \
  --jq '.[] | [.id, .name, .created_at, (.metadata.container.tags | join(","))] | @tsv')"
latest="$(awk -F'\t' '$4 ~ /(^|,)latest(,|$)/ { print $2 }' <<<"$versions")"
[[ -n "$latest" ]] || { printf 'no version is tagged latest; nothing pruned\n'; exit 0; }
previous="$(awk -F'\t' -v latest="$latest" '$4 != "" && $2 != latest' <<<"$versions" \
  | sort -t $'\t' -k3,3r | head -n 1 | cut -f2)"

# The registry token comes from the same credentials; curl reads them from stdin, not argv.
token="$(printf 'user = "%s:%s"\n' "$owner" "$GH_TOKEN" \
  | curl -fsS -K - "https://ghcr.io/token?scope=repository:$repo:pull" | jq -r .token)"
keep="$latest"
for digest in $latest $previous; do
  keep+=$'\n'"$digest"
  keep+=$'\n'"$(printf 'header = "Authorization: Bearer %s"\n' "$token" \
    | curl -fsS -K - \
        -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json' \
        "https://ghcr.io/v2/$repo/manifests/$digest" | jq -r '.manifests[]?.digest')"
done
printf 'keeping %s (latest)%s\n' "$latest" "${previous:+ and $previous}"

deleted=0
while IFS=$'\t' read -r id name created tags; do
  [[ -n "$id" ]] || continue
  grep -qxF -- "$name" <<<"$keep" && continue
  printf 'deleting %s (%s, %s)\n' "$name" "${tags:-untagged}" "$created"
  [[ "${DRY_RUN:-0}" == 1 ]] || gh api -X DELETE "$api/$id" --silent
  deleted=$((deleted + 1))
done <<<"$versions"
printf '%s version(s) %s\n' "$deleted" "$([[ "${DRY_RUN:-0}" == 1 ]] && echo 'would go' || echo deleted)"
