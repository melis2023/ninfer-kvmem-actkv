#!/usr/bin/env bash
# Publishes a build's progress as a commit status, once a minute, while <pid> lives: ninja's own
# [done/total] from <log>, the percentage, and an ETA from the rate over the last ten minutes.
# GitHub shows a running step's log only to signed-in viewers; a status is readable by anyone
# through the API (repos/<repo>/commits/<sha>/statuses) and in the commit's checks.
#
#   build-progress.sh <pid> <log> <context>
set -uo pipefail
pid="$1" log="$2" context="$3"
api="repos/$GITHUB_REPOSITORY/statuses/$GITHUB_SHA"
url="$GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID"
samples=()   # "epoch done" pairs, one a minute

post() {
  gh api --silent -X POST "$api" -f state="$1" -f context="$context" -f description="$2" \
    -f target_url="$url" || true
}

start=$(date +%s)
while kill -0 "$pid" 2>/dev/null; do
  sleep 60
  progress="$(grep -ao '^\[[0-9]*/[0-9]*\]' "$log" 2>/dev/null | tail -n 1 | tr -d '[]')"
  [[ -n "$progress" ]] || { post pending "configuring ($(( ($(date +%s) - start) / 60 )) min)"; continue; }
  done_=${progress%/*} total=${progress#*/} now=$(date +%s)
  samples+=("$now $done_")
  (( ${#samples[@]} > 10 )) && samples=("${samples[@]:1}")
  read -r t0 d0 <<< "${samples[0]}"
  eta='ETA n/a'
  if (( now > t0 && done_ > d0 )); then
    seconds=$(( (total - done_) * (now - t0) / (done_ - d0) ))
    eta="ETA $(( seconds / 3600 ))h$(( seconds % 3600 / 60 ))m"
  fi
  post pending "$done_/$total ($(( done_ * 1000 / total / 10 )).$(( done_ * 1000 / total % 10 ))%), $eta, $(( (now - start) / 60 )) min in"
done
