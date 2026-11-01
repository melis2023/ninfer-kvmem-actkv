#!/usr/bin/env bash
# Publishes a build's progress as a commit status. GitHub shows a running step's log only to
# signed-in viewers; a status is readable by anyone through the API
# (repos/<repo>/commits/<sha>/statuses) and in the commit's checks.
#
#   build-progress.sh watch <pid> <log> <context>
#       In the background of the build step: once a minute while <pid> lives, ninja's own
#       [done/total] from <log>, the percentage, and an ETA from the rate over the last ten minutes.
#   build-progress.sh finish <context> <outcome>
#       In a step after the build, whatever <outcome> (steps.<id>.outcome) it had: stops the
#       watcher, then posts the final state.
#
# The newest status of a context is the one GitHub shows, so the final state must be the last one
# posted. The runner does not stop a step's background processes when the step ends, and a step
# that times out or is cancelled never reaches its own last line; that is why the final state comes
# from a step of its own, and why it first waits until the watcher, with any post it has in flight,
# is gone.
set -uo pipefail
api="repos/$GITHUB_REPOSITORY/statuses/$GITHUB_SHA"
url="$GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID"
pidfile="$RUNNER_TEMP/build-progress.pid"

post() {
  gh api --silent -X POST "$api" -f state="$1" -f context="$context" -f description="$2" \
    -f target_url="$url" || true
}

watch() {
  local pid="$1" log="$2"
  echo $$ > "$pidfile"
  trap 'rm -f "$pidfile"' EXIT
  # Bash runs a trap only once the command in the foreground returns, so a post in flight completes.
  trap 'exit 0' TERM
  local samples=()   # "epoch done" pairs, one a minute
  local start now next progress done_ total t0 d0 eta seconds
  start=$(date +%s) next=$((start + 60))
  while kill -0 "$pid" 2>/dev/null; do
    sleep 1
    now=$(date +%s)
    (( now >= next )) || continue
    next=$((now + 60))
    progress="$(grep -ao '^\[[0-9]*/[0-9]*\]' "$log" 2>/dev/null | tail -n 1 | tr -d '[]')"
    # The build may have ended during the sleep; its outcome belongs to the finish step.
    kill -0 "$pid" 2>/dev/null || break
    [[ -n "$progress" ]] || { post pending "configuring ($(( (now - start) / 60 )) min)"; continue; }
    done_=${progress%/*} total=${progress#*/}
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
}

finish() {
  local outcome="$1" watcher state description
  if [[ -f "$pidfile" ]]; then
    watcher="$(<"$pidfile")"
    kill "$watcher" 2>/dev/null
    for _ in {1..100}; do kill -0 "$watcher" 2>/dev/null || break; sleep 0.1; done
  fi
  case "$outcome" in
    success) state=success description=finished ;;
    cancelled) state=error description=cancelled ;;
    *) state=failure description='failed or timed out' ;;
  esac
  post "$state" "$description"
}

case "${1:-}" in
  watch) context="$4"; watch "$2" "$3" ;;
  finish) context="$2"; finish "$3" ;;
  *) echo "usage: $0 watch <pid> <log> <context> | finish <context> <outcome>" >&2; exit 2 ;;
esac
