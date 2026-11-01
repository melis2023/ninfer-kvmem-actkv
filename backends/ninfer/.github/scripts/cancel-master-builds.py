"""Cancel older master build/publication runs without touching other branches."""
import json
import os
import subprocess


WORKFLOWS = {
    ".github/workflows/ci.yml", ".github/workflows/native-release.yml",
}


def superseded(run, trigger):
    return (run["status"] != "completed" and run["head_branch"] == "master"
            and run["event"] in ("push", "workflow_dispatch")
            and run["path"].split("@", 1)[0] in WORKFLOWS
            and run["head_sha"] != trigger["head_sha"]
            and run["created_at"] <= trigger["created_at"])


def main():
    repo = os.environ["GITHUB_REPOSITORY"]

    def api(path):
        return json.loads(subprocess.check_output(["gh", "api", f"repos/{repo}/{path}"], text=True))

    trigger = api(f"actions/runs/{os.environ['GITHUB_RUN_ID']}")
    if api("commits/master")["sha"] != trigger["head_sha"]:
        print("A newer master coordinator owns cancellation.")
        return
    pages = []
    for status in ("queued", "in_progress", "waiting", "pending", "requested"):
        pages.extend(json.loads(subprocess.check_output(
            ["gh", "api", "--paginate", "--slurp",
             f"repos/{repo}/actions/runs?branch=master&status={status}&per_page=100"], text=True)))
    for page in pages:
        for run in page["workflow_runs"]:
            if not superseded(run, trigger):
                continue
            # Never cancel a newer commit if a push races this coordinator.
            comparison = api(f"compare/{trigger['head_sha']}...{run['head_sha']}")
            if comparison["status"] != "behind":
                continue
            result = subprocess.run(
                ["gh", "api", "-X", "POST", f"repos/{repo}/actions/runs/{run['id']}/cancel", "--silent"],
                capture_output=True, text=True)
            if result.returncode:
                if api(f"actions/runs/{run['id']}")["status"] != "completed":
                    raise RuntimeError(f"Could not cancel {run['name']} ({run['id']})")
            else:
                print(f"Cancellation requested: {run['name']} ({run['id']})", flush=True)


if __name__ == "__main__":
    main()
