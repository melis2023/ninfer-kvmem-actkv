"""Wait for cancelled runs of this workflow and branch to finish cache cleanup."""
import json
import os
import subprocess
import time
from urllib.parse import quote


def main():
    repo = os.environ["GITHUB_REPOSITORY"]

    def api(path):
        return json.loads(subprocess.check_output(["gh", "api", f"repos/{repo}/{path}"], text=True))

    current = api(f"actions/runs/{os.environ['GITHUB_RUN_ID']}")
    query = (f"actions/workflows/{current['workflow_id']}/runs?"
             f"branch={quote(current['head_branch'], safe='')}&status=in_progress&per_page=100")
    pages = json.loads(subprocess.check_output(
        ["gh", "api", "--paginate", "--slurp", f"repos/{repo}/{query}"], text=True))
    previous = {
        run["id"] for page in pages for run in page["workflow_runs"]
        if run["run_number"] < current["run_number"]
    }
    deadline = time.monotonic() + 360
    while previous:
        previous = {run for run in previous if api(f"actions/runs/{run}")["status"] != "completed"}
        if not previous:
            break
        print(f"Waiting for previous build cache cleanup: {sorted(previous)}", flush=True)
        if time.monotonic() >= deadline:
            raise RuntimeError("Previous build cleanup did not finish within six minutes")
        time.sleep(15)
    print("Previous build cleanup is complete; compiler cache handoff can proceed.", flush=True)


if __name__ == "__main__":
    main()
