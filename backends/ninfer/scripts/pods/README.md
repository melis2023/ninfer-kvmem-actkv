# Pod routines

Use Python 3.11: `python3.11 scripts/pods/harness.py COMMAND`.
Provider credentials stay in the provider CLI's local configuration. Source snapshots include
local edits; no push is required. All remote jobs run in named tmux sessions and retain their
script, source identity, output and exit code under `/workspace/ninfer-work/jobs/`.
Deployment preserves timestamps for unchanged contents and timestamps changed files on arrival,
so a local edit made during an older remote build cannot leave its stale object selected by Ninja.
`deploy --snapshot DIRECTORY` reuses a saved `source.tar.gz`/`source.json` pair across hosts.
`NINFER_POD_STATE=.local/pods/a6000/pod.json` selects another rental record and its result directory;
the guard inherits that exact path. The default record and its retained files remain separate.

```sh
python3.11 scripts/pods/harness.py inventory
python3.11 scripts/pods/harness.py offers 'gpu_name=RTX_3090 num_gpus=2 verified=true rentable=true' \
  --hours 3 --disk 180 --download-gb 80
python3.11 scripts/pods/harness.py rent OFFER --hours 3 --disk 180 --download-gb 80 --image CUDA_DEVEL_IMAGE
python3.11 scripts/pods/harness.py wait
python3.11 scripts/pods/harness.py probe
python3.11 scripts/pods/harness.py deploy
python3.11 scripts/pods/harness.py start bootstrap scripts/pods/bootstrap.sh
python3.11 scripts/pods/harness.py start build scripts/pods/build.sh --after bootstrap
python3.11 scripts/pods/harness.py start models scripts/pods/models.sh --after bootstrap
python3.11 scripts/pods/harness.py start checks scripts/pods/engine_checks.sh --after build --gpu
python3.11 scripts/pods/harness.py start ngram-checks scripts/pods/ngram_checks.sh --after build --gpu
python3.11 scripts/pods/harness.py start host-checks scripts/pods/host_checks.sh --after ngram-checks --gpu
python3.11 scripts/pods/harness.py watch build-observer
python3.11 scripts/pods/harness.py status build
python3.11 scripts/pods/harness.py status ngram-checks --file profile-heldout.txt --lines 30
python3.11 scripts/pods/harness.py collect --job checks
python3.11 scripts/pods/harness.py collect
python3.11 scripts/pods/harness.py destroy
```

Each Vast host sets its own price per GB of download and upload traffic on top of the hourly
rate, and a run that downloads models can pay more for traffic than for the GPU: on 2026-10-14 two
hosts charged $0.026 and $0.039 per GB, $6.15 for 196 GB against $0.27 of GPU time. RunPod bills no
traffic. Choosing a rental:

- `offers` leaves out hosts that charge more than $0.005 per GB of downloads
  (`--max-download-cost`, or an `inet_down_cost` term in the query), estimates each remaining offer's
  whole rental from `--hours`, `--disk`, `--download-gb` and `--upload-gb`, and lists the cheapest
  ten by that estimate.
- It quotes the same card on RunPod (the query's `gpu_name`, or `--runpod-gpu`): the community or
  secure hourly price and the container disk. RunPod is the `choice` when it has the card in stock
  and quotes no more than the cheapest Vast offer; rent it with the printed `~/runpod/pod-up.sh`
  command and `attach` the pod.
- `rent` repeats both checks for the chosen offer, refuses a host over the download price or a card
  RunPod quotes for no more (`--over-runpod` rents anyway, for example when RunPod's stock does not
  materialize), and records the estimate in the rental record.
- Download only the files the jobs read, once per rental: no whole repositories, no second copy of
  an artifact the rental already holds, and a resumed download (`aria2c --continue`) rather than a
  fresh one after a failure. Move data between rentals with `copy-from` instead of downloading it
  again when both sit with the same provider.

Install tmux on a fresh image before dispatching jobs. `attach --host HOST --port PORT` also works
with an existing SSH host, including a RunPod created with `~/runpod/pod-up.sh`; attached hosts
cannot be destroyed by this harness. Vast endpoints are refreshed before each operation.
Rentals inject the public SSH key with their startup script. `wait` checks SSH for at most ten
minutes and installs tmux if needed; `discard-empty` removes only an undeployed harness rental
after failed provisioning. `NINFER_POD_SSH_KEY` selects another private-key file locally.

Each job name is unique. `stop-job NAME` interrupts only that tmux session and retains its log;
`start NAME SCRIPT --retry` archives the finished attempt before retrying. The wrapper runs the
routine in a child shell, so a routine's `exec` still produces a completion marker. Repeat `--after`
to require several jobs; a failed dependency or a different source snapshot blocks execution. `--gpu` queues jobs under
one host lock, so independent model checks can run asynchronously without competing for VRAM;
the `executing` timestamp records admission after the lock. `--input FILE` stages a
helper under `$NINFER_JOB_DIR/inputs` and records its SHA256; it can be repeated. Use it to repair
an independent routine without changing the source snapshot during a build. Do not deploy while
jobs are running. `status` distinguishes a live job, a live dependency wait, and an orphaned job
whose tmux session disappeared without an exit marker. A satellite samples host/process/GPU metrics for
two minutes, then exits. Process command arguments and environment are deliberately excluded.

A local guard stops the exact rental at `--hours`, preserving its files. On macOS, launchd owns
the guard independently of the app's tool processes and caffeinate prevents idle sleep. launchd
relaunches the guard whenever it exits, so `stop`, `destroy`, `discard-empty` and the guard itself,
after a confirmed stop or once its rental is destroyed or replaced, remove its job. A guard whose
state file is gone still stops its rental at the deadline.
`guard --hours 1` replaces the deadline for the attached rental and logs its
admission immediately. Sleep prevention does not survive shutdown or logout; this is not a
provider-side spending cap. Collect results before destroying the finished ephemeral pod;
`collect` requires finished jobs and includes their logs, the source snapshot, conversion report,
hot-row profile and the 110 MB layer-12 CPU fixture when present; it excludes full model weights.
`collect --job NAME` downloads one finished job while other jobs may continue; it does not mark
the entire rental collected or permit teardown. Each collection uses its own remote archive, so
collecting a second receipt cannot truncate a large archive still being downloaded.
A new job invalidates the collection marker. Explicitly copy any full model that must
be retained before teardown. Local state and archives live in the ignored `.local/pods/` directory.

`stop` requires all jobs to be terminal and confirms the provider's stopped state, preserving
files while local implementation continues. `resume --hours 0.5` requests compute again and
rearms the guard; follow it with `wait` to confirm SSH/tmux readiness. Retained disk storage may
still be charged by the provider while compute is stopped.
Both explicit stop and the guard require `actual_status=exited/stopped` and
`intended_status=stopped`. A failed or queued resume may still have `actual_status=exited`
while its desired state is running; it must receive a stop request before the guard is removed.
Provider uptime is not an invoice or a timestamped utilization record.
Stopping does not reserve the GPUs: a later start can fail because another tenant uses the
capacity. The installed Vast CLI prints some rejections with exit status zero; the harness
requires a success acknowledgment and reports a redacted rejection category. A guard with an
obsolete deadline cannot stop a newer rental stage.

`copy-from SOURCE_STATE build|models|jobs/runtime-package` requests a provider transfer from a retained rental;
`copy-status` checks the provider marker and delivered files. An acknowledgment is not completion.
A transfer into a running container can leave the copied directories inaccessible (`ESTALE`);
`cancel-copy`, then `reboot` after all jobs finish, restores access. The fallback `restore_slice_inputs.sh` downloads pinned HF inputs, verifies their digests,
and preserves the private native artifact's revision. It takes `restore_slice_inputs.py` and the
retained `model-inputs.json` as job inputs; stage HF authentication first. It omits the unused table.

## Compiler cache, A/B runs and the residency check

`bootstrap.sh` installs ccache and `build.sh` compiles through it. The cache lives in
`/workspace/ccache` and travels between rentals through the private bucket `WaveCut/ninfer-cache`
(`ccache/cuda<release>-sm<arch>-<distribution>-<machine>.tar`): `build.sh` pulls it into a new pod
before the first compile and pushes it after every build (`NINFER_CCACHE_PUSH=0` skips the push).
`NINFER_CUDA_ARCH` selects the build's target and its cache (default `86`), and `NINFER_BUILD_DIR`
its tree.
Both steps need the credential `hf_archive.py --stage-auth` leaves at `/run/ninfer-hf/token`;
without it, or without a remote cache, the build compiles cold and says so. The Hub stores only the
chunks that changed since the last push. `build.sh` prints the build's seconds and ccache's hit
counts.

```sh
uv run --python 3.11 --with huggingface_hub python scripts/pods/hf_archive.py --stage-auth
python3.11 scripts/pods/harness.py start bootstrap scripts/pods/bootstrap.sh
python3.11 scripts/pods/harness.py start build scripts/pods/build.sh --after bootstrap
```

`serve_ab.py PLAN.json --out DIR` runs ninfer-serve configurations interleaved: round r starts at
configuration r mod n, each start passes over the prompts `repeat` times, and every request is
greedy with EOS ignored. It records decode and prefill rates, time to first token, the expert
cache's hits and bytes moved, and each answer's digest, and writes a per-configuration summary with
ranges and the change against the first configuration. Servers run in their own process group and
are ended on exit or SIGTERM, so `stop-job` leaves none behind. Stage the plan and any prompt files
with `--input`.

`residency_check.sh` (MODEL and TABLE in the environment) scores the first 40 KB of the WikiText
stream with host and with disk experts and fails unless the two agree exactly. Run it after any
change to the expert cache, the misses, the slot pool or the prefetch.

## Flash-Next qualification routines

The remaining routines qualified the Flash-Next work of October 2026 and reproduce it. Each is a
job script for `start`; the `compare_*.py` scripts read finished jobs' results locally.

| routines | what they check |
|---|---|
| `mtp_assemble_local.sh`, `mtp_qualify.sh`, `mtp_vision.sh`, `mtp_publish.sh`, `mtp_release.sh` | attaching Unsloth's MTP block to the published models without touching their other objects, host/disk MTP repetition, the media fallback, and the verified public update |
| `ngram_corpus_build.py`, `ngram_profile_broad.py`, `ngram_profile_native_check.sh`, `ngram_profile_attach.sh`, `ngram_profile_engine.sh`, `ngram_profile_publish.sh` | the broad hot-row profile: corpus sampling, per-group held-out hits, byte equality with the C++ profiler, attaching it to the table, Engine RAM hits |
| `ngram_cpu_checks.sh`, `ngram_row_cache_checks.sh`, `ngram_default_check.sh`, `ngram_profile_checks.sh`, `compare_ngram_cache.py` | the row reader's OS queue (io_uring, with the positioned-read fallback), exact rows, failure and eviction, the opt-in row cache; `ngram_cpu_checks.sh` needs no GPU and runs in a plain Linux container |
| `hc_build.sh`, `hc_oracles.sh`, `hc_checks.sh` | the hyper-connection kernels against their oracles and the paired HC benchmark |
| `layer_slice_*.sh`, `standalone_table.sh` | layer-slice conversion and its Engine checks, with and without PLE, and the standalone IQ4_NL table |
| `pipeline_single.sh`, `pipeline_split.sh`, `compare_pipeline.py` | one device against a split over two: three requests at 4,032, 33,024 and 131,008 input tokens; `--eos-token 248046` requires equality through the first end-of-sequence token while every fixed-mode repeat must match in full |
| `native_*`, `fp8_*` | the native-Q2 recipe and the FP8 n-gram table, both kept as experiments |
| `runtime_package.sh`, `bootstrap_runtime.sh`, `runtime_install.sh`, `hf_restore.sh` | running an already-built test executable on a second host of the same image without rebuilding, and restoring pinned private artifacts |

`build_space.sh` frees a rental's download caches when disk runs short; the build runner keeps
compiler scratch in tmpfs, since model storage and compiler files together exhausted a 180 GiB disk.

## Private model archives

Retain intermediate `.ninfer` models in private Hugging Face model repositories. Each experiment
uses an explicitly selected repository so a candidate can be kept for production qualification or
removed later. Temporary transfers use the private bucket `WaveCut/ninfer-cache` under a prefix of
their own; do not create model repositories for them. Update `hf_archive_manifest.json` with explicit artifact
paths, repo names, qualifications, limitations and any companion table before archiving.
The archive contains the model, conversion report, input revisions and a model card. It does not
include arbitrary job logs or source checkpoints. Uploading is not production qualification.

```sh
uv run --python 3.11 --with huggingface_hub==1.17.0 python scripts/pods/hf_archive.py --stage-auth
python3.11 scripts/pods/harness.py start hf-archive scripts/pods/hf_archive.sh \
  --input scripts/pods/hf_archive.py --input scripts/pods/hf_archive_manifest.json
python3.11 scripts/pods/harness.py watch hf-archive-observe
python3.11 scripts/pods/harness.py status hf-archive
python3.11 scripts/pods/harness.py collect --job hf-archive
```

Authentication crosses SSH stdin into a mode-600 file under `/run`, outside the job and upload
folders, and is removed when the upload routine exits. Staging uses hard links; large model
payloads are not copied locally. The resumable Hub upload cache remains on the pod. The uploader
refuses an existing public repository, then checks private visibility, final revision, file size,
the upload client's SHA-256 against Hub metadata, and downloaded report bytes. A completed
`verified.json` record is the archive receipt. A failed attempt preserves the model and cache;
stage authentication again and retry the finished job with `--retry`.
Keep model deletion separate from upload; a failed backup does not authorize deleting the source.

Large artifact work that needs no GPU runs on a CPU Pod: `runpod_cpu.py create` provisions a 4-vCPU
Pod with a temporary volume (select `.local/pods/runpod-cpu/pod.json` with `NINFER_POD_STATE`), the
same harness deploys and runs on it, and `runpod_cpu.py destroy` deletes only that Pod and volume
once a local `cleanup-approved.json` receipt authorizes it. Model weights stay on the remote Linux
host; the release scripts refuse macOS and paths outside `/workspace`, hash-check every source,
read back every output object before uploading it, and record the local SHA-256, the remote hash
and the size before deleting temporary copies. A successful upload does not establish GPU
qualification.

After qualification, `hf_archive.py --manifest MANIFEST --output DIR --refresh-metadata` updates
only existing private model cards and `artifact.json`. It uses local HF authentication, preserves
the model payload, refuses a concurrent revision change and verifies metadata readback plus
unchanged payload size/digest and privacy. Use `--select FILE --only BASENAME` to prepare a
subset from the canonical manifest. This path does not download or reupload model weights.
