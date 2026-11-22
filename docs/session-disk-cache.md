# Session KV cache in RAM and on NVMe

This extends the session ownership and matching introduced by PR #45. There
is still one active inference slot and one GPU working set. Inactive sessions
can retain their host KV in RAM or move their large payloads to an SSD/NVMe
directory. Requests continue to use the normal chat API, optionally with
`"kvmem": {"conversation_id": "my-chat"}`.

Example server arguments (in addition to your model/context/KV options):

```text
--kvmem-conversations 3
--kvmem-session-ram-gb 12
--kvmem-session-nvme-gb 40
--kvmem-session-cache-dir D:/KVMem/session-cache
```

The `GB` options use GiB (1024³ bytes) and accept fractions. `12` and `40` are
example limits, not a guarantee that every model's 200k context fits. The RAM
requirement depends on KV types, attention dimensions, recurrent checkpoints,
MTP and the requested output length.

## Capacity and LRU

- `--kvmem-conversations` limits the total number of cached sessions across
  RAM and disk, including the active session. Exceeding it discards the least
  recently used inactive session.
- `--kvmem-session-ram-gb` aliases `--kvmem-conversations-gb`: both are a **soft
  limit on active plus inactive session RAM**, with or without NVMe. `0` means
  unlimited. An active session larger than the limit continues serving and
  produces a warning. Idle payloads are moved out first; necessary cold metadata
  remains accounted, even when active KV plus that metadata exceeds the limit.
- To free RAM, the least recently used inactive RAM session moves to disk.
  Only insufficient **final** disk space/quota permits LRU eviction. The target's
  files are credited as space freed by restoring it. Temporary transfer pressure
  and I/O failures preserve caches; they are not treated as eviction requests.
  A failed deletion keeps its quota charge. The session count cap still applies.
- File size includes metadata, checksum and temporary writes. Space is reserved
  before writing. RAM includes retained KV, means, block/position metadata,
  token indexes, query state and recurrent/MTP checkpoints. Cold entries still
  need RAM indexes and a frozen allocation manifest, which count toward the cap.
- Touching a session on an accepted request updates its LRU timestamp. Moving
  it between tiers does not. The target and the active request are protected
  from eviction.

The RAM limit describes session storage, **not process RSS**. Model weights,
the GPU working set, backend transfer/compute buffers, the current HTTP request
and media encoder buffers, allocator overhead, and the OS file page cache need
additional memory. Retention uses measured RAM and actual restore sizes, rather
than the conservative future prompt/output estimate printed in diagnostics.
Exceeding the soft limit does not reject a request. The active session must fit
the machine's actual RAM; this setting cannot guarantee protection from OOM.

Disk mode requires KVMem, flash attention, `--kvmem-conversations > 1`, a
an explicit cache directory. Leave `--kvmem-cpu-gb` and
`--kvmem-nvme-gb` at zero and do not enable `--kvmem-raw-k-nvme`: those older
options allocate separate block-tier arenas per session. This session file
cache works on Windows as well as POSIX and does not depend on the older
POSIX NVMe tier or `KVMEM_ENABLE_NVME`.

Without `--kvmem-session-nvme-gb`, the existing RAM-only PR #45 policy remains:
the original RAM cap is a soft retention cap and allows a larger active store.
The [NVMe-disabled IQ3 K8/V4 regression](validation/session-cache/session-nvme-off-regression.md) checks
the default single-session path and three RAM-only sessions on RTX 5060 Ti.

## Save and restore

The adapter drains pending GPU transfers before detaching a session, then moves
ownership of the existing host store without copying the entire KV. Detached
stores are frozen while their allocations migrate. Packed main/MTP KV, exact
F32 mean sums, recurrent states and speculative carry are all included. Shared
checkpoints are indexed once. Model configuration, vector lengths, raw block
metadata and checkpoint references remain in the trusted process-local manifest.

Each file contains a group of complete allocations (normally up to 16 MiB),
with a session id, generation, chunk index, exact lengths and checksum. A single
larger allocation, such as a recurrent checkpoint, stays an indivisible unit;
the planner includes its full restore footprint. I/O calls are bounded to 1 MiB.
No full-session serialization buffer is created. Source memory is released
only after publishing its chunk; source files are removed only after the whole
chunk is read and validated. Files are independently reclaimable.

Before migration, the planner checks the final layout and schedules every move:

1. With sufficient disk workspace, spill outgoing payloads before restoring.
2. With sufficient actual RAM workspace, restore first to reclaim target files.
3. Otherwise, alternate chunk writes and reads, reclaiming each source in turn.

For example, RAM A=10 GiB and disk B=15 GiB with a 20 GiB disk quota can end as
RAM B=15 GiB and disk A=10 GiB, even with a 12 GiB RAM soft limit. It does not
require simultaneous complete 25 GiB copies in either tier. The planner uses
OS physical/commit headroom with a safety margin, separately from the soft cap.
It rechecks RAM before each restore and disk space before each write. These are
observations, not reservations against other processes or an OOM guarantee.

If no safe schedule fits even one allocation, switching returns HTTP 503 with
the caches retained. A mid-transfer I/O/allocation failure likewise leaves a
mixed RAM/disk manifest: a subsequent request can resume either session. There
is no promise of immediately returning all of the old session to RAM after a
failure. The adapter refuses to attach a frozen/partial session. Only complete
restoration allows thawing and attachment. A corrupt target is discarded and
recomputed; unrelated and outgoing sessions retain their own KV. Optional idle
demotion failures do not fail a hot request or clear its completed active cache.

The stored media index uses llama.cpp's placeholder chunks; incoming request
data supplies media if a span must be replayed. This transfer recovery is within
one running process, not durable restart or power-loss recovery.

These files are **only valid within the same server run**. Their small in-memory
indexes include model/runtime identity. Each run creates a unique subdirectory
with a versioned `.kvmem-owner` marker and holds an exclusive OS file lock for
its lifetime. Graceful destruction removes that run's cache. Forced termination
can leave files behind, but the OS releases the lock.

On the next startup with session disk caching enabled, the server reclaims
marked run directories whose locks it can acquire. It removes complete `.kv`
chunks and unfinished `.tmp` chunks before creating its new run. Other live
servers sharing the root are skipped. A persistent `.kvmem-cache.lock` file
serializes scanning, registration and graceful cleanup; do not delete or replace
this small control file while any server uses the root. PID reuse and elapsed
time are not used to decide ownership. Use a local filesystem with working OS
file locks; shared/network filesystem locking is outside this contract.

Cleanup never recurses into directories or follows links/reparse points.
Unknown marker formats and directories containing unexpected entries are left
untouched. Legacy run directories without a marker are also preserved: remove
those manually only after confirming the old servers have stopped. Startup logs
report removed directories and bytes, live/skipped directories, and errors.
Failed chunk deletion retains the marker for a later retry and logs the path and
reason; it does not prevent a new run from starting. A crash before registration
finishes may leave an unmarked directory, but no snapshots have been written yet.
Quotas still apply to each server run, not all servers sharing a drive. Skipped
or undeletable leftovers consume real disk space outside the new run's quota.

The active session is loaded entirely into RAM. Restart persistence and
per-block paging of an active session are outside this implementation.

The portable `kvmem-session-cache-lifecycle-test` (CTest, when Python 3 is
available) starts real child processes, kills them during snapshot writes,
checks live-session restoration while another process cleans, races 12 startups,
and verifies graceful cleanup, legacy/user-file preservation, link exclusion,
and deletion-failure logging/retry. No model or GPU is needed. Run it with:

```sh
ctest --test-dir build --output-on-failure -R kvmem-session-cache-lifecycle-test
```

## Diagnostics and checks

`/slots` → `kvmem.conversations` reports RAM `bytes`/`bytes_max`,
`disk_bytes`/`disk_bytes_max`, `spills`, `restores`, `disk_errors`, and the existing
session/eviction counters. `--kvmem-trace` adds `session_select`, `session_spill`
and `session_restore` events, plus `session_transfer` with the chosen path and
planned peak/final disk bytes. RAM `bytes` may exceed its soft `bytes_max`.

Portable tests (no model required):

```text
cmake -S . -B build-session-host -DKVMEM_BUILD_LLAMA=OFF
cmake --build build-session-host --target kvmem-session-transfer-test kvmem-session-snapshot-test kvmem-conversation-store-test
ctest --test-dir build-session-host --output-on-failure -R "kvmem-(session-transfer|session-snapshot|conversation-store)-test"
```

The snapshot test also accepts `--large-file` for an explicit >4 GiB streaming
roundtrip using a 1 MiB buffer. It temporarily needs just over 4 GiB of free
space and removes its files when it finishes.

The transfer test scales the 10/15/20 GiB scenario to MiB and measures actual
vector allocations and charged disk bytes. It covers all three schedules,
insufficient temporary workspace without mutation, failures at each chunk's
write/rename/read/validation/deletion boundary, partial I/O within a multi-vector
chunk, failed temporary-file cleanup, wrong-session headers, bad checksums,
decreasing available RAM, reverse recovery and exact packed KV restoration.
For the opt-in **1:10** exchange and RTX 5060 Ti real-service stability check,
see [1:10 multi-session exchange stability test](validation/session-cache/session-exchange-stability.md).

Real model regression, including interleaved A/B/C histories, exact answers
against a RAM reference, prefix cache hits, corruption fallback and eviction:

```text
python scripts/test_server_session_disk.py --server PATH/llama-kvmem-server --model PATH/model.gguf --output artifacts/session-disk
```

Add `--mtp` for a model with an MTP head. Use `--mtp-state snapshots` for models
outside the specialized 27B replay kernel (for example the local 0.8B model).
The script starts only its own server processes on unused ports and downloads
no models. On Windows it also locks a middle snapshot chunk to force a real
HTTP 503, resumes the old session, and retries the target with cache hits intact.
For an explicit multi-GPU configuration, pass `--gpu-layers all` and supply
the server's device/split settings through `LLAMA_ARG_DEVICE`,
`LLAMA_ARG_SPLIT_MODE`, and `LLAMA_ARG_TENSOR_SPLIT`.
It also checks count-based LRU with four histories under a three-session cap,
and reconciles the charged disk bytes with actual snapshot file lengths after
each successful request.

For HTTP-level concurrency, Chat/Responses streaming, optional-ID fallback,
cancelled generation, and actual server crash/restart with another live server
sharing the cache root:

```text
python scripts/test_server_session_lifecycle.py --server PATH/llama-kvmem-server --model PATH/small-model.gguf --gpu GPU-UUID --mtp --output artifacts/session-lifecycle
```

This fixture runs two copies of the supplied model concurrently during its
shared-root phase; use a small model that fits twice on the selected GPU.
Run with and without `--mtp` to cover both generation paths. The default eight
rounds submit A/B/C simultaneously over HTTP, while the server continues to
serialize inference through its one active slot.

Image sessions have a separate real-model check. It switches between red, blue
and green images, restores their KV from disk, then changes only the image in
an otherwise identical prompt with the same client ID. The changed image must
miss the old cache and answer yellow; the original red history must remain
reusable. This fixture uses mixed K8/V4 KV and MTP snapshots:

```text
python scripts/test_server_session_multimodal.py --server PATH/llama-kvmem-server --model PATH/model.gguf --mmproj PATH/mmproj.gguf --gpu GPU-UUID --output artifacts/session-images
```

### Full validation (Windows, 2026-09-27 to 2026-09-28)

See [the multi-session validation report](validation/multi-backend/multi-session-validation-2026-09-27.md)
for the 5050 / 5060 Ti matrix, shared-root crash recovery, image sessions,
three approximately 5 GiB sessions under a 10 GiB NVMe quota, and matched
restore-versus-prefill timings.

### Local verification (Windows, 2026-09-26)

- Built with MSVC 19.44 and CUDA 12.9.86; this revision's GPU tests use RTX 5050.
- Transfer, snapshot, conversation-policy and server-options CTest targets passed.
- The earlier streaming implementation's >4 GiB file test remains available.
- Qwen3.5-0.8B Q8_0: interleaved A/B/C disk restores preserved answers and
  resumed 5,579–5,580 prefix tokens. The updated regression sets the RAM soft
  limit below one active session and covers successful requests, corrupt/missing
  files, locked-file recovery, quota eviction and unlimited RAM. It also checks
  that pessimistic capacity forecasts do not demote sessions whose actual RAM fits.
  All 150 checks passed.
- The same 0.8B model with MTP (`--mtp-state snapshots`) preserves answers and
  cached prefixes across main/MTP KV and recurrent/carry restoration. Its initial
  soft cap was about 82 MiB while retained RAM reached about 130 MiB; requests
  still succeeded. These are session counters, not RSS. The specialized 27B
  replay mode was not rerun in this revision: the 16 GiB GPU was in use.
- All 159 MTP checks passed, including evidence that the MTP follower ran in
  every fixture and recovery after a partially completed restore.
- The original RAM-only regression passed all 119 checks, including count LRU, byte-cap
  eviction and rejected-request isolation. Its fixed 90% hit assertion was
  corrected to allow the existing non-MTP query-checkpoint replay (512 tokens
  plus template slack); the test still requires a substantial cached prefix.

Real-model fixtures use roughly 6k-token histories. This is not a 200k-token
inference or disk-throughput benchmark.
