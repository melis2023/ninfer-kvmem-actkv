# Multi-session validation, 2026-09-27–28

Validated server source `154fffb` (orphan session-cache cleanup) on Windows with
RTX 5060 Ti 16 GiB and RTX 5050 Laptop 8 GiB, NVIDIA driver 610.62, MSVC 19.44,
and a Release CUDA 13.2 build. Server SHA-256: `C9BEF58BAF34F0A7455AA07CB91FBDC0F03DDDA1C43F4D258E6F20BF60C36CE3`.
The `llama.cpp` checkout is `b81c99b` with the existing KVMem integration patches
applied in its working tree; those patches were not edited during this audit.
The production C++ source was unchanged during this audit; changes add test
coverage and this report. Raw local records: `artifacts/multisession-audit-20260927/`.

## Results

All 1899 assertions in the final real-server/CLI runs passed. Earlier runs
superseded by the expanded disk suites are excluded from this total.

| Test configuration | Passed |
|---|---:|
| CLI validation | 70/70 |
| 5050 / 0.8B / RAM-only / MTP off | 147/147 |
| 5050 / 0.8B / disk / MTP off | 220/220 |
| 5050 / 0.8B / disk / MTP snapshots | 230/230 |
| 5050 / 0.8B / API and restart / MTP off | 237/237 |
| 5050 / 0.8B / API and restart / MTP snapshots | 237/237 |
| 5060 Ti / IQ3 K8/V4 / RAM-only / MTP replay | 147/147 |
| 5060 Ti / IQ3 / 1:10 service exchange | 118/118 |
| 5060 Ti / IQ3 K8/V4 / three ~5 GiB sessions | 217/217 |
| 5060 Ti / IQ3 K8/V4 / image sessions | 36/36 |
| 5060 Ti / IQ3 K8/V4 / long NVMe-disabled request | 10/10 |
| Both GPUs / Q4_K_M Q8 KV / tensor 5:1 / MTP snapshots | 230/230 |

The Windows host CTest suite also passed 6/6, including seven lifecycle scenarios
exercised with child processes. Its coverage includes OS lock release, 12 simultaneous startups,
live-cache protection, partial writes, unknown files, symlinks/directory junctions,
and failed-deletion retry. The opt-in 1:10 transfer fixture passed both directions
for two cycles: A=1,073,741,820 bytes, B=1,610,612,730 bytes, RAM peak=1,610,612,730,
disk peak=2,040,110,370 against quota=2,147,483,640. The >4 GiB streaming snapshot
roundtrip also passed.

## Behavior checked

- Default single-session and RAM-only modes; prefix matching without client IDs;
  count/byte retention caps; accepted active sessions larger than the RAM soft cap;
  unlimited RAM with zero cap; no unnecessary spills from pessimistic forecasts.
- Four histories under a three-session cap use count LRU. Insufficient final disk
  quota uses disk LRU. Recent sessions survive and evicted histories recompute.
  Charged bytes match actual `.kv`/`.tmp` lengths after every successful short turn.
- Main KV, recurrent state and MTP state survive disk restores with MTP off,
  MTP snapshots and the 27B MTP replay path. Checks require correct channel answers
  and substantial prefix hits.
- Corrupt/missing snapshots recompute. A locked middle chunk returns HTTP 503
  without dropping unrelated sessions; the outgoing session and then the target
  recover with cached prefixes. Host tests inject write, rename, read, checksum,
  validation and deletion failures and insufficient transfer workspace.
- Chat Completions and Responses, streaming and non-streaming; invalid/unknown
  ID fallback; reusing an ID for different content; original-history preservation;
  oversized-request rejection without cache-state changes; eight rounds of three
  concurrent HTTP requests; recovery after a disconnected generation stream.
- Two real servers share a cache root. Restart reclaims only the killed server's
  files, logs the exact removed byte count, and preserves the live server's usable
  snapshots. A restarted process starts with a cold session, not stale runtime state.
- Image histories restore red/blue/green correctly. Changing only the image to
  yellow with the same text and client ID prevents stale-prefix reuse, while the
  original red history stays reusable.
- The pre-existing service's Q4_K_M model and tensor 5:1 GPU split receive the full
  short disk/fault regression with MTP snapshots.

## Three large sessions

Measured session storage at the end of growth: A=4.783 GiB, B=4.779 GiB, C=4.782 GiB. RAM soft cap=5 GiB,
NVMe quota=10 GiB. Five A/B/C rounds completed all 15 disk restores, without a
populated-session eviction, disk error or cache-clear fallback. Restore routes:
`{'exchange': 5, 'ram_first': 10}`. Maximum planned transient disk use was 10,736,351,752 bytes,
below the 10,737,418,240-byte quota. The host transfer test above checks actual
allocation/file peaks after every move; the service log's peak is the planner's
bound, not a continuously sampled filesystem measurement.

| Transfer path | Samples | Mean logged restore | Range |
|---|---:|---:|---:|
| exchange | 5 | 23.89 s | 23.02–25.20 s |
| ram_first | 10 | 26.75 s | 23.96–29.47 s |

These paths were selected by the planner during normal switching. The samples
have different target histories and memory availability, so they are not a
controlled comparison that forces both algorithms to perform the same transfer.

| First restore | Prompt tokens | Hot request | Identical cold request | Speedup |
|---|---:|---:|---:|---:|
| A | 143,522 | 24.90 s | 173.45 s | 6.97x |
| B | 142,603 | 27.17 s | 171.46 s | 6.31x |
| C | 141,809 | 26.22 s | 170.02 s | 6.48x |

With all session/NVMe flags omitted, the same A input took
171.70 s cold and 1.49 s on its
follow-up, with 143,528 cached tokens.
No session disk cache or transfer was initialized. Timings are individual local
runs, not a statistical benchmark; this machine also ran the small-model tests
during part of the large-session preparation.

## Reproduction and limits

Use `scripts/test_server_conversations.py` for RAM-only cases,
`scripts/test_server_session_disk.py` for disk and fault cases,
`scripts/test_server_session_lifecycle.py` for API/concurrency/restart behavior,
and `scripts/test_server_session_multimodal.py` for image sessions. The large runs
use `test_server_session_exchange_1to10.py --cycles 3`,
`test_server_three_session_5g.py --cycles 5`, and
`test_server_nvme_off_long.py` with the generated `cold-messages-A.json`.
See [session cache usage and test commands](../../session-disk-cache.md).

The tensor run sets `CUDA_VISIBLE_DEVICES` to 5060 Ti then 5050, with
`LLAMA_ARG_DEVICE=CUDA0,CUDA1`, `LLAMA_ARG_SPLIT_MODE=tensor`, and
`LLAMA_ARG_TENSOR_SPLIT=5,1`. Its model is Qwen3.8-27B-UD-Q4_K_M with Q8 KV and
`--mtp --mtp-state snapshots --gpu-layers all`. The first attempt was rejected
before model load because the fixture used `--gpu-layers 99`; the fixture now
accepts `all`, as required by the existing multi-GPU argument validation.
No server implementation change was needed.

The original service on port 18200 was restored after the tests using its
original Vulkan-enabled binary and exactly the same command line (including
the Q4 model, tensor 5:1 split, Q8 KV, MTP snapshots and Vulkan2 vision projector).
Health returned `ok` and a live completion returned the expected `5`.
Restoration timestamp: `2026-09-28T00:09:49.4326915+08:00`.

Fault tests deliberately constrain the transfer planner or a test cache's quota;
they do not exhaust the entire machine's RAM or filesystem. Forced process death
is covered; real power cuts, network filesystems, multi-day soak, and every model
or context length are outside this run. Existing unmarked legacy directories
remain intentionally untouched. Session caching is still process-local.
