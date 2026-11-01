# Cutting a release

The [Native release](../../.github/workflows/native-release.yml) workflow builds Windows x64
executables and Debian 13 x64 binaries. As soon as both Linux matrix jobs succeed, its `containers`
job calls [Container image](../../.github/workflows/docker.yml) to package their archives without
compilation. This dependency excludes Windows and native release publication. A Windows failure
therefore does not block Docker. A failed or cancelled Linux job prevents both Docker variants.
CI builds every target and runs CTest for the same two multiarch CUDA configurations.

## Container publication

| Toolkit | Floating tag | Commit tag | NInfer architectures |
|---|---|---|---|
| CUDA 12.9.2 | `cuda12.9` | `cuda12.9-<commit12>` | `70;75;80;86;89` |
| CUDA 13.4.3 | `cuda13.4` | `cuda13.4-<commit12>` | `75;80;86;89;120a` |

Both images use Debian 13 and the binaries and CUDA runtime libraries from the calling native run.
The reusable workflow downloads artifacts from that run and checks out its exact commit. It needs
no dispatch token or separate workflow-completion event. The parent workflow's concurrency group
cancels obsolete builds and their Docker jobs together.
The handoff verifies volume checksums, packaged files, full source revision, CUDA version and the
complete architecture list. Each image keeps its device images in the shared backend runtime.
`commit12` is the first 12 hexadecimal characters of the source SHA; `VERSION` is not a tag.

The matrix publishes commit tags first. Only after both jobs succeed does promotion update the
floating CUDA tags, checking the current master SHA before each update. Registry tag updates are
separate operations: cancellation, a newer commit or registry failure can leave them temporarily
pointing to different revisions. Use both commit tags when the same revision is required.
A failed matrix does not promote either tag. Each variant reuses its previous image's inline
BuildKit cache. Native artifact jobs save shared compiler caches; CI only reads them.

The pipeline no longer updates `latest`, `multi`, version-only or `sha-…` tags. Existing registry
images are not deleted. The old latest-based pruning policy is removed because it cannot retain
both CUDA families safely; registry storage grows with published commit images until a separate
retention policy is configured. Manual native test runs build both images locally without pushing.
Publishing is enabled for master pushes, or for a manual master run with `publish: true`.

## Native publication

Each platform builds two variants from the same source commit:

| Toolkit | Device images |
| --- | --- |
| CUDA 12.9.2 | `70;75;80;86;89` |
| CUDA 13.4.3 | `75;80;86;89;120a` |

These are the latest CUDA 12 and 13 releases in the
[NVIDIA archive](https://developer.nvidia.com/cuda-toolkit-archive), checked on 20 October 2026.
Volta requires CUDA 12. This project's Blackwell kernels require CUDA 13.1 or newer.
Windows uses the official NVIDIA network installer, VS 2022, Ninja and pinned vcpkg libraries.
It imports `vcvars64.bat`; setup-msbuild alone does not provide the compiler environment.
Linux uses Debian 13. CUDA 13 uses NVIDIA's exact Debian 13 Toolkit metapackage. CUDA 12 uses
NVIDIA's official runfile installer, with only the Toolkit selected. Debian 13 rejects the SHA-1
binding in NVIDIA's Debian 12 APT key; the workflow does not relax signature verification.
Debian 13 supplies the FFmpeg version required by NInfer. CUDA 12.9 also needs matching
`noexcept(true)` declarations for `sinpi`, `cospi`, `sinpif` and `cospif` with glibc 2.41.
A version-guarded script patches those declarations and inline helper signatures, without changing
function bodies. `BUILD-INFO.json` records the patch. The pre-build probe compiles CUDA headers
through both host C++ and nvcc before testing cache reuse.
Compilation needs no GPU. Driver installation is not part of either build.

1. Compile release products with `NINFER_SHARED_RUNTIME=ON`. Six small command executables load
   one private runtime per CUDA family. Windows also keeps GGML in its existing kernel DLL.
   Component archives remain the compile owners. Linux compiles them as PIC for the shared runtime.
2. Check every requested architecture with `cuobjdump`. Run executable `--help` checks.
   Windows clears developer directories from `PATH`. Linux checks the dynamic library closure
   and uses a temporary CUDA driver stub, which is never packaged.
3. Package checksums, source metadata, launchers and licences. Windows includes runtime DLLs.
   Linux includes the CUDA runtime libraries and their EULA, records the required system packages,
   and includes a Dockerfile that installs those packages and copies the prebuilt binaries.
4. Test the archives and split them into volumes no larger than 1900 MiB. Keep Actions artifacts
   for seven days. Names are `ninfer-<os>-x64-cuda<major.minor>-sm<arches>-<commit12>`;
   `VERSION` remains metadata inside the archive and does not affect its name.
5. Assemble one universal package per OS from the two verified single-CUDA archives. Require matching
   source revisions, command sets, metadata and checksums. No compilation occurs in these jobs.
   Universal names are `ninfer-<os>-x64-cuda12.9-13.4-<commit12>`. Docker continues to use single-CUDA archives.
6. After all four builds and both assembly jobs pass, validate all six manifests, create a draft tagged `YYYY-MM-DD.N` (UTC date and daily build number),
   upload assets, and compare their remote SHA-256 digests and sizes.
   The publisher generates the release description with `.github/scripts/render-native-release-notes.py`.
   It lists all six package choices, their platforms and CUDA backends, exact part filenames and links,
   checksum manifests, part counts, compressed download totals and extraction commands. Counts and sizes
   come from the verified manifests and archive files. Draft retries refresh this guide when publishing.
   A retry finds its draft by the full source revision and keeps its original date/number. Numbers are
   reserved by both drafts and existing Git tags. Creation returns the draft ID directly, without a
   read-after-write lookup through the release list.
7. Publish and mark latest only if the source is still the head of `master`. Superseded builds
   retain their Actions artifacts or the draft created before a newer commit arrived.

Published assets are never overwritten. A retry repairs the same commit's unfinished draft, or
verifies an already published latest snapshot and retries its cleanup without uploading or editing assets.
After asset verification and publication as latest, the publisher removes older published native
releases from ancestor revisions. It retains Git tags, drafts, prereleases and unrelated release tags.
Cleanup checks latest and its verified asset inventory again before each deletion. A failure before
publication retains the previous release. A cleanup failure does not undo the verified new release.
Legacy `native-<commit12>` releases remain valid retry targets; new releases use `YYYY-MM-DD.N`.
Archive filenames still identify their full source through `<commit12>` and `BUILD-INFO.json`.

Every push to `master` compares inputs with the published native revision. Product C++/CUDA, headers,
dependencies, build configuration and shipped runtime inputs start native builds and Docker packaging.
Documentation and publication/control scripts run only their affected checks. Pending source changes
from a cancelled or failed build remain in this comparison. Manual dispatch forces builds and checks.
A new run cancels the previous run and publishes on success when build inputs changed. In **Actions → Native release
→ Run workflow**, publication defaults to off; enable `publish` only on `master` when intended.
Feature branches run only through manual dispatch and cannot publish. Build jobs have read
access; only the publication job receives `contents: write`. The master cancellation coordinator
also cancels older container publishers. Closing a PR cancels that PR's remaining runs.
A superseded upload can leave an unpublished draft; it cannot replace the current release.

## Portable runtime boundary

`cmake/NinferRuntime.cmake` links the command objects and their static component dependencies into
`ninfer-runtime.dll` or `libninfer-runtime.so`. The private C boundary exposes command dispatch and
a build-ID check. It exports no C++ SDK. Allocation, Engine objects and exceptions stay inside the
runtime. Existing CUDA compile owners remain intact. No whole-archive linkage is used.

The launcher imports no CUDA library. It reads the selected device ordinals through the Driver API,
then checks the available backend's generated `architectures.txt` and minimum Driver API family.
CUDA 12 requires API 12.0 and CUDA 13 requires API 13.0. Selection prefers CUDA 13 on overlapping
architectures. `--cuda-backend auto|12|13` overrides the choice. The launcher never loads a second
backend after the first one fails. One process cannot mix a Volta-only and Blackwell-only backend.
Router mode reads the existing model catalog and checks the union of its GPU ordinals, respecting
command-line precedence over presets. The n-gram profiler is a CPU command and does not query GPU
compatibility. Persisted prefix caches identify the module that owns inference code, including its
backend directory, file size and modification time, rather than the small launcher.

The packagers capture real command help into `help/`. Shipped `--help` and `--version` need no
backend or driver. CI validates architecture images in the runtime modules, dependency closure,
startup and archive contents. These checks do not qualify GPU execution or performance.
The shared runtime changes Linux archives to PIC, so affected objects need an initial rebuild.
Existing HF snapshots remain available. Windows keeps the original CUDA object compile owners.

## Build caches

The manual `Linux cache bootstrap` workflow uses the same reusable `native-linux.yml` job as
Native release. Register two temporary Debian 13 x64 runners with label `ninfer-native-debian13`
and work directory `/__w` before dispatch. Each machine runs one CUDA variant at the hosted path
`/__w/ninfer-all/ninfer-all`. Both variants can compile and upload independently. The compiler
process limit is at most 32 and is reduced for the container's CPU quota, affinity and RAM limit.
The workflow supplies the existing bucket credential. Keep compiler versions and flags unchanged
to preserve cache compatibility. Remove the runner registrations and rentals after verifying the
archives and HF snapshots. Native release also accepts `linux_runner: ninfer-native-debian13`;
its default remains GitHub-hosted runners in Debian 13 containers.

The shared-runtime bootstrap writes `native-linux-cuda-<toolkit>-shared-v1`, leaving any older
`host-v2` producer running without a conflicting writer. Both artifact workflows serialize writes
per branch and Toolkit. They restore the existing `host-v2` family when the new family is absent,
so completed compatible PIC objects remain reusable. This bootstrap does not cancel Windows jobs.

Linux native builds and CUDA CI set `LANG`, `LC_ALL` and `LC_CTYPE` to `C.UTF-8`.
Ccache hashes these variables separately, even when they select the same effective locale.
The explicit values keep rented runners and hosted containers compatible. Entries created with
different locale variables require one rebuild; the resulting entries can then be shared.

Manual runs can select `platforms: linux` or `windows` for a targeted build without publication.
After all Windows object shards have succeeded, select `platforms: windows` and
`windows_phase: package` to retry only the two final jobs. Each final job requires all eight HF
shard snapshots, verifies their archives, and builds against the selected source revision before
packaging. Missing snapshots fail restoration; changed inputs rebuild through ccache normally.
The default `windows_phase: full` builds the shards first.
Publication requires `platforms: all`. Starting any new native run on the same branch cancels the
whole previous run, including its other platform. The cleanup wait preserves its completed caches.

Native Windows, Debian and CUDA CI use pinned ccache 4.14.2. Native release and its Linux artifact bootstrap write compiler
snapshots. Only Native release writes Windows vcpkg snapshots to the private HF bucket `WaveCut/ninfer-cache`, under `github-actions/`.
`HF_BUILD_CACHE_TOKEN` has read/write access to this bucket only. Native jobs use it for snapshots;
CI reuses it only in its restore step, by owner decision. The token itself can write, but CI has no
upload steps, receives no token during compilation, and the save command rejects CI jobs. Existing
pod caches and model files use separate prefixes and are not changed. The client is pinned to `huggingface_hub==1.17.0`.

CI uses Debian 13, the same official Toolkit installation, pinned ccache, Release flags, shared-runtime/PIC
settings, architecture sets and `build-release` directory as the Linux artifact build. The shared
configure script enables additional test targets for CI. Both CUDA variants compile all targets;
CTest runs without a GPU and device tests skip. Windows host checks also use the two complete
architecture sets. Test-only objects compile locally and are discarded after CI; they do not
feed back into the production cache. Broader test builds can take longer on a cold cache.

Each native or CI variant has an 8 GiB local compiler-cache limit. This is a capacity limit, not a
measured working-set size; ccache can evict older objects. Each native platform/toolkit has its own
family, with separate branch prefixes and two rotating snapshots. CI reads the corresponding
`native-linux-cuda-<toolkit>-shared-v1` family. It takes the latest available verified snapshot without waiting
for an artifact build to finish. It tries its branch, then master; a PR starts from its target branch.
Source commits are not cache-family boundaries. Missing credentials, unavailable storage or rejected
snapshots produce a warning and allow a cold CI build. Fork PRs receive no repository secret.
The `host-v2` families leave the previous families untouched and can read their entries when
the new family is absent. Corrected compiler lookup adds the CUDA host compiler's content to
the cache key. Previous CUDA entries therefore miss once; unchanged C++ entries remain eligible.
The old snapshots are retained, but retaining them does not make the old CUDA keys compatible.

Windows uses eight object shards per Toolkit, each with a separate
`native-windows-cuda-<toolkit>-host-v2-shard-<index>` family. The final Windows job merges those
snapshots, links and packages the same multiarch executables, and saves the aggregate family
without the shard suffix. Only shard zero and the subsequent final job write the shared vcpkg
family. The reusable Windows workflow remains part of Native release; CI has no cache writer.
To bootstrap master from a completed branch build, set repository variable
`HF_BUILD_CACHE_SEED_REF` to that branch's full ref, such as `refs/heads/build/windows-release`.
Readers try this explicit seed only after their branch and master have no usable snapshot.
They verify its checksum and do not write to the seed. After master saves its own snapshot,
that snapshot takes precedence. This avoids overwriting master's cache from a feature branch.

CI does not upload compiler caches to HF or Actions Cache. Python dependency caches remain in
Actions Cache, within the repository's free 10 GiB limit. Docker publication downloads the compiled
Linux archive from the same run after both Linux jobs succeed, checks its commit, CUDA version,
architecture set and checksums, and packages it without invoking a compiler or reading HF caches.
Local source Docker builds retain their BuildKit cache mounts.

Each Linux artifact job also uploads its archive volumes, checksum manifest and build metadata
to `WaveCut/ninfer-cache/native-artifacts/v1/iamwavecut/ninfer-all/<run>/<artifact>/`.
It downloads each uploaded file on the runner, checks SHA-256, and publishes `COMPLETE.json`
only after all checks pass. This step allows sixty minutes for upload and download verification
of the multi-gigabyte archives. These private backups do not publish a GitHub release.

Ccache checks compiler content, source, headers and compilation flags before reusing an entry.
Depend mode hashes raw source and compiler-reported header dependencies, without falling back to
device-only preprocessing. The pre-build probe checks unchanged C++ and multiarch CUDA objects,
then verifies that changing host-only CUDA code or a host-only header causes a cache miss.
Compilation reports cache counters, available RAM and disk space every minute.

Successful compiler invocations commit their entries to local disk. During native artifact builds, the
supervisor uploads a snapshot every ten minutes after the previous upload completes. Windows
configuration also checkpoints completed vcpkg archives. Uploads run while compilation continues;
an entry that finishes after the snapshot starts is retained by a subsequent snapshot. Temporary
files and mutable cache statistics are excluded. An upload writes the inactive archive slot,
then publishes its manifest with size and SHA-256. Readers verify the archive and try the previous
slot if the latest archive is damaged. A partial upload does not publish a new manifest.

After success, failure or ordinary cancellation, cleanup stops compiler work and saves completed
objects before packaging. Native builds stop after 300 minutes of compilation, or earlier at a
deadline set 330 minutes from the first job step. CUDA CI uses the same compilation limit and
deadline, reserving the remaining time for tests and diagnostics. The job limit is 350 minutes.
The supervisor exits with code 124 and leaves cleanup time; the hard step timeout is five minutes longer than its duration limit.
Cancellation first sends a console interrupt, with up to four seconds for the build parent to exit.
It then stops all remaining compiler descendants, even if that parent has already exited. POSIX
uses the retained process group; Windows uses a dedicated Job Object that owns the entire tree.
`Build workflow checks` tests cancellation on Windows and release tooling on Linux before native
compilation. It can also be dispatched separately without starting compilers or accessing HF.
Native save steps also run after a hard compilation-step timeout. A new native run waits for older
runs of the same workflow and branch to finish cleanup before it can write snapshots. This allows
the cancelled producer to flush completed objects before handing ownership to the new producer.
The cache command rejects writes from other workflows/jobs and prevents an older run from replacing
snapshots already saved by a newer run. CI cancels immediately and has no shared-cache cleanup.

The final save has a twenty-minute limit after normal completion or failure. Cancellation reduces
that limit to three minutes within GitHub's five-minute cancellation window. A measured 6.2 GB
Linux snapshot took 183 seconds to upload on the bootstrap rental, excluding archive creation;
such a transfer does not fit the cancellation limit. Completed periodic snapshots remain available.
Runner
loss, force cancellation or a failed upload can lose work since the last completed checkpoint.
That interval can exceed ten minutes by the upload duration or by a failed checkpoint. An executing
compiler process itself cannot resume. Periodic checkpoint failures produce a warning and retry;
final-save failures fail the step. Retention, transfer duration and useful reuse must be measured
on the complete CUDA working set before claiming that every matrix job will finish.

Migration preserves existing Actions caches under `github-actions/migration/`. Each archive is
uploaded and downloaded on a GitHub runner, and its manifest is published only after a matching
SHA-256. Restore can merge these preserved snapshots to seed the new cache family. Existing
single-architecture Ubuntu pod caches are not assumed compatible with Debian multiarch builds.

## Verification status

The four-variant native run on 20 October reached the 300-minute compilation limit on every runner:
Windows CUDA 12.9 completed 531/789 Ninja steps, Windows CUDA 13.4 completed 559/816,
Linux CUDA 12.9 completed 542/793, and Linux CUDA 13.4 completed 564/820. No compiler failures
were recorded in the final sccache counters; no release archive was produced.
Each variant reported 4,304–4,594 successful cache writes, with 12, 12, 96 and 70 write errors respectively.
On 21 October, the repository cache limit was 10 GiB and only 2,242 entries for the test branch remained.
These counts show that successful writes alone do not establish retention or useful reuse on the next run.
The new telemetry must establish useful reuse before claiming that the full matrix can finish.
The retired compiler-cache recovery workflow tested transport with a single C++ object on fresh
Windows and Linux runners. It was removed when Native release became the only shared-cache writer;
its stored snapshots and historical runs remain. These probes test recovery, not retention of the
full CUDA working set. Before the HF migration on 21 October 2026, both timeout and cancellation
modes saved their Actions caches on Windows 2022 and Ubuntu 24.04.
Separate restore runs on fresh runners each recorded one compiler cache hit and zero misses:
[after timeout](https://github.com/iamwavecut/ninfer-all/actions/runs/39569140529) and
[after cancellation](https://github.com/iamwavecut/ninfer-all/actions/runs/39569242802).
The producers were [39568919978](https://github.com/iamwavecut/ninfer-all/actions/runs/39568919978)
and [39568972360](https://github.com/iamwavecut/ninfer-all/actions/runs/39568972360).
The HF migration preserved 13 snapshots (4,084,951,040 archive bytes), including the final caches
from all four cancelled native variants and all three CUDA CI architectures. Every migration
uploaded and downloaded its archive on the runner and verified SHA-256 before publishing metadata.
The first native-Linux restore missed because its Debian archive compression differed from the
Ubuntu runner; restoring in the original Debian environment resolved it. The Actions source
caches were retained. Migration records remain under `github-actions/migration/`; the temporary
migration workflow was removed after verification.

The HF [timeout producer](https://github.com/iamwavecut/ninfer-all/actions/runs/39584240411)
saved both periodic and final snapshots. Its [fresh-runner restore](https://github.com/iamwavecut/ninfer-all/actions/runs/39584567971)
recorded one hit and zero misses on Windows and Linux. A second
[restore](https://github.com/iamwavecut/ninfer-all/actions/runs/39584614325) recorded the same result
while its [producer](https://github.com/iamwavecut/ninfer-all/actions/runs/39584404425) was still
running, demonstrating that recovery does not depend on the job reaching cleanup. These probes
use a ten-second checkpoint interval; production uses ten minutes. They do not measure multi-GiB
upload times or full CUDA working-set retention. Both final saves also succeeded after ordinary
cancellation; the [subsequent restore](https://github.com/iamwavecut/ninfer-all/actions/runs/39585023434)
again recorded one hit and zero misses on each OS.

In the first [native HF run](https://github.com/iamwavecut/ninfer-all/actions/runs/39584883032),
Windows CUDA 13.4 restored its objects in six seconds and its vcpkg archives in seven seconds.
Configuration took 30 seconds; the first production-build sample after one minute recorded 244
hits and zero misses, reusing every object retained from the cancelled Windows build. The earlier
migration of the 1,082,204,160-byte Linux CUDA 12.9 archive took 8.7 seconds for upload, download
and SHA-256 verification; the 548,229,120-byte Windows CUDA 12.9 archive took 19.4 seconds.
These transfer measurements exclude archive creation and do not predict a full cache's duration.

The former single-architecture CI run
[39584887267](https://github.com/iamwavecut/ninfer-all/actions/runs/39584887267) compiled all three
CUDA variants but failed its final HF saves with `PermissionError`; CTest was consequently skipped.
Periodic snapshots had succeeded. Removing CI writers removes that final-save path. In the replacement
[CI run](https://github.com/iamwavecut/ninfer-all/actions/runs/39608375823), CUDA 13.4 restored the
native snapshot in 16 seconds. Its first minute recorded 564 direct cache hits and zero completed
misses, while the native producer was still running. Compilers still in flight are not included in
those counters.

The complete dual-CUDA [CI run](https://github.com/iamwavecut/ninfer-all/actions/runs/39689626725)
restored the corrected `host-v2` snapshots on fresh GitHub-hosted Debian containers, compiled every
target and completed CTest. Each variant passed
90 tests and skipped 247 GPU tests, with no failures. This required recognizing
`cudaErrorStubLibrary` as a no-GPU condition; the preceding run failed four tests per variant on
that message. Final ccache counters were 768 hits and 344 misses for CUDA 12.9, and 794 hits and
344 misses for CUDA 13.4, including cache probes. CI does not save its test-only cache entries.

[Native run 39664338995](https://github.com/iamwavecut/ninfer-all/actions/runs/39664338995)
produced both Linux archives, verified their HF backups by downloading and hashing them, and
uploaded both GitHub artifacts. The dependent Docker jobs verified those archives and built and
loaded both images without compilation. Registry cache imports found no existing CUDA tags, so
these image builds used no registry layer cache. This test run did not publish images or a release.

All 16 Windows object shards succeeded and saved their HF snapshots. Both final jobs restored
all eight required snapshots but stopped at the cache probe: its run-wide source identity reused
an earlier job's modified probe object and mistook that valid cache hit for failed invalidation.
The probe now generates fresh source content for each invocation. In the
[package-only retry](https://github.com/iamwavecut/ninfer-all/actions/runs/39728471200), both variants
passed the probe twice, restored their compiled objects and reached linking in under 90 seconds
of build time. Both failed with `LNK1248`: their images exceeded the 2 GiB linker limit. Both final
aggregate caches were saved. The [PTX-pruning retry](https://github.com/iamwavecut/ninfer-all/actions/runs/39729876288)
retained all configured machine-code images and linked all three CUDA 12.9 executables. Packaging
then stopped because the component installer omitted the Toolkit's root `EULA.txt`. The workflow
now extracts the licence from the exact Toolkit release's official CUDA runtime archive, after
checking its SHA-256 against NVIDIA's redistribution manifest. In the
[complete-link retry](https://github.com/iamwavecut/ninfer-all/actions/runs/39731382003), CUDA 13.4
exceeded the limit in all three executables: benchmark `80D58000`, CLI `80DCC000`, server `80F92000`
(hexadecimal image sizes, limit `80000000`). The workflow now links GGML kernels as a separate
DLL, retaining all GPU targets.
The [first object-library DLL retry](https://github.com/iamwavecut/ninfer-all/actions/runs/39736621855)
compiled CUDA 13.4 objects but failed DLL linking with an unresolved `current_device_tier()` call
from GGML FlashAttention. Its final cache was saved. The device-tier probe and cache now belong
to that DLL in Windows DLL builds, with both GGML and static Ops using the exported functions.
The earlier complete-link retry's CUDA 12.9 job found all configured device images in `ninfer.exe`,
but Windows refused to start it with "not a valid application for this OS platform". The DLL build
no longer reproduces that failure. Packaging records module sizes and PE headers when `dumpbin`
is available.

In [run 39741764110](https://github.com/iamwavecut/ninfer-all/actions/runs/39741764110), both variants
linked all three executables and the kernel DLL. CUDA 13.4 took 96 seconds for the build step,
with 789 direct cache hits and four misses; CUDA 12.9 took 105 seconds with 763 hits and four
misses. These timings exclude
Toolkit installation, cache transfers and packaging. Fixing the compile PDB path preserved
GGML's cache identity when moving from a static archive to the object library used by the DLL.
Both variants verified all five configured device images in each module, passed all three
`--help` checks with only Windows on `PATH`, verified their four-volume archives and uploaded
them: CUDA 12.9 is 6.29 GB and CUDA 13.4 is 6.37 GB. The uploaded artifact directories and SHA-256
manifests were read back; no full local download was used. CI is the authority for subsequent
artifact checks; do not repeat archive downloads or validation outside CI without a concrete
failure or a user request. GPU inference and performance remain untested by these jobs; no
release or image was published. The large archive size remains a limitation: all three
executables statically include the main CUDA code, so only the GGML portion is shared through
the DLL.

The preceding cancelled CUDA 12.9 job stopped compiler work and then saved its 6.69 GB final
snapshot in 87 seconds. The replacement run restored that exact snapshot. Completed object
entries survive cancellation; an unfinished translation unit must compile again. Packaging now
uses the process monitor with cache writes disabled, reports hashing and startup stages, and
enables 7-Zip progress output. These diagnostics were added after the validated build's source
revision and passed the Windows workflow checks.

## Local packaging

Both packagers read CUDA and architecture labels from the configured build tree. Set
`NINFER_BUILD_ROOT` to that tree. Build release products with benchmarks enabled. Linux accepts
separate CLI/server binaries or `NINFER_MULTICALL=ON`; CI uses the latter. Run
`scripts/package-release.sh` on the Linux build host, or `scripts\package-release.ps1` on Windows.
The repository's `VERSION` is included as source metadata, with matching release notes when present. Neither
packager uses `VERSION` in the archive name. Modified local sources add `-dirty`; CI rejects them.
Linux packaging needs Python 3.11, `strip`, `patchelf`, `ldd`, `dpkg-query`, `tar` and the build's CUDA Toolkit.
The runtime Dockerfile is included only when packaging a Debian 13 build.

## Gotchas

**Always pass `-R iamwavecut/ninfer-all` to `gh`.** The checkout also has the `upstream` and
`ashalliants` remotes, and `gh` without `-R` may resolve to another fork.

**Windows executables import cuBLAS, and the archive must carry it.** The cuBLAS prefill route makes
every executable import `cublas64_<major>.dll` (which imports `cublasLt64_<major>.dll`), so a machine without
the CUDA Toolkit cannot start any of them (`0xC0000135`), even when `--prefill-cublas` is never used.
`package-release.ps1` copies both DLLs from the build compiler's Toolkit and refuses to produce an archive whose
executables do not start with only Windows on `PATH`. Do not weaken that check.

**Redistributing NVIDIA's DLLs has conditions.** The CUDA Toolkit EULA (Attachment A) lists the
cuBLAS DLLs as redistributable. The grant requires that the application has material functionality
of its own, that the DLLs are used only by it, and that the distribution terms are consistent with
the EULA and protect NVIDIA's rights. The packager therefore ships the toolkit's `EULA.txt` as
`NVIDIA-CUDA-EULA.txt`, and the archive README names the two DLLs as NVIDIA's, outside the
project's Apache-2.0 licence. Re-read the EULA of the toolkit that builds a release.

**Linux links cuBLAS, FFmpeg and glibc dynamically.** The packager copies the CUDA runtime closure
into `lib/`, checks the executable loader path, and ships NVIDIA's EULA. It lists system libraries'
Debian packages in `packages/multi.txt`; the runtime Dockerfile installs those without adding a
CUDA APT repository. An unowned system library or unresolved dependency fails packaging.

**Asset sizes.** Every executable links every device image. Native packages use 1900 MiB archive
volumes to remain below GitHub's 2 GiB asset limit. This does not remove MSVC's linked-image limit
or the hosted runner's memory and build-time limits. All three CUDA 13.4 Windows executables
exceeded the linker limit after PTX pruning; see the kernel DLL change and validation record above.

**Smoke-test the launcher default on the real card.** A desktop holding 2-3 GiB of a 24 GB card
refuses the default DFlash2 profile, both on the engine's runtime reservation and, on Windows, on
pinned host state. `run.{sh,bat}` step down by themselves when startup is refused for lack of GPU
memory ([launcher profiles](launcher-profiles.md)). Start the launcher with nothing overridden and
confirm it serves a request.

## Checks before publishing

- `scripts/check-linux-scripts.sh` passes.
- Every package: files match the inner `SHA256SUMS.txt`, and the outer checksum manifest verifies.
- Windows: all six command executables start with only Windows on `PATH`.
- Linux: a fresh unpack runs `./ninfer-serve --help` and `./run.sh --help`.
- The default launcher profile starts and serves a chat completion on a real 24 GB card.

## CI change scopes

`.github/scripts/build-inputs.sh` and `change-scopes.py` select individual CI jobs. Empty or
Markdown-only PRs run the short change detector and skip the check/build jobs. Python tooling,
shell syntax, launcher behavior, calculator tests and release tools have separate scopes. C++/CUDA
test changes run CI compilation without repackaging native binaries. Runtime/package input changes
can rebuild native packages without repeating the independent CI CUDA build. Missing comparison
metadata fails the decision job without admitting builds. Manual dispatch explicitly enables every
scope and supports an initial build before a published comparison revision exists.

CI still has a separate CUDA compilation/CTest matrix when its compiled inputs change. Moving CTest
into the native build tree to compile once is a separate, unimplemented change.
