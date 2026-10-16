# ninfer server 编译与启动记录（Windows / RTX 5060 Ti sm_120a）

| 项目 | 值 |
|---|---|
| 目标 | `backends/ninfer`（pin `1383a50e` + `backends/patches/ninfer-kvmem.patch`，含 ActKV 移植） |
| 构建脚本 | `scripts/windows/build-backends.ps1 -Backend ninfer` |
| 工具链 | VS 2022 Professional 14.44.35207 + CUDA 13.4.59 + CMake 4.3.1 + Ninja 1.13 |
| 依赖 | vcpkg x64-windows：`ffmpeg[zlib] 9.0.2`、`curl 8.22.0`、`pkgconf 3.0.3`（树在 `C:\vcpkg`） |
| 卡 | RTX 5060 Ti 16 GB（`GPU-dba26dd9-930f-1b01-2b05-22c1fca6cdce`），`-DCMAKE_CUDA_ARCHITECTURES=120a` |
| 首次记录 | 2026-10-10（pin `8319e8f5`） |
| 最近重建 | 2026-10-17（pin `1383a50e`，vcpkg 搬到 `C:\vcpkg`，见 §8） |

---

## 1. 为什么必须先装 vcpkg 依赖

`backends/ninfer/cmake/Dependencies.cmake` 在 Windows 上**硬性要求**一个预置的 vcpkg triplet 树：

```cmake
if(WIN32 AND NOT VCPKG_TOOLCHAIN AND DEFINED ENV{VCPKG_ROOT} AND DEFINED ENV{VCPKG_TARGET_TRIPLET})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{VCPKG_ROOT}/installed/$ENV{VCPKG_TARGET_TRIPLET}")
endif()
if(WIN32 OR DEFINED VCPKG_TARGET_TRIPLET)
  find_package(FFMPEG REQUIRED)   # cmake/FindFFMPEG.cmake 读 $ENV{VCPKG_ROOT}/installed/$ENV{VCPKG_TARGET_TRIPLET}
```

`FindFFMPEG.cmake` 只认 `$VCPKG_ROOT/installed/$VCPKG_TARGET_TRIPLET/{include,lib}` 里的
`libavformat/libavcodec/libavutil/libswscale`；curl 走 vcpkg 的 CMake 包。**没有 vcpkg 就配置不过**。

## 2. 三个坑（都实际踩到过）

### 2.1 `bootstrap-vcpkg.bat` 下载失败
`scripts/tls12-download.exe` 在本机 TLS 环境下拉不下 `vcpkg.exe`。改用 Git 自带的 OpenSSL 版 curl：

```powershell
$gcurl = 'D:\git\Git\mingw64\bin\curl.exe'      # 系统 curl.exe 走 schannel，同样失败
& $gcurl -L -o I:\vcpkg\vcpkg.exe `
  https://github.com/microsoft/vcpkg-tool/releases/download/2026-09-26/vcpkg.exe
Set-Content -Value "" -Path I:\vcpkg\vcpkg.disable-metrics -Force
```

### 2.2 显式设 `VCPKG_VISUAL_STUDIO_PATH` 反而失败
设置后 vcpkg 报 `Unable to find a valid Visual Studio instance`（尽管该 VS 的 `vcvarsall.bat` 完全可用）。
**不要设这个变量**，让 vcpkg 自己用 vswhere 找（它会正确选中 VS 2022 Professional 的 14.44 工具集）。

### 2.3 VS 的 dev shell 会覆盖 `VCPKG_ROOT`（关键）
`vcvarsall.bat` 会设置 `VCPKG_ROOT=<VS>\VC\vcpkg`（VS 自带、**没有安装任何包**）。
而 `build-backends.ps1` 在 `$env:VSCMD_VER` 为空时会自己调用 `Launch-VsDevShell`，
于是我们设的 `VCPKG_ROOT=I:\vcpkg` 被覆盖 →
`FindFFMPEG.cmake` 去找 `d:\...\VC\vcpkg/installed/x64-windows/lib`，报 `missing swscale.lib`。

**正确做法**：自己先把 vcvars 导进当前进程（这同时设置了 `VSCMD_VER`，脚本就会跳过它自己的 dev shell），
然后再覆盖 `VCPKG_ROOT`：

```powershell
$vcvars = 'D:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat'
cmd /c "call `"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}
$env:VCPKG_ROOT = 'I:\vcpkg'          # 必须在导入之后
$env:VCPKG_TARGET_TRIPLET = 'x64-windows'
```

## 3. 安装依赖

```powershell
$env:VCPKG_ROOT='I:\vcpkg'; $env:VCPKG_DEFAULT_TRIPLET='x64-windows'; $env:VCPKG_MAX_CONCURRENCY='12'
I:\vcpkg\vcpkg.exe install ffmpeg[zlib]:x64-windows curl:x64-windows pkgconf:x64-windows
```
实测 **18 分钟**（其中 ffmpeg 源码编译 17 分钟；curl 1.5 分钟）。
注意本机有代理 `127.0.0.1:7890`，vcpkg 会自动使用；msys2 镜像偶尔 TLS 失败会自动换源重试。

> `bootstrap-vcpkg.bat` 失败后**不需要**再跑它；手动放好 `vcpkg.exe` 即可。

## 4. 编译

```powershell
# 在导入了 vcvars 并覆盖 VCPKG_ROOT 的同一个 PowerShell 里
Set-Location I:\llama\kvmem-llama.cpp-multi-backend-framework
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\build-backends.ps1 `
    -Backend ninfer -CudaPath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4' -Jobs 10
```

脚本会依次：`prepare-backends.py --backend ninfer`（拉子模块 + 校验并应用补丁）→
CMake 配置（`-DNINFER_KVMEM_SOURCE_DIR=<repo>/kvmem`、`-DNINFER_BUILD_APPS=ON`、`120a`、`NINFER_SM120_NATIVE=ON`）→
构建目标 `ninfer-serve`。

产物：`build-backends/ninfer/apps/ninfer-serve.exe`（约 0.5 GB，含整张 kernel image）；
`ninfer_stage_runtime_dlls` 会把 vcpkg 的 FFmpeg/curl DLL 一起铺到该目录。
（10/10 时这个路径是 `apps/serve/`，后来上游挪到了 `apps/`，以实际构建输出为准。）

## 5. 放到启动器期望的位置

`scripts/windows/start-backend.ps1` 与 `start-kvmem-ninfer.bat` 找的是
`workers/ninfer/infer-kvmem-server.exe`（发行包的命名），所以编译后需要 staging：

```powershell
$src = 'build-backends\ninfer\apps'
$dst = 'workers\ninfer'
New-Item -ItemType Directory -Path $dst -Force
Copy-Item "$src\ninfer-serve.exe" "$dst\infer-kvmem-server.exe"
Copy-Item "$src\*.dll" $dst                    # FFmpeg/curl/z 已由 CMake 铺好
Copy-Item 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\cublas64_13.dll'   $dst
Copy-Item 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\cublasLt64_13.dll' $dst
```

## 6. 启动（参数）

`start-kvmem-ninfer.bat` 已按本机配置好：模型 `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.ninfer`、
GPU UUID `GPU-dba26dd9-…`、端口 18200、`-Budget 36864 -Reserve 16384 -HostMiB 12288`、
`-KvType int8`、`-MtpDrafts 4 -AdaptiveMtp -NgramDrafts 31`、Vision（CPU 驻留）、
以及 ActKV 四个标志 `--kvmem-action-lrfu --kvmem-lrfu-lambda 0.50 --kvmem-lrfu-top-frac 0.90 --kvmem-score-agg max`。

ninfer 侧与 llama 侧的差异（`docs/multi-backend-differences.md`）：page/block **固定 64 token**；
**没有 `--kvmem-method`**；KV 类型名是 `int8/bf16/fp8/nvfp4/k8v4/rk8v4/...` 而不是 `q8_0`；冷盘快照禁用。

## 7. 启动测试结果（2026-10-10）

| 项目 | 状态 |
|---|---|
| 编译 | ✅ `ninfer-serve.exe` **512.5 MB**，sm_120a / CUDA 13.4 / VS 2022 14.44，构建脚本 exit 0 |
| ActKV 是否进二进制 | ✅ `--help` 列出四个参数；非法值（`lrfu-lambda 1.5`、`score-agg bogus`）被范围校验拒绝 |
| staging | ✅ `workers/ninfer/infer-kvmem-server.exe` + FFmpeg/curl DLL + `cublas64_13.dll` / `cublasLt64_13.dll`（cublas 需从 `CUDA\v13.4\bin\x64\` 取，`bin\` 下没有） |
| **通过 bat 启动** | ✅ 模型加载 11.3 GiB / 5.3 s，CUDA graphs ready，`listening on http://0.0.0.0:18200` |
| ActKV 参数是否到达引擎 | ✅ `/v1/models` 的 `status.args` 回显了 `--kvmem-action-lrfu --kvmem-lrfu-lambda 0.50 --kvmem-lrfu-top-frac 0.90 --kvmem-score-agg max` |
| **KVMem 长上下文检索** | ✅ 见下 |

### 发现并修掉的一个真 bug：`--kvmem-score-agg max` 被误拒

第一次启动直接 `Server exited with code 1`，日志第一行：

```
ninfer-serve: --kvmem-score-agg must be mean or max
```

原因：ninfer 的 `require_value()` 返回 `const char*`（`src/serve/serve_options.cpp:614`），
而移植时写成了 `const auto mode = require_value(...)`，于是 `mode != "max"` 变成**指针比较**，
任何取值都被判为非法。ninfer 项目自己的写法是先绑定 `std::string_view`。

已改为 `const std::string_view mode = require_value("--kvmem-score-agg");`，
重新生成补丁（`patched_tree 96659b14`、`patch_sha256 eec4019c`）并重编。
llama.cpp 侧不受影响：那里的 `eq()` 用的是 `std::strcmp`。

### 功能测试（真实模型，RTX 5060 Ti）

短请求：

```
POST /v1/chat/completions  reasoning_effort=none
  回复 "NINFER ACTKV OK"，7 completion tokens，0.5 s
```

长上下文检索（把金库密码放在上下文**开头**，后面 ~470k 字符填充）：

```
prompt_tokens = 101,049        # KVMem 预算只有 36,864，超了近 3 倍
回复           = ZX-4471-QQ    # 开头那根针被正确召回
finish_reason  = stop, 9 completion tokens, 268.5 s
```

启动时 `capacity | KV 53,248 tokens, int8, explicit | pages 832/832 | runtime 2.50 GiB`
= `--kvmem-budget 36864 + --kvmem-gen-reserve 16384`，与 bat 一致。

### 运行状态与停止

测试结束时服务仍在运行：进程 `infer-kvmem-server`（PID 因会话而异），
显存约 15.6 GiB / 16.3 GiB，主机 RAM 约 11 GB。停止：

```powershell
Get-Process infer-kvmem-server | Stop-Process
```

### 仍未验证

* **ActKV 开/关的收益没有做 A/B**。单次长上下文针在两种设置下都可能召回到（和历史数据一致：
  单针在预算充足时饱和），LRFU 的收益要在**多轮**会话里才显现；需要两臂各跑一遍多轮脚本。
* `--log-level debug`（或 KVMEM 的 trace 环境变量）下逐次换页/检索日志没有采集，因此
  "选了哪些块、换出多少" 这一层没有直接观测。

---

## 8. 2026-10-17 重建（上游线 + vcpkg 搬迁）

base 从 `8319e8f5` 换成上游 `1383a50e`（Turing/Volta RC，含 `sm_70`/`sm_75`，即 20 系支持），
本地 ActKV 移植随之上线；`kProtectRescueFraction` 由 `0.25` 调为 `0.15`。

### 8.1 vcpkg 从 `I:\vcpkg` 搬到 `C:\vcpkg`

搬到 `C:\vcpkg` —— 上游 `build-native.bat` 里 `NINFER_VCPKG_ROOT` 的默认值 —— 并把
`VCPKG_ROOT` / `VCPKG_TARGET_TRIPLET` 写成**用户级环境变量**。搬迁是安全的：`installed/` 下的
`.cmake` 用 `${CMAKE_CURRENT_LIST_DIR}/../../` 推导前缀，全树 0 处硬编码旧盘符（迁移后已验证），
`robocopy /MOVE` 61,439 个文件 / 2.215 GB 零失败。

### 8.2 §2.3 那个坑现在由脚本自己兜住

`build-backends.ps1` 在 `Launch-VsDevShell` **之后**补了一段：若 `VCPKG_ROOT` 形如 `*\VC\vcpkg`
（即刚被 vcvars 覆盖成 VS 自带的空树），就从机器级/用户级环境变量把它还原。于是不再需要
本文 §2.3 那套"手动 vcvars + 再覆盖 `VCPKG_ROOT`"的仪式，直接跑 §4 的命令即可。配置阶段的证据：

```
-- Found FFMPEG: C:\vcpkg/installed/x64-windows/include
-- Found CURL:   C:/vcpkg/installed/x64-windows/share/curl/CURLConfig.cmake (8.22.0)
```

### 8.3 结果

| 项目 | 值 |
|---|---|
| 编译 | exit 0，814 个目标，00:20 → 01:03（约 43 分钟，`-Jobs 10`） |
| 产物 | `build-backends/ninfer/apps/ninfer-serve.exe`，**538 MB**（10/10 那次是 512.5 MB） |
| 二进制自述 | `build 0.12.0-rtx3090+1383a50e-dirty` |
| ActKV 参数 | `--help` 列出 `--kvmem-lrfu`、`--kvmem-protect-semantics`；`--kvmem-action-lrfu` 已变成直接报错的改名陷阱（`serve_options.cpp:723`） |
| staging | `workers/ninfer/infer-kvmem-server.exe`，旧版备份为 `infer-kvmem-server.exe.1014-bak` |
| DLL | vcpkg 树与旧版同名同版本（`avcodec-63`/`avformat-63`/`avutil-61`/`swresample-7`/`swscale-10`/`libcurl`/`z`）；cublas 从 `CUDA\v13.4\bin\x64\` 取（`bin\` 下没有） |

启动后设备档案会提示一次不匹配并自动重标定（正常）：

```
WARN engine | device-profiles.json: the entry for nvidia-geforce-rtx-5060-ti-sm120 was calibrated
against route catalog ac2d7b3e311831b0 (build 8319e8f-dirty), this build's is 5893cf72e3d60377;
it is ignored. Recalibrate (ninfer-calibrate, or --device-profile calibrate)
```

