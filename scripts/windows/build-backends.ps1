#Requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet('core', 'llamacpp', 'ninfer', 'all')][string]$Backend = 'core',
    [string]$BuildRoot,
    [string]$Python = 'python',
    [string]$CudaPath = $env:CUDA_PATH,
    [string]$CudaArchitectures = '120a',
    [ValidateRange(1, 64)][int]$Jobs = 8,
    [string[]]$CMakeArgs = @(),
    [string]$WebUiDir,
    [switch]$ConfigureOnly,
    [switch]$BuildOnly,
    [switch]$SkipMsvcIncludePrefix,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($ConfigureOnly -and $BuildOnly) { throw 'Choose ConfigureOnly or BuildOnly, not both.' }
$source = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$BuildRoot) { $BuildRoot = Join-Path $source 'build-backends' }
$BuildRoot = [IO.Path]::GetFullPath($BuildRoot)
$backends = if ($Backend -eq 'all') { @('llamacpp', 'ninfer') } else { @($Backend) }
$plans = foreach ($name in $backends) {
    $build = Join-Path $BuildRoot $name
    $tree = if ($name -eq 'ninfer') { Join-Path $source 'backends/ninfer' } else { $source }
    $configure = @('-S', $tree, '-B', $build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release')
    $target = @()
    if ($name -eq 'core') {
        $configure += @('-DKVMEM_BUILD_LLAMA=OFF', '-DKVMEM_BUILD_SERVER_TESTS=OFF')
    } elseif ($name -eq 'llamacpp') {
        $configure += @('-DKVMEM_BUILD_LLAMA=ON', '-DGGML_CUDA=ON',
            '-DGGML_CUDA_FA_QUANTS=all', '-DBUILD_SHARED_LIBS=OFF',
            "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures")
        $target = @('--target', 'llama-kvmem-server')
    } else {
        $configure += @("-DNINFER_KVMEM_SOURCE_DIR=$source/kvmem", '-DNINFER_BUILD_APPS=ON',
            '-DNINFER_BUILD_BENCHMARKS=OFF', "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures")
        if ($CudaArchitectures -eq '120a') { $configure += '-DNINFER_SM120_NATIVE=ON' }
        $target = @('--target', 'ninfer-serve')
        # WebUI：把 llama.cpp 那套 WebUI 编进 ninfer-serve（由 GET / 提供）。不显式传 -WebUiDir 时
        # 自动取 build/share/kvmem/ui —— 也就是 scripts/windows/build.ps1 构建 llama.cpp 后端时
        # 放好的那份。两者都没有就**不传该开关**：那样 exe 不含 WebUI，启动日志里也不会有
        # `webui http://…` 那一行（CMake 侧 NINFER_WEBUI_DIR 默认为空，这是合法的）。
        $webui = $WebUiDir
        if (!$webui) {
            $candidate = Join-Path $source 'build/share/kvmem/ui'
            if (Test-Path -LiteralPath (Join-Path $candidate 'index.html')) { $webui = $candidate }
        }
        if ($webui -and
            !(Test-Path -LiteralPath (Join-Path $webui 'index.html')) -and
            !(Test-Path -LiteralPath (Join-Path $webui 'index.html.gz'))) {
            throw "WebUiDir has no index.html: $webui"
        }
        if ($webui) { $configure += "-DNINFER_WEBUI_DIR=$webui" }
    }
    if ($name -ne 'core' -and $CudaPath) {
        $configure += @("-DCMAKE_CUDA_COMPILER=$CudaPath/bin/nvcc.exe", "-DCUDAToolkit_ROOT=$CudaPath")
    }
    $configure += $CMakeArgs
    @{ backend = $name; configure = $configure; dir = $build
       build = @('--build', $build, '--parallel', "$Jobs") + $target }
}
if ($DryRun) { $plans | ConvertTo-Json -Depth 5; return }
if (!$env:VSCMD_VER) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $vs = & $vswhere -version '[17.0,18.0)' -latest -products '*' `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (!$vs) { throw 'Install Visual Studio 2022 C++ Build Tools or use its x64 developer shell.' }
    & (Join-Path $vs 'Common7/Tools/Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
}
if ($CudaPath) { $env:PATH = "$CudaPath/bin;$CudaPath/bin/x64;$env:PATH" }

# Pin the MSVC include-dependency prefix instead of letting CMake detect it.
#
# cl.exe prints /showIncludes in the system language, and CMake 4.3 stores the
# prefix it detects double-encoded: on a Chinese install rules.ninja gets the
# UTF-8 bytes of "注意: 包含文件:" decoded as the ANSI code page and re-encoded, so
# the value matches nothing cl ever prints. ninja then records an EMPTY
# dependency list for every C++ object while CUDA objects keep theirs (nvcc emits
# a depfile), which makes the build wrong rather than slow: a header edit
# recompiles the CUDA translation units and leaves the C++ ones alone, so one
# class has two member layouts inside a single link. That is how
# docs/kvmem-header-rebuild-trap.md's crash happened (a KvMemStore member read
# past the end of an object sized by an older layout), and how adding one float
# to ninfer::KvmemOptions made --kvmem-budget alone fail startup validation.
#
# The prefix is read from the compiler rather than hardcoded per language, exactly
# as scripts/windows/build.ps1 already does for the llama.cpp backend; this script
# was missing it. CMake takes CMAKE_CL_SHOWINCLUDES_PREFIX over a
# CMAKE_USER_MAKE_RULES_OVERRIDE file, which survives every reconfigure.
if (!$SkipMsvcIncludePrefix) {
    $probeDir = Join-Path ([IO.Path]::GetTempPath()) ("kvmem-msvc-probe-" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path $probeDir | Out-Null
    # cl.exe emits /showIncludes as UTF-8 whatever the console code page is, and native
    # output was observed arriving here already mojibake: Windows PowerShell 5.1 decodes
    # it with the ANSI code page unless the console output encoding says otherwise. Pin
    # UTF-8 for the probe so the detected prefix does not depend on the caller's console.
    $previousOutputEncoding = [Console]::OutputEncoding
    [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
    try {
        [IO.File]::WriteAllText((Join-Path $probeDir 'kvmem-msvc-probe.h'), '', [Text.UTF8Encoding]::new($false))
        [IO.File]::WriteAllText((Join-Path $probeDir 'probe.cpp'),
            "#include `"kvmem-msvc-probe.h`"`nint main(){return 0;}`n", [Text.UTF8Encoding]::new($false))
        $probeLines = & cl /nologo /showIncludes /c (Join-Path $probeDir 'probe.cpp') "/Fo$(Join-Path $probeDir 'probe.obj')" 2>&1
        if ($LASTEXITCODE -ne 0) { throw "MSVC include-prefix probe failed: $probeLines" }
        $includePrefix = $null
        foreach ($line in $probeLines) {
            if ("$line" -match '^(.+?)[A-Za-z]:[\\/].*kvmem-msvc-probe\.h$') {
                $includePrefix = $Matches[1].TrimEnd() + ' '
                break
            }
        }
        if (!$includePrefix) { throw 'Cannot detect the MSVC include dependency prefix' }
    } finally {
        [Console]::OutputEncoding = $previousOutputEncoding
        Remove-Item -LiteralPath $probeDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    $escapedPrefix = $includePrefix.Replace('\', '\\').Replace('"', '\"')
    Write-Verbose "MSVC include prefix: '$includePrefix'"
    foreach ($plan in $plans) {
        New-Item -ItemType Directory -Force -Path $plan.dir | Out-Null
        $rulesOverride = Join-Path $plan.dir 'kvmem-msvc-rules.cmake'
        [IO.File]::WriteAllText($rulesOverride,
            "set(CMAKE_CL_SHOWINCLUDES_PREFIX `"$escapedPrefix`")`n", [Text.UTF8Encoding]::new($false))
        $plan.configure += "-DCMAKE_USER_MAKE_RULES_OVERRIDE=$rulesOverride"
    }
}

foreach ($plan in $plans) {
    if ($plan.backend -ne 'core') {
        & $Python (Join-Path $source 'scripts/prepare-backends.py') --backend $plan.backend
        if ($LASTEXITCODE -ne 0) { throw "Backend preparation failed: $($plan.backend)" }
    }
    if (!$BuildOnly) {
        $configureArgs = $plan.configure
        & cmake @configureArgs
        if ($LASTEXITCODE -ne 0) { throw "Configure failed: $($plan.backend)" }
    }
    if (!$ConfigureOnly) {
        $buildArgs = $plan.build
        & cmake @buildArgs
        if ($LASTEXITCODE -ne 0) { throw "Build failed: $($plan.backend)" }
    }
}
