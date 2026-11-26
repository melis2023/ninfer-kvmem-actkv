# Local ninfer rebuild wrapper (iteration loop only).
# Mirrors scripts/windows/build-backends.ps1 for Backend=ninfer, but skips
# prepare-backends.py: this machine has no git and the backend tree is already
# vendored/configured. Do not use this for release validation.
#
# Usage:
#   pwsh scripts/windows/build-ninfer-local.ps1                # configure + build
#   pwsh scripts/windows/build-ninfer-local.ps1 -ConfigureOnly
#   pwsh scripts/windows/build-ninfer-local.ps1 -BuildOnly
#   pwsh scripts/windows/build-ninfer-local.ps1 -Targets ninfer
[CmdletBinding()]
param(
    [string]$CudaPath = $env:CUDA_PATH,
    [string]$CudaArchitectures = '120a',
    [int]$Jobs = 8,
    [string[]]$Targets = @('ninfer', 'ninfer-serve'),
    [switch]$ConfigureOnly,
    [switch]$BuildOnly
)
$ErrorActionPreference = 'Stop'
$source = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$build  = Join-Path $source 'build-backends/ninfer'
$tree   = Join-Path $source 'backends/ninfer'

if (!$env:VSCMD_VER) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $vs = & $vswhere -version '[17.0,18.0)' -latest -products '*' `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (!$vs) { throw 'VS 2022 C++ tools not found.' }
    & (Join-Path $vs 'Common7/Tools/Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
}
# VsDevShell repoints VCPKG_ROOT at VS's bundled vcpkg; the ffmpeg/curl import
# libs live in the standalone tree. Reset after the shell init.
$env:VCPKG_ROOT = 'C:/vcpkg'
$env:VCPKG_TARGET_TRIPLET = 'x64-windows'
# VsDevShell sometimes fails to detect the Windows SDK (WindowsSdkDir empty), so
# the ucrt/um include and lib dirs are missing from INCLUDE/LIB and cl cannot
# open stddef.h. Append the latest installed SDK manually when that happens.
if (!$env:WindowsSdkDir) {
    $sdkRoot = 'C:\Program Files (x86)\Windows Kits\10'
    $sdkVer = (Get-ChildItem "$sdkRoot\Include" -Directory | Sort-Object Name | Select-Object -Last 1).Name
    if ($sdkVer) {
        $env:WindowsSdkDir = "$sdkRoot\"
        $env:WindowsSDKVersion = "$sdkVer\"
        $env:INCLUDE = "$sdkRoot\Include\$sdkVer\ucrt;$sdkRoot\Include\$sdkVer\um;$sdkRoot\Include\$sdkVer\shared;$env:INCLUDE"
        $env:LIB = "$sdkRoot\Lib\$sdkVer\ucrt\x64;$sdkRoot\Lib\$sdkVer\um\x64;$env:LIB"
    }
}
if ($CudaPath) { $env:PATH = "$CudaPath/bin;$CudaPath/bin/x64;$env:PATH" }

$configure = @('-S', $tree, '-B', $build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
    "-DNINFER_KVMEM_SOURCE_DIR=$source/kvmem", '-DNINFER_BUILD_APPS=ON',
    '-DNINFER_BUILD_BENCHMARKS=OFF', "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures",
    "-DCMAKE_CUDA_COMPILER=$CudaPath/bin/nvcc.exe", "-DCUDAToolkit_ROOT=$CudaPath",
    '-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake',
    '-DCMAKE_PREFIX_PATH=C:/vcpkg/installed/x64-windows')
if ($CudaArchitectures -eq '120a') { $configure += '-DNINFER_SM120_NATIVE=ON' }

if (!$BuildOnly) {
    Write-Host "==> cmake configure"
    & 'C:\Program Files\CMake\bin\cmake.exe' @configure
    if ($LASTEXITCODE -ne 0) { throw "Configure failed ($LASTEXITCODE)" }
}
if (!$ConfigureOnly) {
    $buildArgs = @('--build', $build, '--parallel', "$Jobs")
    foreach ($t in $Targets) { $buildArgs += @('--target', $t) }
    Write-Host "==> cmake build $($Targets -join ', ')"
    & 'C:\Program Files\CMake\bin\cmake.exe' @buildArgs
    if ($LASTEXITCODE -ne 0) { throw "Build failed ($LASTEXITCODE)" }
}
