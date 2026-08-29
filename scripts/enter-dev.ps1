# Enable the installed Microsoft C++ toolchain in this process only.
# No persistent PATH, PowerShell profile, or execution-policy changes.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
# MSVC/Ninja can disagree on non-ASCII paths. Use a checked junction;
# this is an additional entry to the same files, not a copy or a move.
$projectItem = Get-Item -LiteralPath $projectRoot
if ($projectItem.LinkType -eq 'Junction') {
    $projectRoot = [string](@($projectItem.Target)[0])
}
$devRoot = Join-Path ([IO.Path]::GetPathRoot($projectRoot)) 'KernelLabDev'
$asciiRoot = Join-Path $devRoot 'gpu-kernel-lab'
New-Item -ItemType Directory -Path $devRoot -Force | Out-Null
if (Test-Path -LiteralPath $asciiRoot) {
    $entry = Get-Item -LiteralPath $asciiRoot
    if ($entry.LinkType -ne 'Junction' -or ([string](@($entry.Target)[0])).TrimEnd('\') -ne $projectRoot.TrimEnd('\')) {
        throw "Refusing to replace an unrelated path: $asciiRoot"
    }
} else {
    New-Item -ItemType Junction -Path $asciiRoot -Target $projectRoot | Out-Null
}
$tempRoot = Join-Path $devRoot 'gpu-kernel-lab-temp'
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
$env:KERNEL_LAB_ROOT = $asciiRoot
# Match MSVC diagnostic output and CMake/Ninja dependency parsing. This
# changes only this console session, not the Windows system locale.
& chcp.com 65001 | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Unable to enable UTF-8 for this console.' }
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = [Text.UTF8Encoding]::new($false)
$vsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vsWhere)) {
    throw 'Visual Studio Installer/vswhere was not found. Install the Desktop C++ Build Tools workload.'
}
$vsRoot = & $vsWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($vsRoot)) {
    throw 'No complete Visual Studio C++ toolchain was found.'
}
$launchScript = Join-Path $vsRoot 'Common7\Tools\Launch-VsDevShell.ps1'
& $launchScript -Arch amd64 -HostArch amd64 -SkipAutomaticLocation -NoLogo
foreach ($relative in @('Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin', 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja')) {
    $candidate = Join-Path $vsRoot $relative
    if ((Test-Path -LiteralPath $candidate) -and (($env:PATH -split ';') -notcontains $candidate)) {
        $env:PATH = "$candidate;$env:PATH"
    }
}
foreach ($tool in @('cl.exe', 'cmake.exe', 'ctest.exe', 'ninja.exe', 'link.exe')) {
    $resolved = Get-Command $tool -ErrorAction Stop
    Write-Host ('{0}: {1}' -f $tool, $resolved.Source)
}
Set-Location -LiteralPath $asciiRoot
Write-Host 'CPU development environment ready (MSVC x64 / C++17 / CMake / Ninja).'
Write-Host 'Run: & .\scripts\run-cpu.ps1'
