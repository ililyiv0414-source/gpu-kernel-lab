param (
    [ValidateSet('Release', 'Debug')]
    [string]$Configuration = 'Release',
    [switch]$SkipBenchmark
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$previousLocation = Get-Location
$runId = (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '-' + $Configuration.ToLowerInvariant()
$resultDir = Join-Path $projectRoot "results\$runId"
New-Item -ItemType Directory -Path $resultDir -Force | Out-Null

function Invoke-LoggedNative {
    param([string]$Program, [string[]]$Arguments, [string]$LogName)
    $commandLine = $Program + ' ' + (($Arguments | ForEach-Object { '"' + $_ + '"' }) -join ' ')
    $commandLine | Add-Content -LiteralPath (Join-Path $resultDir 'commands.txt') -Encoding UTF8
    # Windows PowerShell may wrap native stderr in ErrorRecords; exit status
    # below is authoritative. Preserve both stdout and stderr in the log.
    $ErrorActionPreference = 'Continue'
    & $Program @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $resultDir $LogName) | Out-Host
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Program failed with exit code $exitCode. See $resultDir\$LogName"
    }
}

try {
    & (Join-Path $PSScriptRoot 'enter-dev.ps1')
    $preset = 'windows-cpu-' + $Configuration.ToLowerInvariant()
    $buildDir = Join-Path $env:KERNEL_LAB_ROOT ('build\msvc-cpu-' + $Configuration.ToLowerInvariant())
    @(
        "Date: $([DateTimeOffset]::Now.ToString('o'))"
        "Configuration: $Configuration"
        "Source entry: $env:KERNEL_LAB_ROOT"
        "Build directory: $buildDir"
        "Temporary directory: $env:TEMP"
        "Console output encoding: $([Console]::OutputEncoding.WebName)"
        'CUDA: OFF; CPU-only validation'
        "PowerShell: $($PSVersionTable.PSVersion)"
        "MSVC tools: $env:VCToolsVersion"
        "Windows SDK: $env:WindowsSDKVersion"
        "Compiler: $((Get-Command cl.exe).Source)"
        "CMake: $(& cmake.exe --version | Select-Object -First 1)"
        "Ninja: $(& ninja.exe --version)"
        "CPU: $((Get-CimInstance Win32_Processor).Name -join '; ')"
        "Display adapters: $((Get-CimInstance Win32_VideoController).Name -join '; ')"
        'Timing scope: CPU wall time including output allocation; not a GPU speedup.'
    ) | Set-Content -LiteralPath (Join-Path $resultDir 'environment.txt') -Encoding UTF8
    $sourceFiles = @(
        Get-ChildItem -LiteralPath (Join-Path $projectRoot 'src'), (Join-Path $projectRoot 'include'), (Join-Path $projectRoot 'tests'), $PSScriptRoot -Recurse -File
        Get-Item -LiteralPath (Join-Path $projectRoot 'CMakeLists.txt'), (Join-Path $projectRoot 'CMakePresets.json')
    )
    $sourceFiles | Where-Object { $_.FullName -notmatch '__pycache__' -and $_.Extension -ne '.pyc' } | Sort-Object FullName | ForEach-Object {
        $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
        [pscustomobject]@{ RelativePath = $_.FullName.Substring($projectRoot.Length + 1); SHA256 = $hash.Hash }
    } | Export-Csv -LiteralPath (Join-Path $resultDir 'source-sha256.csv') -NoTypeInformation -Encoding UTF8

    Invoke-LoggedNative 'cmake.exe' @('--preset', $preset) 'configure.log'
    Invoke-LoggedNative 'cmake.exe' @('--build', '--preset', $preset, '--parallel', '2') 'build.log'
    Invoke-LoggedNative 'ninja.exe' @('-C', $buildDir, '-t', 'deps') 'dependencies.log'
    $dependencies = Get-Content -LiteralPath (Join-Path $resultDir 'dependencies.log') -Raw
    if ($dependencies -notmatch 'softmax\.hpp' -or $dependencies -notmatch 'convolution\.hpp') {
        throw 'Ninja header dependencies are missing. Reconfigure this preset with cmake --fresh, then rerun.'
    }
    Invoke-LoggedNative 'ctest.exe' @('--preset', $preset, '--verbose', '--no-tests=error') 'ctest.log'
    if (-not $SkipBenchmark) {
        $csv = Join-Path $env:KERNEL_LAB_ROOT "results\$runId\cpu_benchmark.csv"
        Invoke-LoggedNative (Join-Path $buildDir 'cpu_benchmark.exe') @($csv) 'benchmark.log'
        $rows = @(Import-Csv -LiteralPath $csv)
        if ($rows.Count -ne 6) { throw "Expected 6 benchmark records; got $($rows.Count)." }
        foreach ($row in $rows) {
            $latency = [double]::Parse($row.median_ms, [Globalization.CultureInfo]::InvariantCulture)
            if ([double]::IsNaN($latency) -or [double]::IsInfinity($latency) -or $latency -lt 0) {
                throw 'Invalid benchmark latency.'
            }
        }
    }
    Copy-Item -LiteralPath (Join-Path $buildDir 'CMakeCache.txt') -Destination (Join-Path $resultDir 'CMakeCache.txt')
    @(
        'PASS: configure, build, header dependency tracking and existing CPU correctness tests succeeded.'
        "Benchmark executed and six records validated: $(-not $SkipBenchmark)"
        'CUDA: not built or tested.'
    ) | Set-Content -LiteralPath (Join-Path $resultDir 'SUCCESS.txt') -Encoding UTF8
    Write-Host "SUCCESS. Evidence: $resultDir"
} catch {
    ($_ | Out-String) | Set-Content -LiteralPath (Join-Path $resultDir 'FAILURE.txt') -Encoding UTF8
    Write-Error $_ -ErrorAction Continue
    exit 1
} finally {
    Set-Location -LiteralPath $previousLocation.Path
}
