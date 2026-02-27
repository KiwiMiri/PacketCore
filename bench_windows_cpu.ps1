$ErrorActionPreference = 'Stop'
$cores = [int](Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
Write-Output "cores=$cores"

function Measure-Run {
    param(
        [string]$Label,
        [string]$PyExe,
        [string[]]$Args,
        [int]$DurationSec,
        [string]$ProgressRegex
    )

    $stdoutPath = "$Label.bench.stdout.log"
    $stderrPath = "$Label.bench.stderr.log"
    Remove-Item $stdoutPath, $stderrPath -ErrorAction SilentlyContinue

    $p = Start-Process -FilePath $PyExe -ArgumentList $Args -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    $start = Get-Date
    Write-Output "[$Label] pid=$($p.Id) start=$($start.ToString('o'))"

    $samples = New-Object System.Collections.Generic.List[object]
    $prevCpu = $null
    $prevTs = $null
    $lastPrintMark = -10

    while ($true) {
        Start-Sleep -Seconds 2
        $p.Refresh()
        if ($p.HasExited) { break }

        $now = Get-Date
        $elapsed = ($now - $start).TotalSeconds
        if ($elapsed -ge $DurationSec) { break }

        $gp = Get-Process -Id $p.Id -ErrorAction SilentlyContinue
        if (-not $gp) { break }

        $cpu = [double]$gp.CPU
        $threads = [int]$gp.Threads.Count

        if ($prevCpu -ne $null -and $prevTs -ne $null) {
            $dt = ($now - $prevTs).TotalSeconds
            if ($dt -gt 0) {
                $cpuPct = (($cpu - $prevCpu) / $dt) * 100.0
                $samples.Add([pscustomobject]@{
                    CpuPct = $cpuPct
                    Threads = $threads
                })

                $elapsedInt = [int][math]::Floor($elapsed)
                if ($elapsedInt - $lastPrintMark -ge 10) {
                    Write-Output ("[{0}] t={1}s cpu_pct={2} threads={3}" -f $Label, $elapsedInt, [math]::Round($cpuPct, 1), $threads)
                    $lastPrintMark = $elapsedInt
                }
            }
        }

        $prevCpu = $cpu
        $prevTs = $now
    }

    if (-not $p.HasExited) {
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 300
    }

    $end = Get-Date
    $runtimeSec = ($end - $start).TotalSeconds

    if ($samples.Count -eq 0) {
        $avgCpu = 0.0
        $maxCpu = 0.0
        $avgThreads = 0.0
    } else {
        $avgCpu = ($samples | Measure-Object -Property CpuPct -Average).Average
        $maxCpu = ($samples | Measure-Object -Property CpuPct -Maximum).Maximum
        $avgThreads = ($samples | Measure-Object -Property Threads -Average).Average
    }

    $avgCpuNorm = if ($cores -gt 0) { $avgCpu / $cores } else { 0.0 }
    $maxCpuNorm = if ($cores -gt 0) { $maxCpu / $cores } else { 0.0 }

    Write-Output ("RESULT label={0} runtime_s={1:N1} samples={2} avg_cpu_pct={3:N1} max_cpu_pct={4:N1} avg_threads={5:N1} avg_cpu_system_pct={6:N2} max_cpu_system_pct={7:N2}" -f $Label, $runtimeSec, $samples.Count, $avgCpu, $maxCpu, $avgThreads, $avgCpuNorm, $maxCpuNorm)

    $progressLine = $null
    if (Test-Path $stderrPath) {
        $m = Select-String -Path $stderrPath -Pattern $ProgressRegex -ErrorAction SilentlyContinue
        if ($m) { $progressLine = $m[-1].Line }
    }
    if (-not $progressLine -and (Test-Path $stdoutPath)) {
        $m2 = Select-String -Path $stdoutPath -Pattern $ProgressRegex -ErrorAction SilentlyContinue
        if ($m2) { $progressLine = $m2[-1].Line }
    }

    if ($progressLine) {
        Write-Output "LAST_PROGRESS label=$Label line=$progressLine"
    } else {
        Write-Output "LAST_PROGRESS label=$Label line=<none>"
    }
}

$py = ".venv\\Scripts\\python.exe"

Measure-Run -Label "convert_ijson_orjson" -PyExe $py -Args @(
    "convert_json_array_to_jsonl.py",
    "--in", "modbus.json",
    "--out", "NUL",
    "--progress-every", "100000",
    "--progress-seconds", "10"
) -DurationSec 70 -ProgressRegex "^\\[progress\\]"

Measure-Run -Label "feature_modbus_preprocessing" -PyExe $py -Args @(
    "_bench_modbus_preprocess_runner.py"
) -DurationSec 70 -ProgressRegex "처리중"
