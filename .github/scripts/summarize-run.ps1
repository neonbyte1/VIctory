# Turns a victory log into the table of the test-report template.
# Usage (Windows PowerShell 5 or newer, nothing to install):
#   powershell -ExecutionPolicy Bypass -File .github\scripts\summarize-run.ps1 C:\path\to\log.txt
param([Parameter(Mandatory = $true, Position = 0)][string]$Log)

function Get-Median([double[]]$Values) {
    $s = @($Values | Sort-Object)
    $n = $s.Count
    if ($n % 2 -eq 1) { return $s[($n - 1) / 2] }
    return ($s[$n / 2 - 1] + $s[$n / 2]) / 2
}
function Fmt([double]$V) { return $V.ToString('0.0', [Globalization.CultureInfo]::InvariantCulture) }

$esc = [char]27
$ansi = [regex]($esc + '\[[0-9;]*m')
$frameRe = [regex]'([\d.]+) fps, ([\d.]+) ms/title frame \(([\d.]+) preparing.*?; ([\d.]+) on the GPU; ([\d.]+) copying back.*?dropped (\d+) without'
$vramRe = [regex]'video memory: (\d+) MB of a (\d+) MB budget'
$gpuRe = [regex]'host GPU name=(.+?) vendor=\S+ driver="?([^" ]+)"? dedicated_mib=(\d+)'
$crashRe = [regex]'(?i)device removed|panicked|access violation|unhandled exception|crashes\.log|fatal'
$buildRe = [regex]'(?i)\b(?:commit|git)[ =:]+([0-9a-f]{7,40})\b|victory (\d+\.\d+\.\d+\S*)'

$frames = New-Object System.Collections.ArrayList
$vrams = New-Object System.Collections.ArrayList
$refused = @{}
$crash = $null; $gpu = $null; $build = $null

foreach ($raw in [IO.File]::ReadLines((Resolve-Path -LiteralPath $Log).Path)) {
    $line = $ansi.Replace($raw, '').TrimEnd()
    $m = $frameRe.Match($line)
    if ($m.Success) {
        [void]$frames.Add(@(1..6 | ForEach-Object { [double]::Parse($m.Groups[$_].Value, [Globalization.CultureInfo]::InvariantCulture) }))
        continue
    }
    $m = $vramRe.Match($line)
    if ($m.Success) { [void]$vrams.Add([double]$m.Groups[1].Value) }
    if ($line.Contains('shader does not translate')) {
        $h = [regex]::Match($line, 'hash=(\w+)')
        if ($h.Success) { $refused[$h.Groups[1].Value] = $true }
    }
    if (-not $crash -and ($line.Contains('victory') -or $line.Contains('panicked')) -and
        $crashRe.IsMatch($line) -and -not $line.Contains('pipeline-journal')) {
        $crash = $line.Substring(0, [Math]::Min(300, $line.Length))
    }
    if (-not $gpu) {
        $m = $gpuRe.Match($line)
        if ($m.Success) { $gpu = '{0}, driver {1}, {2} MiB' -f $m.Groups[1].Value, $m.Groups[2].Value, $m.Groups[3].Value }
    }
    if (-not $build -and -not $line.Contains('gamescope')) {
        $m = $buildRe.Match($line)
        if ($m.Success) { $build = if ($m.Groups[1].Success) { $m.Groups[1].Value } else { $m.Groups[2].Value } }
    }
}

if ($frames.Count -eq 0) {
    Write-Error "No frame lines ('N fps, X ms/title frame ...') found; was VICTORY_LOG=info set?"
    exit 1
}
$take = [Math]::Min(30, $frames.Count)
$last = @($frames | Select-Object -Last $take)
function Col([int]$i) { return [double[]]@($last | ForEach-Object { $_[$i] }) }
$vr = 'n/a'
if ($vrams.Count -gt 0) {
    $v = @($vrams | Select-Object -Last 30)
    $vr = [string][int](Get-Median ([double[]]$v))
}
$minFrame = ($last | ForEach-Object { $_[1] } | Measure-Object -Minimum).Minimum

"Frame lines found: {0} (using the last {1})" -f $frames.Count, $take
'| Frame ms | Min ms | fps | Preparing ms | GPU ms | Copy-back ms | Dropped | VRAM MB |'
'|---|---|---|---|---|---|---|---|'
'| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} |' -f (Fmt (Get-Median (Col 1))), (Fmt $minFrame), (Fmt (Get-Median (Col 0))),
    (Fmt (Get-Median (Col 2))), (Fmt (Get-Median (Col 3))), (Fmt (Get-Median (Col 4))), [int](Get-Median (Col 5)), $vr
"Shaders refused (unique): {0}" -f $refused.Count
"Crash/error line: {0}" -f $(if ($crash) { $crash } else { 'none found' })
"Host GPU: {0}" -f $(if ($gpu) { $gpu } else { 'not logged' })
"Build/commit in log: {0}" -f $(if ($build) { $build } else { 'not logged' })
