param(
    [int]$IdleThresholdSeconds = 900,
    [int]$CheckIntervalSeconds = 60
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$StateDir = Join-Path $RepoRoot "captures"
$LogFile = Join-Path $StateDir "tapo_watchdog.log"
$CommandLogFile = Join-Path $StateDir "tapo_watchdog_child.log"
$IdleScript = Join-Path $ScriptDir "windows_idle_seconds.ps1"
$DotenvPath = Join-Path $RepoRoot ".env"

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null

function Get-DotenvValue {
    param(
        [string]$Path,
        [string]$Key
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }

    foreach ($rawLine in Get-Content -LiteralPath $Path) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            continue
        }
        if (-not $line.Contains("=")) {
            continue
        }
        $parts = $line.Split("=", 2)
        $k = $parts[0].Trim()
        if ($k -ne $Key) {
            continue
        }
        $value = $parts[1].Trim()
        if (
            $value.Length -ge 2 -and
            (
                ($value.StartsWith('"') -and $value.EndsWith('"')) -or
                ($value.StartsWith("'") -and $value.EndsWith("'"))
            )
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        return $value
    }

    return $null
}

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogFile -Value "[$ts] $Message"
}

function Get-WindowsIdleSeconds {
    try {
        $value = & powershell -NoProfile -ExecutionPolicy Bypass -File $IdleScript
        if ($value -match '^\d+$') {
            return [int]$value
        }
        return 0
    } catch {
        return 0
    }
}

function Test-TapoRunning {
    try {
        $procs = Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'python3.exe'" -ErrorAction Stop
        foreach ($p in $procs) {
            if (
                $null -ne $p.CommandLine -and
                $p.CommandLine -match 'scripts[\\/]+(tapo_opencv_test|tapo_opencv_classifier_test)\.py'
            ) {
                return $true
            }
        }
        return $false
    } catch {
        # Fallback for restricted environments where process command line isn't readable.
        $py = Get-Process -Name python, python3 -ErrorAction SilentlyContinue
        return ($null -ne $py)
    }
}

function Start-Tapo {
    $envCommand = Get-DotenvValue -Path $DotenvPath -Key "WATCHDOG_TAPO_COMMAND"
    if ($envCommand) {
        $cmdArgs = "/c ""$envCommand >> `"$CommandLogFile`" 2>&1"""
        Push-Location $RepoRoot
        try {
            Start-Process `
                -FilePath "cmd.exe" `
                -ArgumentList $cmdArgs `
                -WorkingDirectory $RepoRoot `
                -WindowStyle Hidden | Out-Null
        } finally {
            Pop-Location
        }
        Write-Log "started command from .env: WATCHDOG_TAPO_COMMAND (child log: $CommandLogFile)"
        return
    }

    $args = @(
        "scripts/tapo_opencv_test.py",
        "--zone-polygon", "0.4113,0.5238;0.4845,0.5324;0.4821,0.6393;0.4078,0.625",
        "--motion-threshold", "1.4",
        "--process-fps", "5",
        "--snapshot-cooldown", "10",
        "--alert-seconds", "4",
        "--save-clip-on-alert",
        "--zone-edit",
        "--clip-seconds", "10",
        "--cat-model", "models/yolov8m.pt",
        "--cat-confidence", "0.08",
        "--cat-enter-frames", "1",
        "--cat-hold-seconds", "1.5",
        "--cat-detect-mode", "always",
        "--cat-zone-overlap", "0.25",
        "--cat-imgsz", "1920",
        "--launch-origin", "watchdog_ps1",
        "--device", "cuda"
    )

    Push-Location $RepoRoot
    try {
        Start-Process -FilePath "python" -ArgumentList $args -WorkingDirectory $RepoRoot -WindowStyle Hidden | Out-Null
    } finally {
        Pop-Location
    }
}

Write-Log "watchdog (powershell) started"

while ($true) {
    $idleSeconds = Get-WindowsIdleSeconds
    $running = Test-TapoRunning

    if (-not $running -and $idleSeconds -ge $IdleThresholdSeconds) {
        Write-Log "windows idle ${idleSeconds}s and script not running -> start"
        Start-Tapo
    }

    Start-Sleep -Seconds $CheckIntervalSeconds
}
