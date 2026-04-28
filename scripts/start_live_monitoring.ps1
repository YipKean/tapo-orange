param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$DotenvPath = Join-Path $RepoRoot ".env"

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

function Convert-ToEncodedCommand {
    param([string]$CommandText)
    $bytes = [System.Text.Encoding]::Unicode.GetBytes($CommandText)
    return [Convert]::ToBase64String($bytes)
}

function New-RepoCommand {
    param([string]$Body)

    $repoLiteral = $RepoRoot.Replace("'", "''")
    return @"
`$ErrorActionPreference = "Stop"
try {
    Set-Location -LiteralPath '$repoLiteral'
$Body

    if (`$null -ne `$global:LASTEXITCODE -and `$global:LASTEXITCODE -ne 0) {
        throw "Command exited with code `$global:LASTEXITCODE."
    }
} catch {
    Write-Host ""
    Write-Host "Command failed: `$(`$_.Exception.Message)" -ForegroundColor Red
    Read-Host "Press Enter to close this window"
    exit 1
}
"@
}

function Start-CmdWindow {
    param(
        [string]$Title,
        [string]$PowerShellBody
    )

    $encoded = Convert-ToEncodedCommand -CommandText (New-RepoCommand -Body $PowerShellBody)
    $cmdLine = "title $Title && powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $encoded"

    if ($DryRun) {
        Write-Host "Would start: $Title"
        return
    }

    Start-Process `
        -FilePath "cmd.exe" `
        -ArgumentList @("/c", $cmdLine) `
        -WorkingDirectory $RepoRoot | Out-Null
}

function Get-MatchingProcessCommandLines {
    param([string]$Pattern)

    try {
        $processes = Get-CimInstance `
            Win32_Process `
            -Filter "Name = 'python.exe' OR Name = 'python3.exe' OR Name = 'powershell.exe' OR Name = 'cmd.exe'" `
            -ErrorAction Stop
    } catch {
        Write-Warning "Could not inspect existing processes: $($_.Exception.Message)"
        return @()
    }

    $commandLines = @()
    foreach ($process in $processes) {
        if ($null -eq $process.CommandLine) {
            continue
        }
        if ($process.CommandLine -match $Pattern) {
            $commandLines += $process.CommandLine
        }
    }

    return $commandLines
}

function Start-CmdWindowIfNotRunning {
    param(
        [string]$Title,
        [string]$PowerShellBody,
        [string]$ProcessPattern
    )

    $existingCommandLines = Get-MatchingProcessCommandLines -Pattern $ProcessPattern
    if ($existingCommandLines.Count -gt 0) {
        Write-Host "Already running, skipping: $Title"
        if ($DryRun) {
            foreach ($commandLine in $existingCommandLines) {
                Write-Host "  $commandLine"
            }
        }
        return $false
    }

    Start-CmdWindow -Title $Title -PowerShellBody $PowerShellBody
    return $true
}

$watchdogCommand = Get-DotenvValue -Path $DotenvPath -Key "WATCHDOG_TAPO_COMMAND"
if (-not $watchdogCommand) {
    throw "WATCHDOG_TAPO_COMMAND was not found in .env. Open scripts\command_builder_gui.py and save the watchdog command first."
}

$classifierBody = @'
function Get-DotenvValue {
    param(
        [string]$Path,
        [string]$Key
    )

    foreach ($rawLine in Get-Content -LiteralPath $Path) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            continue
        }

        $parts = $line.Split("=", 2)
        if ($parts[0].Trim() -ne $Key) {
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

$command = Get-DotenvValue -Path ".env" -Key "WATCHDOG_TAPO_COMMAND"
if (-not $command) {
    throw "WATCHDOG_TAPO_COMMAND was not found in .env."
}

Write-Host "Starting classifier command from .env WATCHDOG_TAPO_COMMAND..."
Write-Host $command
Invoke-Expression $command
'@

$discordBody = @'
$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

Write-Host "Starting Discord alert bot..."
& $python "scripts\discord_alert_bot.py" --poll-seconds 1.0
'@

$watchdogBody = @'
Write-Host "Starting Tapo watchdog..."
& powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\watchdog_tapo.ps1"
'@

Write-Host "Launching live monitoring windows from: $RepoRoot"
$classifierStarted = Start-CmdWindowIfNotRunning `
    -Title "Tapo Classifier" `
    -PowerShellBody $classifierBody `
    -ProcessPattern 'scripts[\\/]+tapo_opencv_classifier_test\.py'

if ($classifierStarted) {
    Start-Sleep -Seconds 2
}

Start-CmdWindowIfNotRunning `
    -Title "Tapo Discord Bot" `
    -PowerShellBody $discordBody `
    -ProcessPattern 'scripts[\\/]+discord_alert_bot\.py' | Out-Null

Start-Sleep -Seconds 1

Start-CmdWindowIfNotRunning `
    -Title "Tapo Watchdog" `
    -PowerShellBody $watchdogBody `
    -ProcessPattern 'scripts[\\/]+watchdog_tapo\.ps1' | Out-Null

Write-Host "Done. Missing services were opened; already-running services were left alone."
