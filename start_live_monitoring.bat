@echo off
setlocal

cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_live_monitoring.ps1"

if errorlevel 1 (
    echo.
    echo Live monitoring launcher failed.
    pause
)
