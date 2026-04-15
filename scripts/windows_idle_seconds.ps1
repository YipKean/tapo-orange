$signature = @"
using System;
using System.Runtime.InteropServices;
public static class IdleTime {
    [StructLayout(LayoutKind.Sequential)]
    public struct LASTINPUTINFO {
        public uint cbSize;
        public uint dwTime;
    }

    [DllImport("user32.dll")]
    public static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);
}
"@

Add-Type -TypeDefinition $signature -ErrorAction SilentlyContinue | Out-Null

$lii = New-Object IdleTime+LASTINPUTINFO
$lii.cbSize = [System.Runtime.InteropServices.Marshal]::SizeOf($lii)
[IdleTime]::GetLastInputInfo([ref]$lii) | Out-Null

$idleMs = [Environment]::TickCount - $lii.dwTime
if ($idleMs -lt 0) { $idleMs = 0 }

[int]($idleMs / 1000)
