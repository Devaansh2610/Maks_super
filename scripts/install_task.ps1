# Registers Maks to start automatically at logon, using Windows Task
# Scheduler (built in, no extra tooling, no paid service). Run from the
# project root in PowerShell: .\scripts\install_task.ps1
# Does not need an elevated prompt -- registering a task for your own logon
# is a normal-user operation.

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot "venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Host "venv not found -- run .\scripts\install_windows.ps1 first."
    exit 1
}

$TaskName = "Maks"
$Action = New-ScheduledTaskAction -Execute $PythonExe -Argument "-m maks.main" -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
    -Description "Maks personal voice assistant" | Out-Null

Write-Host "==> Task '$TaskName' registered. It will start automatically at your next logon."
Write-Host "    Start it right now with:  Start-ScheduledTask -TaskName $TaskName"
Write-Host "    Check on it with:         Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo"
Write-Host "    Remove it with:           Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
