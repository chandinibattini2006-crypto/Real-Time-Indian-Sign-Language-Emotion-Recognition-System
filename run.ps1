$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -Path $ScriptDir
if (Test-Path "$ScriptDir\.venv\Scripts\python.exe") {
    & "$ScriptDir\.venv\Scripts\python.exe" "$ScriptDir\main.py" @args
} else {
    Write-Error ".venv not found. Please set up the virtual environment first."
}
