# Build "Cassiopea Pipeline.exe" — a tiny, no-console launcher for the UI.
#
# Run from anywhere:  .\packaging\build_launcher.ps1
# Output:             <project root>\Cassiopea Pipeline.exe
#
# The .exe just starts the UI via the project's venv; it does NOT bundle
# Python/torch, so keep it in the project root next to venv\ and scripts\.

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$py = ".\venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "venv not found. Run setup.ps1 first."
    exit 1
}

# Ensure PyInstaller is installed in the venv (idempotent — prints
# "already satisfied" if present). No stream redirection: redirecting a
# native exe's stderr in Windows PowerShell turns its log lines into errors.
Write-Host "Ensuring PyInstaller is installed..."
& $py -m pip install pyinstaller
if ($LASTEXITCODE -ne 0) {
    Write-Error "pip install pyinstaller failed."
    exit 1
}

# Regenerate the icon from the SVG (keeps it in sync if the SVG changes)
& $py "packaging\make_ico.py"

# Absolute paths: PyInstaller resolves --icon relative to the .spec location
# (which --specpath moves into build\), so a relative icon path would miss.
$icon = Join-Path $root "assets\app_icon.ico"

Write-Host "Building Cassiopea Pipeline.exe..."
& $py -m PyInstaller `
    --onefile --windowed --noconfirm --clean `
    --name "Cassiopea Pipeline" `
    --icon "$icon" `
    --distpath "$root" `
    --workpath "$root\build\pyinstaller" `
    --specpath "$root\build" `
    "$root\packaging\launcher.py"

if (Test-Path ".\Cassiopea Pipeline.exe") {
    Write-Host "`nDone -> $root\Cassiopea Pipeline.exe"
} else {
    Write-Error "Build did not produce the expected .exe."
    exit 1
}
