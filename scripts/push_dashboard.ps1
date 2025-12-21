# Push dashboard assets to Android device
# This script builds the dashboard on Windows and transfers it to the Android device

param(
    [string]$AndroidPath = "/data/local/tmp/exo-dashboard"
)

Write-Host "Building dashboard on Windows..." -ForegroundColor Green

# Build the dashboard
Push-Location dashboard
try {
    npm run build
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build dashboard"
        exit 1
    }
} finally {
    Pop-Location
}

Write-Host "Dashboard built successfully" -ForegroundColor Green

# Create a temporary directory for transfer
$tempDir = Join-Path $env:TEMP "exo-dashboard-transfer"
if (Test-Path $tempDir) {
    Remove-Item $tempDir -Recurse -Force
}
New-Item -ItemType Directory -Path $tempDir | Out-Null

# Copy dashboard build to temp directory
Copy-Item "dashboard/build/*" $tempDir -Recurse

Write-Host "Transferring dashboard to Android device..." -ForegroundColor Green

# Push to Android device
adb push $tempDir $AndroidPath

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to push dashboard to Android device"
    exit 1
}

# Clean up temp directory
Remove-Item $tempDir -Recurse -Force

Write-Host "Dashboard transferred successfully!" -ForegroundColor Green
Write-Host "Set DASHBOARD_DIR=$AndroidPath in your exo environment to use the dashboard" -ForegroundColor Yellow
