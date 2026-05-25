# CIRCA — RunPod Upload Package Builder
# ====================================
# This PowerShell script packages exactly the required dataset and scripts
# for cloud training, excluding unnecessary files (like virtual environments,
# local runs, logs, Git history) to minimize file size and upload time.

$ErrorActionPreference = "Stop"

# 1. Print Banner
Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "          CIRCA RunPod Upload Package Builder             " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# 2. Path Configurations
$RootPath = Resolve-Path "d:\FYP\CIRCA"
$StagingPath = Join-Path $RootPath "runpod_staging"
$ZipPath = Join-Path $RootPath "CIRCA_runpod.zip"

$DatasetSource = Join-Path $RootPath "datasets\unified_pcb_v3"
$TrainEngineSource = Join-Path $RootPath "train_engine.py"
$ReqsSource = Join-Path $RootPath "requirements_runpod.txt"
$SetupPySource = Join-Path $RootPath "scripts\runpod_setup.py"
$SetupShSource = Join-Path $RootPath "scripts\runpod_setup.sh"

# 3. Validations
Write-Host "[*] Validating required files and directories..." -ForegroundColor Yellow

if (-not (Test-Path $DatasetSource)) {
    Write-Error "Dataset source not found at $DatasetSource. Please run the build script first."
}
if (-not (Test-Path $TrainEngineSource)) {
    Write-Error "Training engine not found at $TrainEngineSource."
}
if (-not (Test-Path $ReqsSource)) {
    Write-Error "Requirements file not found at $ReqsSource."
}
if (-not (Test-Path $SetupPySource)) {
    Write-Error "Setup script (Python) not found at $SetupPySource."
}
if (-not (Test-Path $SetupShSource)) {
    Write-Error "Setup script (Shell) not found at $SetupShSource."
}

Write-Host "  [+] All required files and directories found." -ForegroundColor Green

# 4. Clean up previous files if they exist
if (Test-Path $StagingPath) {
    Write-Host "[*] Removing previous staging folder..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $StagingPath
}
if (Test-Path $ZipPath) {
    Write-Host "[*] Removing previous ZIP package..." -ForegroundColor Yellow
    Remove-Item -Force $ZipPath
}

# 5. Create Staging Directory
Write-Host "[*] Creating temporary staging environment..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path $StagingPath | Out-Null
New-Item -ItemType Directory -Path (Join-Path $StagingPath "datasets") | Out-Null
New-Item -ItemType Directory -Path (Join-Path $StagingPath "scripts") | Out-Null

# 6. Copy Files to Staging
Write-Host "[*] Copying files to staging (excluding logs, caches, and backups)..." -ForegroundColor Yellow

# Copy dataset (excluding any temp files)
Copy-Item -Recurse -Path $DatasetSource -Destination (Join-Path $StagingPath "datasets")

# Copy scripts
Copy-Item -Path $SetupPySource -Destination (Join-Path $StagingPath "scripts")
Copy-Item -Path $SetupShSource -Destination (Join-Path $StagingPath "scripts")

# Copy root scripts and requirement sheets
Copy-Item -Path $TrainEngineSource -Destination $StagingPath
Copy-Item -Path $ReqsSource -Destination $StagingPath

Write-Host "  [+] Staging copying complete." -ForegroundColor Green

# 7. Create ZIP Package
Write-Host "[*] Compressing staging folder into 'CIRCA_runpod.zip' (this may take a minute)..." -ForegroundColor Yellow
try {
    # Windows native utility to create ZIP
    Compress-Archive -Path "$StagingPath\*" -DestinationPath $ZipPath -Force
    Write-Host "  [+] Compression successful!" -ForegroundColor Green
}
catch {
    Write-Error "Compression failed: $_"
}

# 8. Clean up Staging Directory
Write-Host "[*] Cleaning up temporary staging files..." -ForegroundColor Yellow
Remove-Item -Recurse -Force $StagingPath
Write-Host "  [+] Clean up successful." -ForegroundColor Green

# 9. Get File Info
$ZipFile = Get-Item $ZipPath
$ZipSizeMB = [Math]::Round(($ZipFile.Length / 1MB), 2)

# 10. Success Report
Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "                PACKAGE CREATED SUCCESSFULLY!             " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "File Name     : CIRCA_runpod.zip"
Write-Host "File Location : $ZipPath"
Write-Host "File Size     : $ZipSizeMB MB"
Write-Host "==========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps for RunPod Upload:"
Write-Host "-----------------------------"
Write-Host "1. Start your RunPod GPU Instance (RTX 3090 Secure Cloud recommended)."
Write-Host "2. Access the instance Jupyter Lab interface."
Write-Host "3. Upload the generated 'CIRCA_runpod.zip' file."
Write-Host "4. Open a terminal in Jupyter Lab and run:"
Write-Host "   unzip CIRCA_runpod.zip -d /workspace/CIRCA"
Write-Host "   cd /workspace/CIRCA"
Write-Host "   bash scripts/runpod_setup.sh"
Write-Host "5. Paste the launch command printed by the setup script to start Phase 1!"
Write-Host "==========================================================" -ForegroundColor Green
Write-Host ""
