# ==============================================================================
# GitMesh Local Testing CLI Wrapper (PowerShell)
# Automatically fetches a fresh gcloud access token and binds it to Modal
# ==============================================================================

# Force Python stdout/stderr encoding to UTF-8 to prevent Windows Unicode charmap crashes
$env:PYTHONIOENCODING = "utf-8"

# Fetch fresh access token from gcloud
$oldErrorAction = $ErrorActionPreference
$ErrorActionPreference = 'SilentlyContinue'
$token = (gcloud auth print-access-token)
$ErrorActionPreference = $oldErrorAction

# Detect local virtual environment modal path
$modalPath = "modal"
if (Test-Path ".\venv\Scripts\modal.exe") {
    $modalPath = ".\venv\Scripts\modal.exe"
} elseif (Test-Path ".\venv\bin\modal") {
    $modalPath = ".\venv\bin\modal"
}

if (-not $token) {
    Write-Host "[!] Warning: Could not fetch access token via 'gcloud auth print-access-token'." -ForegroundColor Yellow
    Write-Host "Please make sure you are logged in locally (run 'gcloud auth login')." -ForegroundColor Yellow
    Write-Host "Executing modal command without Vertex AI token fallback..."
    & $modalPath run @args
} else {
    $maskedToken = $token.Substring(0, 12) + "..." + $token.Substring($token.Length - 8)
    Write-Host "[*] Automatically bound Vertex AI Access Token: $maskedToken" -ForegroundColor Green
    
    # Execute modal run forwarding all arguments along with the token argument
    & $modalPath run @args --google-access-token $token
}

.\venv\Scripts\modal.exe volume get --force gitmesh-storage /assets/animated_trellis_mesh_a_comfortable_modern_fabric_sofa_lowpoly.glb trellis_mesh_a_comfortable_modern_fabric_sofa_lowpoly.glb
