# ==============================================================================
# GitMesh Local Testing CLI Wrapper (PowerShell)
# Automatically fetches a fresh gcloud access token and binds it to Modal
# ==============================================================================

# Force Python stdout/stderr encoding to UTF-8 to prevent Windows Unicode charmap crashes
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

# Detect local virtual environment modal path
$modalPath = "modal"
if (Test-Path ".\venv\Scripts\modal.exe") {
    $modalPath = ".\venv\Scripts\modal.exe"
} elseif (Test-Path ".\venv\bin\modal") {
    $modalPath = ".\venv\bin\modal"
}

# The modal app now natively uses Vertex AI SDK and ADC, 
# so we no longer need to pass google-access-token manually.
& $modalPath run @args
