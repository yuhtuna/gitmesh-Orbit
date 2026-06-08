#!/bin/bash
# ==============================================================================
# GitMesh Local Testing CLI Wrapper (Bash)
# Automatically fetches a fresh gcloud access token and binds it to Modal
# ==============================================================================

# Force Python stdout/stderr encoding to UTF-8 to prevent Unicode encoding crashes
export PYTHONIOENCODING="utf-8"

# Fetch fresh access token from gcloud
TOKEN=$(gcloud auth print-access-token 2>/dev/null)

# Detect local virtual environment modal path
MODAL_BIN="modal"
if [ -f "./venv/Scripts/modal.exe" ]; then
  MODAL_BIN="./venv/Scripts/modal.exe"
elif [ -f "./venv/bin/modal" ]; then
  MODAL_BIN="./venv/bin/modal"
fi

if [ -z "$TOKEN" ]; then
  echo "[!] Warning: Could not fetch access token via 'gcloud auth print-access-token'."
  echo "Please make sure you are logged in locally (run 'gcloud auth login')."
  echo "Executing modal command without Vertex AI token fallback..."
  "$MODAL_BIN" run "$@"
else
  # Print masked token prefix for debugging
  MASKED_TOKEN="${TOKEN:0:12}...${TOKEN: -8}"
  echo "[*] Automatically bound Vertex AI Access Token: $MASKED_TOKEN"
  
  # Run modal run forwarding all arguments along with the token argument
  "$MODAL_BIN" run "$@" --google-access-token "$TOKEN"
fi
