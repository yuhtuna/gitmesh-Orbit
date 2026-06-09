# GitMesh

GitMesh is a remote-first GitLab + Modal pipeline for automated 3D asset generation and animation delivery.

This repository is organized for production operation from GitLab issue triggers, not local desktop execution.

## What This Project Does

When a user opens a GitLab issue with title prefix `MeshGen:`, GitMesh can:

1. Trigger a GitLab pipeline via webhook
2. Run staged 3D generation and processing on Modal
3. Post progress updates back to the GitLab issue
4. Upload final outputs and close the issue (optional)

## Remote Components

Core files for remote operation:

1. `.gitlab-ci.yml`
   - CI orchestration, stage order, issue comments, completion logic
2. `setup_remote.ps1`
   - Bootstrap utility for Modal deploy + secrets + GitLab vars + webhook
3. `gitlab_webhook.py`
   - Modal webhook endpoint; triggers GitLab pipeline for new issues
4. `modal_app.py`
   - Compute stage implementations (generation, segmentation, validation, animation)
5. `.env`
   - Operator-provided values used by bootstrap
6. `ADK_MIGRATION_TRACKER.md`
   - ADK migration status and role map

## End-to-End Flow

1. User opens issue: `MeshGen: <asset request>`
2. GitLab webhook calls deployed `gitlab_webhook.py`
3. Webhook triggers GitLab pipeline with issue variables
4. GitLab CI invokes Modal stages in sequence
5. Issue receives progress/status comments
6. Final outputs are posted and issue may auto-close

## Configuration Reference

### A) Required `.env` Values (Operator Input)

```text
GCP_PROJECT_ID=...
GCP_SERVICE_ACCOUNT_JSON={...}
GITLAB_PROJECT_ID=...
GITLAB_API_TOKEN=...
GITLAB_TRIGGER_TOKEN=...
GITLAB_WEBHOOK_SECRET=...
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
```

### B) Common Optional `.env` Values

```text
GITLAB_URL=https://gitlab.com
GITLAB_TRIGGER_REF=main
LLM_PROVIDER=vertex
IMAGE_MODEL=gemini-3.1-flash-image
AUTO_CLOSE_ISSUE=true
WEBHOOK_URL=
USE_ADK_ORCHESTRATOR=true
ADK_HARD_FAIL=false
PIPELINE_DRY_RUN=false
```

### C) Runtime Constraints in CI

Current remote CI rules:

1. `LLM_PROVIDER` is expected to be `vertex`
2. `GEMINI_API_KEY` must not be set for production GitLab runs
3. Modal secret `gitmesh-keys` must exist before stage execution

ADK orchestration controls:

1. `USE_ADK_ORCHESTRATOR=true` runs the ADK orchestrator first
2. `ADK_HARD_FAIL=false` allows fallback to the direct Modal chain if ADK fails during migration
3. `ADK_HARD_FAIL=true` makes the pipeline fail immediately if ADK fails
4. `PIPELINE_DRY_RUN=false` is required for real product-quality runs

## One-Time Setup

### 1) Cloud prerequisites

1. Create/select Google Cloud project
2. Enable Vertex AI API
3. Create service account with Vertex permissions
4. Capture service account JSON for `.env`

### 2) Fill `.env`

Copy from `.env.example` and populate all required values.

### 3) Run bootstrap

```powershell
pwsh ./setup_remote.ps1
```

Bootstrap responsibilities:

1. Validate `.env`
2. Create/update Modal secret `gitmesh-keys`
3. Deploy `gitlab_webhook.py`
4. Deploy `modal_app.py`
5. Upsert GitLab CI/CD variables
6. Create/update GitLab issue webhook

### 4) Verify setup

1. Confirm pipeline variables exist in GitLab project settings
2. Confirm webhook points to deployed Modal endpoint
3. Confirm Modal secret `gitmesh-keys` exists

## Bootstrap Variants

Useful `setup_remote.ps1` flags:

```powershell
pwsh ./setup_remote.ps1 -DryRun
pwsh ./setup_remote.ps1 -SkipDeploy
pwsh ./setup_remote.ps1 -SkipGitLabApi
pwsh ./setup_remote.ps1 -ProtectSensitiveVars
pwsh ./setup_remote.ps1 -WebhookUrl https://<your-webhook-url>
```

When to use:

1. `-DryRun`: config validation only
2. `-SkipDeploy`: update GitLab vars/webhook without redeploying Modal apps
3. `-SkipGitLabApi`: deploy Modal apps and secret only
4. `-WebhookUrl`: force webhook URL when auto-detect fails

## GitLab CI Jobs

`.gitlab-ci.yml` currently includes:

1. `bootstrap_modal_remote` (manual)
   - Bootstraps Modal and webhook from GitLab runner context
2. `run_gitmesh_pipeline` (triggered)
   - Runs staged compute pipeline
   - Runs ADK-first when `USE_ADK_ORCHESTRATOR=true`
   - Falls back to the direct Modal chain when `ADK_HARD_FAIL=false`
   - Supports `PIPELINE_DRY_RUN=true` for control-plane validation only

## Operating the Pipeline

### Trigger from issue

Open GitLab issue title:

```text
MeshGen: lowpoly pirate chest with metal latch
```

### Expected stage progression

1. Stage 1: Prompt analysis
2. Stage 2: Reference image
3. Stage 3: Mesh generation
4. Stage 3b: GLB validation
5. Stage 4: Segmentation
6. Stage 7: Part labeling
7. Stage 8: Animation plan
8. Stage 9: Plan validation
9. Stage 10: Final render/export

## Troubleshooting

### Error: Could not connect to the Modal server

Likely causes:

1. Network/proxy path to `api.modal.com`
2. Missing/invalid `MODAL_TOKEN_ID` or `MODAL_TOKEN_SECRET`
3. Modal CLI auth not available in execution context

Checks:

1. Validate Modal tokens in GitLab CI variables
2. Run bootstrap from a network that can reach Modal
3. Use `bootstrap_modal_remote` job if local network is restricted

### Error: Modal secret `gitmesh-keys` not found

Fix:

1. Re-run `setup_remote.ps1`
2. Or run manual `bootstrap_modal_remote` CI job

### Error: Blender subprocess no file `xvfb-run`

Current behavior in `modal_app.py`:

1. Uses `xvfb-run` when present
2. Falls back to direct `blender -b` when missing

Action:

1. Redeploy `modal_app.py` so fallback logic is active remotely

### Pipeline starts but no webhook trigger

Check:

1. GitLab webhook configured for Issue events
2. Secret token matches `GITLAB_WEBHOOK_SECRET`
3. Issue title starts with `MeshGen:`

## Security Notes

1. Treat `.env` as sensitive; never commit real secrets
2. Use masked GitLab CI variables for tokens/secrets
3. Rotate credentials immediately if exposed
4. Keep service account permissions minimal (least privilege)

## ADK Migration Status

The repository is actively migrating to ADK-first orchestration.

See `ADK_MIGRATION_TRACKER.md` for:

1. Role-based logical agents
2. Function mappings
3. Cutover phases and rollback strategy

## Scope

This README intentionally focuses on remote GitLab production operation.

## Local Testing (Optional)

This section is optional. If you only care about remote GitLab operation, you can ignore everything below.

Local/testing helper files:

1. `agent.py`
   - ADK orchestration test entrypoint
2. `run_modal.ps1`
   - Windows helper wrapper for `modal run`
3. `run_modal.sh`
   - Bash helper wrapper for `modal run`
4. `run_modal_download.ps1`
   - Windows helper for pulling assets from Modal volume
5. `server.ts` and `package.json`
   - Optional local Node/TS interface for testing workflows
6. `Dockerfile.pipeline`
   - Local container replica for dependency/runtime debugging

Minimal local smoke test example:

```powershell
pwsh ./run_modal.ps1 modal_app.py::validate_glb --issue-iid "1" --gitlab-token "<token>"
```

Again, none of the above is required for normal remote GitLab pipeline usage.
