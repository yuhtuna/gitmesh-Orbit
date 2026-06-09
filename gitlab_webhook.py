import os

import modal
from fastapi import HTTPException, Request

app = modal.App("gitmesh-webhook")

# Engine project: owns the .gitlab-ci.yml pipeline and the trigger token.
GITLAB_PROJECT_ID = os.environ.get("GITLAB_PROJECT_ID", "")
GITLAB_URL = os.environ.get("GITLAB_URL", "https://gitlab.com").rstrip("/")
GITLAB_TRIGGER_TOKEN = os.environ.get("GITLAB_TRIGGER_TOKEN", "")
GITLAB_TRIGGER_REF = os.environ.get("GITLAB_TRIGGER_REF", "main")
GITLAB_WEBHOOK_SECRET = os.environ.get("GITLAB_WEBHOOK_SECRET", "")

REGISTRY_DICT_NAME = "gitmesh-project-registry"

webhook_image = modal.Image.debian_slim().pip_install("fastapi[standard]")


def _lookup_registry(project_id: str) -> dict:
    """Return the onboarding record for a source project, or {} if not registered."""
    if not project_id:
        return {}
    try:
        registry = modal.Dict.from_name(REGISTRY_DICT_NAME, create_if_missing=True)
        return registry.get(str(project_id)) or {}
    except Exception as e:
        print(f"[Registry] lookup failed for project {project_id}: {e}")
        return {}


@app.function(image=webhook_image)
@modal.fastapi_endpoint(method="POST")
async def gitlab_issue_listener(req: Request):
    """
    Receives JSON payloads from GitLab Webhooks.
    Filters specifically for when an 'issue' is 'opened'.

    Multi-project routing:
      - The source project is read from the webhook payload (project.id).
      - If the source project is registered, its per-project webhook secret and
        trigger ref are used. Otherwise the engine-level env defaults apply
        (backward-compatible single-project behavior).
      - The engine pipeline is always triggered, with TARGET_* variables so the
        compute stages report back to the originating project.
    """
    import urllib.request
    import urllib.parse

    if not GITLAB_PROJECT_ID or not GITLAB_TRIGGER_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="Missing required env vars: GITLAB_PROJECT_ID and/or GITLAB_TRIGGER_TOKEN"
        )

    body = await req.json()

    # Resolve the source (target) project from the payload.
    source_project_id = str(body.get("project", {}).get("id", "") or "").strip()
    record = _lookup_registry(source_project_id)

    # Per-project webhook secret if onboarded, else engine-level default.
    expected_secret = record.get("webhook_secret") or GITLAB_WEBHOOK_SECRET
    if expected_secret:
        inbound_token = req.headers.get("x-gitlab-token", "")
        if inbound_token != expected_secret:
            raise HTTPException(status_code=401, detail="Invalid webhook token")

    # Check if the webhook event is "Issue Opened"
    if body.get("object_kind") == "issue" and body.get("object_attributes", {}).get("action") == "open":
        # Extract issue info
        issue_title = body.get("object_attributes", {}).get("title", "")
        issue_desc = body.get("object_attributes", {}).get("description", "")
        issue_iid = body.get("object_attributes", {}).get("iid", "")

        # Only trigger pipeline if the issue title starts with "MeshGen:" (case-insensitive)
        if not issue_title.lower().startswith("meshgen:"):
            print(f"Ignored: Issue '{issue_title}' doesn't start with 'MeshGen:'")
            return {"status": "ignored", "reason": "Missing MeshGen: prefix"}

        # Extract the pure prompt by stripping "MeshGen:" from it
        prompt = issue_title.split(":", 1)[1].strip()

        # Target project metadata for routing comments/uploads back.
        target_project_id = source_project_id or GITLAB_PROJECT_ID
        target_gitlab_url = (record.get("gitlab_url") or GITLAB_URL).rstrip("/")
        ref = record.get("trigger_ref") or GITLAB_TRIGGER_REF
        print(f"Triggering 3D Pipeline for prompt: {prompt} (target project {target_project_id})")

        url = f"{GITLAB_URL}/api/v4/projects/{GITLAB_PROJECT_ID}/trigger/pipeline"
        form_data = urllib.parse.urlencode({
            "token": GITLAB_TRIGGER_TOKEN,
            "ref": ref,
            "variables[ISSUE_TITLE]": prompt,
            "variables[ISSUE_DESC]": issue_desc,
            "variables[ISSUE_IID]": issue_iid,
            "variables[TARGET_PROJECT_ID]": target_project_id,
            "variables[TARGET_GITLAB_URL]": target_gitlab_url,
        }).encode("utf-8")

        # Call out to GitLab CI API
        request = urllib.request.Request(url, data=form_data)
        try:
            with urllib.request.urlopen(request) as response:
                print("Pipeline triggered successfully! Response:", response.read().decode())
                return {
                    "status": "triggered",
                    "pipeline_branch": ref,
                    "target_project_id": target_project_id,
                }
        except Exception as e:
            print(f"Error triggering pipeline: {e}")
            return {"status": "error", "message": str(e)}

    # Ignore comments, issue updates, or other events
    return {"status": "ignored_event"}
