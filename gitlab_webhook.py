import os

import modal
from fastapi import HTTPException, Request

app = modal.App("gitmesh-webhook")

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


@app.function(
    image=webhook_image,
    secrets=[modal.Secret.from_name("gitmesh-keys")]
)
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
    import json
    import urllib.request
    import urllib.parse

    # Engine project: owns the .gitlab-ci.yml pipeline and the trigger token.
    gitlab_project_id = os.environ.get("GITLAB_PROJECT_ID", "")
    gitlab_url = os.environ.get("GITLAB_URL", "https://gitlab.com").rstrip("/")
    gitlab_trigger_token = os.environ.get("GITLAB_TRIGGER_TOKEN", "")
    gitlab_api_token = os.environ.get("GITLAB_API_TOKEN", "")
    gitlab_trigger_ref = os.environ.get("GITLAB_TRIGGER_REF", "main")
    gitlab_webhook_secret = os.environ.get("GITLAB_WEBHOOK_SECRET", "")

    if not gitlab_project_id or not gitlab_trigger_token:
        raise HTTPException(
            status_code=500,
            detail=f"Missing required env vars: GITLAB_PROJECT_ID and/or GITLAB_TRIGGER_TOKEN (Found: {bool(gitlab_project_id)}, {bool(gitlab_trigger_token)})"
        )

    body = await req.json()

    # Resolve the source (target) project from the payload.
    source_project_id = str(body.get("project", {}).get("id", "") or "").strip()
    
    # Lookup registry asynchronously
    record = {}
    if source_project_id:
        try:
            registry = modal.Dict.from_name(REGISTRY_DICT_NAME, create_if_missing=True)
            record = await registry.get.aio(source_project_id) or {}
        except Exception as e:
            print(f"[Registry] lookup failed for project {source_project_id}: {e}")

    # Per-project webhook secret if onboarded, else engine-level default.
    expected_secret = record.get("webhook_secret") or gitlab_webhook_secret
    if expected_secret:
        inbound_token = req.headers.get("x-gitlab-token", "")
        if inbound_token != expected_secret:
            raise HTTPException(status_code=401, detail="Invalid webhook token")

    # Check if the webhook event is "Issue Opened" or "Issue Comment Created"
    object_kind = body.get("object_kind")
    action = body.get("object_attributes", {}).get("action")
    noteable_type = body.get("object_attributes", {}).get("noteable_type")
    note_text = body.get("object_attributes", {}).get("note", "")

    is_issue_open = (object_kind == "issue" and action == "open")
    is_issue_comment = (object_kind == "note" and noteable_type == "Issue")

    if not (is_issue_open or is_issue_comment):
        return {"status": "ignored_event", "reason": "Not an issue open or issue comment event"}

    if is_issue_open:
        issue_title = body.get("object_attributes", {}).get("title", "")
        issue_desc = body.get("object_attributes", {}).get("description", "")
        issue_iid = body.get("object_attributes", {}).get("iid", "")

        # Only trigger pipeline if the issue title starts with "MeshGen:" (case-insensitive)
        if not issue_title.lower().startswith("meshgen:"):
            print(f"Ignored: Issue '{issue_title}' doesn't start with 'MeshGen:'")
            return {"status": "ignored", "reason": "Missing MeshGen: prefix"}

        # Extract the pure prompt by stripping "MeshGen:" from it
        prompt = issue_title.split(":", 1)[1].strip()

    elif is_issue_comment:
        # Check if the comment starts with "/meshgen" (case-insensitive)
        if not note_text.lower().startswith("/meshgen"):
            print("Ignored: Comment doesn't start with '/meshgen'")
            return {"status": "ignored", "reason": "Missing /meshgen slash command"}

        # Extract the prompt following the "/meshgen" slash command
        parts = note_text.split(None, 1)
        prompt = parts[1].strip() if len(parts) > 1 else ""
        if not prompt:
            print("Ignored: /meshgen command contains no prompt")
            return {"status": "ignored", "reason": "Empty prompt description"}

        # Extract issue info from the root "issue" payload object
        issue_title = body.get("issue", {}).get("title", "")
        issue_desc = body.get("issue", {}).get("description", "")
        issue_iid = body.get("issue", {}).get("iid", "")

    # Target project metadata for routing comments/uploads back.
    target_project_id = source_project_id or gitlab_project_id
    target_gitlab_url = (record.get("gitlab_url") or gitlab_url).rstrip("/")
    ref = record.get("trigger_ref") or gitlab_trigger_ref

    # Duplicate-run guard: if a pipeline for the same issue is already
    # pending/running, skip creating another one.
    if gitlab_api_token:
        def _get_json(api_url: str):
            req_obj = urllib.request.Request(
                api_url,
                headers={"PRIVATE-TOKEN": gitlab_api_token},
            )
            with urllib.request.urlopen(req_obj, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _find_active_duplicate() -> str:
            base = f"{gitlab_url}/api/v4/projects/{gitlab_project_id}"
            for status in ("running", "pending"):
                try:
                    pipelines = _get_json(
                        f"{base}/pipelines?status={status}&order_by=id&sort=desc&per_page=25"
                    )
                except Exception as e:
                    print(f"Duplicate check skipped ({status} list failed): {e}")
                    continue

                for pipeline in pipelines:
                    pipeline_id = str(pipeline.get("id", "") or "").strip()
                    if not pipeline_id:
                        continue
                    try:
                        vars_list = _get_json(f"{base}/pipelines/{pipeline_id}/variables")
                    except Exception:
                        continue

                    vmap = {
                        str(v.get("key", "")): str(v.get("value", ""))
                        for v in vars_list if isinstance(v, dict)
                    }
                    same_issue = str(vmap.get("ISSUE_IID", "")).strip() == str(issue_iid).strip()
                    same_target = str(vmap.get("TARGET_PROJECT_ID", target_project_id)).strip() == str(target_project_id).strip()
                    if same_issue and same_target:
                        return pipeline_id
            return ""

        existing_pipeline_id = _find_active_duplicate()
        if existing_pipeline_id:
            print(
                f"Duplicate pipeline suppressed for issue #{issue_iid} "
                f"(target project {target_project_id}). Existing pipeline: {existing_pipeline_id}"
            )
            return {
                "status": "duplicate_suppressed",
                "existing_pipeline_id": existing_pipeline_id,
                "target_project_id": target_project_id,
            }

    print(f"Triggering 3D Pipeline for prompt: {prompt} (target project {target_project_id})")

    url = f"{gitlab_url}/api/v4/projects/{gitlab_project_id}/trigger/pipeline"
    form_data = urllib.parse.urlencode({
        "token": gitlab_trigger_token,
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
