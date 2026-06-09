"""GitMesh multi-project registry.

EXPERIMENTAL / NIGHTLY — under development, NOT production-tested. This module
backs the opt-in cross-project onboarding feature. It is dormant unless a
project is explicitly registered; the main single-project flow does not use it.

Stores per-target-project onboarding configuration in a Modal Dict so the
central webhook and compute stages can route GitLab comments, uploads, and
pipeline triggers to the correct external project.

Registry value schema (JSON-serializable):
    {
        "project_id": "123",
        "gitlab_url": "https://gitlab.com",
        "api_token": "<project access token>",
        "webhook_secret": "<per-project webhook secret>",
        "trigger_ref": "main",
        "auto_close": "true"
    }

The Dict is private to your Modal workspace. Treat stored tokens as secrets.
"""

import json
import sys

import modal

REGISTRY_DICT_NAME = "gitmesh-project-registry"

app = modal.App("gitmesh-registry")

registry_image = modal.Image.debian_slim()


def _registry():
    """Return the shared registry Dict, creating it on first use."""
    return modal.Dict.from_name(REGISTRY_DICT_NAME, create_if_missing=True)


def _normalize_key(project_id) -> str:
    return str(project_id).strip()


@app.function(image=registry_image)
def register_project(
    project_id: str,
    gitlab_url: str = "https://gitlab.com",
    api_token: str = "",
    webhook_secret: str = "",
    trigger_ref: str = "main",
    auto_close: str = "true",
) -> dict:
    """Create or update a target project's onboarding record."""
    key = _normalize_key(project_id)
    if not key:
        raise ValueError("project_id is required")

    record = {
        "project_id": key,
        "gitlab_url": (gitlab_url or "https://gitlab.com").rstrip("/"),
        "api_token": api_token or "",
        "webhook_secret": webhook_secret or "",
        "trigger_ref": trigger_ref or "main",
        "auto_close": auto_close or "true",
    }
    _registry()[key] = record

    safe = dict(record)
    safe["api_token"] = "***" if record["api_token"] else ""
    safe["webhook_secret"] = "***" if record["webhook_secret"] else ""
    print(f"[Registry] Registered project {key}: {json.dumps(safe)}")
    return safe


@app.function(image=registry_image)
def get_project(project_id: str) -> dict:
    """Return a target project's record, or an empty dict if not registered."""
    record = _registry().get(_normalize_key(project_id))
    return record or {}


@app.function(image=registry_image)
def remove_project(project_id: str) -> bool:
    """Delete a target project's record. Returns True if it existed."""
    key = _normalize_key(project_id)
    registry = _registry()
    if key in registry:
        del registry[key]
        print(f"[Registry] Removed project {key}")
        return True
    print(f"[Registry] Project {key} was not registered")
    return False


@app.function(image=registry_image)
def list_projects() -> list:
    """Return all registered project records with secrets redacted."""
    out = []
    for key in _registry().keys():
        record = _registry().get(key) or {}
        safe = dict(record)
        if safe.get("api_token"):
            safe["api_token"] = "***"
        if safe.get("webhook_secret"):
            safe["webhook_secret"] = "***"
        out.append(safe)
    return out


@app.local_entrypoint()
def main(
    action: str = "list",
    project_id: str = "",
    gitlab_url: str = "https://gitlab.com",
    api_token: str = "",
    webhook_secret: str = "",
    trigger_ref: str = "main",
    auto_close: str = "true",
):
    """CLI entrypoint: modal run project_registry.py --action register --project-id 123 ..."""
    if action == "register":
        result = register_project.remote(
            project_id=project_id,
            gitlab_url=gitlab_url,
            api_token=api_token,
            webhook_secret=webhook_secret,
            trigger_ref=trigger_ref,
            auto_close=auto_close,
        )
        print(json.dumps(result, indent=2))
    elif action == "get":
        print(json.dumps(get_project.remote(project_id), indent=2))
    elif action == "remove":
        print(json.dumps({"removed": remove_project.remote(project_id)}))
    elif action == "list":
        print(json.dumps(list_projects.remote(), indent=2))
    else:
        print(f"Unknown action: {action}", file=sys.stderr)
        raise SystemExit(2)
