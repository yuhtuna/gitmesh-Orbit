import modal
from fastapi import Request

app = modal.App("gitmesh-webhook")

GITLAB_PROJECT_ID = "82717291"
GITLAB_TRIGGER_TOKEN = "glptt-F9ycN4syYk_UAhK3NnoM"

webhook_image = modal.Image.debian_slim().pip_install("fastapi[standard]")

@app.function(image=webhook_image)
@modal.fastapi_endpoint(method="POST")
async def gitlab_issue_listener(req: Request):
    """
    Receives JSON payloads from GitLab Webhooks.
    Filters specifically for when an 'issue' is 'opened'.
    """
    import urllib.request
    import urllib.parse
    
    body = await req.json()
    
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
        print(f"Triggering 3D Pipeline for prompt: {prompt}")

        # 'ref' is simply the branch name you want the pipeline to run on.
        # We will use 'main'. If your default branch is 'master', change this to 'master'.
        ref = "main"

        url = f"https://gitlab.com/api/v4/projects/{GITLAB_PROJECT_ID}/trigger/pipeline"
        form_data = urllib.parse.urlencode({
            "token": GITLAB_TRIGGER_TOKEN,
            "ref": ref,
            "variables[ISSUE_TITLE]": prompt,
            "variables[ISSUE_DESC]": issue_desc,
            "variables[ISSUE_IID]": issue_iid
        }).encode("utf-8")

        # Call out to GitLab CI API
        request = urllib.request.Request(url, data=form_data)
        try:
            with urllib.request.urlopen(request) as response:
                print("Pipeline triggered successfully! Response:", response.read().decode())
                return {"status": "triggered", "pipeline_branch": ref}
        except Exception as e:
            print(f"Error triggering pipeline: {e}")
            return {"status": "error", "message": str(e)}

    # Ignore comments, issue updates, or other events
    return {"status": "ignored_event"}
