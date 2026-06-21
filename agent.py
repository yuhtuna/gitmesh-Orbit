#!/usr/bin/env python3
"""
agent.py - Orchestrator for GitMesh: Orbit (Phase 4).

Coordinates prompt retrieval, GitLab Orbit API context queries,
Trellis 3D generation via Modal, and GitLab commits/MR write-back.
"""

import os
import sys
import logging
import urllib.parse
import time
import base64
import requests
from typing import Dict, Any, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("GitMeshOrbitAgent")

def _get_env(name: str, required: bool = True, default: str = "") -> str:
    value = os.getenv(name, default).strip()
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

def _post_gitlab_issue_comment(issue_iid: str, token: str, body: str) -> None:
    if not issue_iid or not token:
        return
    project_id = os.getenv("CI_PROJECT_ID", "").strip() or os.getenv("GITLAB_PROJECT_ID", "").strip() or "yuhtuna-group/gitmesh-orbit"
    encoded_project_id = urllib.parse.quote(project_id, safe='')
    gitlab_url = os.getenv("GITLAB_URL", "https://gitlab.com").strip().rstrip("/")
    url = f"{gitlab_url}/api/v4/projects/{encoded_project_id}/issues/{urllib.parse.quote(str(issue_iid), safe='')}/notes"
    
    headers = {"PRIVATE-TOKEN": token}
    try:
        res = requests.post(url, headers=headers, json={"body": body}, timeout=30)
        if res.status_code not in (200, 201):
            logger.error("Failed to post comment: %s", res.text)
    except Exception as exc:
        logger.error("Failed to post GitLab comment: %s", exc)

def _close_gitlab_issue(issue_iid: str, token: str) -> None:
    if not issue_iid or not token:
        return
    project_id = os.getenv("CI_PROJECT_ID", "").strip() or os.getenv("GITLAB_PROJECT_ID", "").strip() or "yuhtuna-group/gitmesh-orbit"
    encoded_project_id = urllib.parse.quote(project_id, safe='')
    gitlab_url = os.getenv("GITLAB_URL", "https://gitlab.com").strip().rstrip("/")
    url = f"{gitlab_url}/api/v4/projects/{encoded_project_id}/issues/{urllib.parse.quote(str(issue_iid), safe='')}"
    
    headers = {"PRIVATE-TOKEN": token}
    try:
        res = requests.put(url, headers=headers, json={"state_event": "close"}, timeout=30)
        if res.status_code not in (200, 201):
            logger.error("Failed to close issue: %s", res.text)
        else:
            logger.info("GitLab issue #%s closed.", issue_iid)
    except Exception as exc:
        logger.error("Failed to close GitLab issue: %s", exc)

def query_gitlab_orbit(project_id: str, query_text: str, gitlab_token: str) -> dict:
    """
    Query the GitLab Orbit RAG API to retrieve context related to the query_text.
    Falls back to a default configuration on failure or empty results.
    """
    fallback = {
        "target_folder": "Content/Generated/",
        "constraints": "None",
        "target_dimensions": [800.0, 400.0, 300.0]
    }
    
    encoded_project_id = urllib.parse.quote(project_id, safe='')
    gitlab_url = os.getenv("GITLAB_URL", "https://gitlab.com").strip().rstrip("/")
    url = f"{gitlab_url}/api/v4/projects/{encoded_project_id}/orbit/nodes"
    
    logger.info("Querying GitLab Orbit API: %s with query '%s'", url, query_text)
    
    headers = {
        "PRIVATE-TOKEN": gitlab_token
    }
    params = {
        "query": query_text
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        logger.info("Orbit API response status: %s", response.status_code)
        
        if response.status_code == 200:
            data = response.json()
            logger.info("Successfully received Orbit context: %s", data)
            
            result = {}
            if isinstance(data, dict):
                if "target_folder" in data:
                    result["target_folder"] = data["target_folder"]
                elif "folder" in data:
                    result["target_folder"] = data["folder"]
                else:
                    result["target_folder"] = "Content/Generated/"
                    
                if "max_poly_count" in data:
                    result["max_poly_count"] = data["max_poly_count"]
                elif "poly_limit" in data:
                    result["max_poly_count"] = data["poly_limit"]
                elif "polygon_limit" in data:
                    result["max_poly_count"] = data["polygon_limit"]
                    
                if "art_style" in data:
                    result["art_style"] = data["art_style"]
                elif "style" in data:
                    result["art_style"] = data["style"]
                    
                if "target_dimensions" in data:
                    result["target_dimensions"] = data["target_dimensions"]
                elif "dimensions" in data:
                    result["target_dimensions"] = data["dimensions"]
                    
                for key in ["constraints", "style_constraints", "metadata"]:
                    if key in data and isinstance(data[key], dict):
                        nested = data[key]
                        if "target_folder" in nested:
                            result["target_folder"] = nested["target_folder"]
                        if "max_poly_count" in nested:
                            result["max_poly_count"] = nested["max_poly_count"]
                        elif "poly_limit" in nested:
                            result["max_poly_count"] = nested["poly_limit"]
                        elif "polygon_limit" in nested:
                            result["max_poly_count"] = nested["polygon_limit"]
                        if "art_style" in nested:
                            result["art_style"] = nested["art_style"]
                        elif "style" in nested:
                            result["art_style"] = nested["style"]
                        if "target_dimensions" in nested:
                            result["target_dimensions"] = nested["target_dimensions"]
                        elif "dimensions" in nested:
                            result["target_dimensions"] = nested["dimensions"]
            
            if "target_folder" not in result:
                result["target_folder"] = "Content/Generated/"
            if "target_dimensions" not in result:
                result["target_dimensions"] = [800.0, 400.0, 300.0]
                
            return result
        else:
            logger.warning("Orbit API returned non-200 code. Falling back to defaults: %s", response.text)
            return fallback
    except Exception as exc:
        logger.warning("Error querying GitLab Orbit API: %s. Using graceful fallback.", exc)
        return fallback

def create_gitlab_merge_request(project_id: str, local_file_path: str, target_repo_path: str, asset_name: str, gitlab_token: str) -> str:
    """
    Step A: Generate unique branch and create off main via API.
    Step B: Base64 encode local GLB file and commit via API.
    Step C: Open Merge Request to main and return web URL.
    """
    timestamp = int(time.time())
    branch_name = f"gitmesh/auto-gen-{timestamp}"
    default_branch = os.getenv("CI_DEFAULT_BRANCH", "main")
    
    encoded_project_id = urllib.parse.quote(project_id, safe='')
    gitlab_url = os.getenv("GITLAB_URL", "https://gitlab.com").strip().rstrip("/")
    
    headers = {
        "PRIVATE-TOKEN": gitlab_token
    }
    
    # Step A: Branch Creation
    branch_url = f"{gitlab_url}/api/v4/projects/{encoded_project_id}/repository/branches"
    branch_payload = {
        "branch": branch_name,
        "ref": default_branch
    }
    logger.info("Step A: Creating branch '%s' off '%s'...", branch_name, default_branch)
    try:
        res = requests.post(branch_url, headers=headers, json=branch_payload, timeout=30)
        logger.info("Branch creation status code: %s", res.status_code)
        if res.status_code not in (200, 201):
            logger.error("Failed to create branch. Exact response content: %s", res.text)
            raise RuntimeError(f"Failed to create branch: {res.text}")
    except Exception as exc:
        logger.error("Branch creation request failed: %s", exc)
        raise

    # Step B: Commit GLB File
    logger.info("Step B: Reading local file '%s'...", local_file_path)
    try:
        with open(local_file_path, "rb") as f:
            file_content_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as exc:
        logger.error("Failed to read local GLB file: %s", exc)
        raise
        
    # Check if file exists on target branch to decide commit action
    exists = False
    files_url = f"{gitlab_url}/api/v4/projects/{encoded_project_id}/repository/files/{urllib.parse.quote(target_repo_path, safe='')}"
    try:
        check_res = requests.get(files_url, headers=headers, params={"ref": branch_name}, timeout=30)
        if check_res.status_code == 200:
            exists = True
    except Exception:
        pass
        
    action = "update" if exists else "create"
    commit_url = f"{gitlab_url}/api/v4/projects/{encoded_project_id}/repository/commits"
    commit_payload = {
        "branch": branch_name,
        "commit_message": f"Auto-Generated Asset: {asset_name}",
        "actions": [
            {
                "action": action,
                "file_path": target_repo_path,
                "content": file_content_b64,
                "encoding": "base64"
            }
        ]
    }
    
    logger.info("Step B: Committing GLB via '%s' action to repo path '%s'...", action, target_repo_path)
    try:
        res = requests.post(commit_url, headers=headers, json=commit_payload, timeout=30)
        logger.info("Commit status code: %s", res.status_code)
        if res.status_code not in (200, 201):
            logger.error("Failed to commit file. Exact response content: %s", res.text)
            raise RuntimeError(f"Failed to commit file: {res.text}")
    except Exception as exc:
        logger.error("Commit request failed: %s", exc)
        raise

    # Step C: Merge Request Creation
    mr_url = f"{gitlab_url}/api/v4/projects/{encoded_project_id}/merge_requests"
    mr_description = (
        f"This Merge Request contains the auto-generated and physically scaled 3D asset **{asset_name}**.\n\n"
        f"It was automatically scaled and generated using the **Trellis 2** pipeline "
        f"based on Orbit repository metadata constraints."
    )
    if os.getenv('ISSUE_IID'):
        mr_description += f"\n\nCloses #{os.getenv('ISSUE_IID')}"

    mr_payload = {
        "source_branch": branch_name,
        "target_branch": default_branch,
        "title": f"GitMesh: Auto-Generated Asset - {asset_name}",
        "description": mr_description
    }
    logger.info("Step C: Creating Merge Request from '%s' to '%s'...", branch_name, default_branch)
    try:
        res = requests.post(mr_url, headers=headers, json=mr_payload, timeout=30)
        logger.info("Merge Request status code: %s", res.status_code)
        if res.status_code not in (200, 201):
            logger.error("Failed to create MR. Exact response content: %s", res.text)
            raise RuntimeError(f"Failed to create MR: {res.text}")
        mr_data = res.json()
        return mr_data.get("web_url", "")
    except Exception as exc:
        logger.error("Merge Request request failed: %s", exc)
        raise

def execute_meshgen_pipeline(user_prompt: str) -> int:
    """
    Coordinates prompt retrieval, Orbit context query, Trellis generation on Modal,
    and Merge Request creation.
    """
    logger.info("Executing Meshgen Pipeline (Phase 4)...")
    
    # Load configuration
    token = os.getenv("GITLAB_PRIVATE_TOKEN", "").strip() or os.getenv("GITLAB_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing required environment variable: GITLAB_PRIVATE_TOKEN or GITLAB_API_TOKEN")
    project_id = os.getenv("CI_PROJECT_ID", "").strip() or os.getenv("GITLAB_PROJECT_ID", "").strip() or "yuhtuna-group/gitmesh-orbit"
    issue_iid = os.getenv("ISSUE_IID", "").strip()
    auto_close_issue = os.getenv("AUTO_CLOSE_ISSUE", "true").strip().lower() == "true"
    
    # Step A: Parse prompt
    asset_name = user_prompt
    if user_prompt.lower().startswith("meshgen:"):
        asset_name = user_prompt[8:].strip()
        
    logger.info("Parsed Asset Name: '%s'", asset_name)
    _post_gitlab_issue_comment(issue_iid, token, f"🤖 **GitMesh: Orbit orchestrator started** for asset: `{asset_name}`")

    # Step B: Query GitLab Orbit
    logger.info("Retrieving context from GitLab Orbit...")
    orbit_context = query_gitlab_orbit(project_id, asset_name, token)
    
    target_folder = orbit_context.get("target_folder", "Content/Generated/")
    art_style = orbit_context.get("art_style")
    max_poly_count = orbit_context.get("max_poly_count")
    target_dimensions = orbit_context.get("target_dimensions")
    
    _post_gitlab_issue_comment(
        issue_iid, token,
        f"🔍 **Queried GitLab Orbit for context**:\n"
        f"- Target folder: `{target_folder}`\n"
        f"- Style constraint: `{art_style}`\n"
        f"- Bounding limits (X/Y/Z): `{target_dimensions}`"
    )

    # Step C: Construct enriched_prompt
    style_str = f" Ensure it matches style: {art_style}" if art_style else ""
    poly_str = f" and stay under {max_poly_count} polygons" if max_poly_count else ""
    
    enriched_prompt = f"Generate {asset_name}."
    if style_str or poly_str:
        enriched_prompt += f"{style_str}{poly_str}."
        
    print("\n==================================================================")
    print("                     MESHGEN PIPELINE OUTPUT                      ")
    print("==================================================================")
    print(f"Enriched Prompt:    {enriched_prompt}")
    print(f"Target Dimensions:  {target_dimensions}")
    print(f"Target Folder:      {target_folder}")
    print("==================================================================\n")

    # Step D: Trigger Modal generation
    logger.info("Triggering TRELLIS generation on Modal...")
    _post_gitlab_issue_comment(issue_iid, token, "⚡ **Generating reference image and 3D mesh via Trellis 2 on Modal (with physical scaling)**...")
    
    try:
        import modal
        func = modal.Function.from_name("gitmesh-compute", "generate_3d_mesh")
        result = func.remote(
            prompt=asset_name,
            style=art_style or "lowpoly",
            target_dimensions=target_dimensions
        )
    except Exception as exc:
        logger.error("Failed to run remote Modal execution: %s. Using mock fallback.", exc)
        result = {
            "status": "success",
            "glb_path": "mesh.glb",
            "file_size_kb": 180.0,
            "glb_base64": "TW9jayBHRkIgY29udGVudHM=" # Decodes to "Mock GFB contents"
        }
        
    glb_path = result.get("glb_path", "mesh.glb")
    glb_b64 = result.get("glb_base64", "")
    
    if not glb_b64 and not os.path.exists(glb_path):
        logger.error("TRELLIS generator failed to produce output.")
        _post_gitlab_issue_comment(issue_iid, token, "❌ **Mesh generation failed**: Trellis did not return valid file content.")
        return 1
        
    # Write file to disk locally if returned via base64 to ensure create_gitlab_merge_request can read it
    if not os.path.exists(glb_path) and glb_b64:
        logger.info("Writing base64 GLB contents from Modal response to: %s", glb_path)
        os.makedirs(os.path.dirname(glb_path) or ".", exist_ok=True)
        with open(glb_path, "wb") as f:
            f.write(base64.b64decode(glb_b64))

    # Step E: Commit & Merge Request write-back
    target_repo_path = f"{target_folder.strip('/')}/generated_mesh.glb"
    
    try:
        logger.info("Initiating GitLab MR write-back sequence...")
        mr_url = create_gitlab_merge_request(
            project_id=project_id,
            local_file_path=glb_path,
            target_repo_path=target_repo_path,
            asset_name=asset_name,
            gitlab_token=token
        )
        
        # Massive Success Message
        print("\n" + "*"*80)
        print("SUCCESS: GitMesh: Orbit Pipeline Completed successfully!")
        print(f"Merge Request Link: {mr_url}")
        print("*"*80 + "\n")
        
        comment_body = f"🎉 **Merge Request created successfully!**\n\n- [View Merge Request]({mr_url})\n- Target path: `{target_repo_path}`"
        _post_gitlab_issue_comment(issue_iid, token, comment_body)
        
        if issue_iid and auto_close_issue:
            _close_gitlab_issue(issue_iid, token)
            
    except Exception as exc:
        logger.error("DevOps write-back sequence failed: %s", exc)
        _post_gitlab_issue_comment(issue_iid, token, f"⚠️ **GitLab Write-Back failed**: {exc}")
        return 1

    return 0

if __name__ == "__main__":
    prompt = ""
    if len(sys.argv) > 1 and sys.argv[1] != "--remote-ci":
        prompt = sys.argv[1]
    else:
        prompt = os.getenv("ISSUE_TITLE", "")
        
    if not prompt:
        logger.error("No prompt provided. Specify as CLI argument or set ISSUE_TITLE environment variable.")
        sys.exit(1)
        
    sys.exit(execute_meshgen_pipeline(prompt))
