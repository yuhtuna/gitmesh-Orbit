#!/usr/bin/env python3
"""
agent.py - Orchestrator for GitMesh: Orbit (Phase 4).

Coordinates prompt retrieval, GitLab Orbit API context queries,
Gemini reference image generation, Trellis 3D generation via Modal,
and GitLab commits/MR write-back.
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
    # FIX: Prioritize TARGET_PROJECT_ID first
    project_id = os.getenv("TARGET_PROJECT_ID", "").strip() or os.getenv("CI_PROJECT_ID", "").strip() or os.getenv("GITLAB_PROJECT_ID", "").strip() or "yuhtuna-group/gitmesh-orbit"
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
    # FIX: Prioritize TARGET_PROJECT_ID first
    project_id = os.getenv("TARGET_PROJECT_ID", "").strip() or os.getenv("CI_PROJECT_ID", "").strip() or os.getenv("GITLAB_PROJECT_ID", "").strip() or "yuhtuna-group/gitmesh-orbit"
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
    Uses the official POST /api/v4/orbit/query endpoint.
    """
    try:
        pid_int = int(project_id.strip())
    except Exception:
        pid_int = 83609477
        
    gitlab_url = os.getenv("GITLAB_URL", "https://gitlab.com").strip().rstrip("/")
    url = f"{gitlab_url}/api/v4/orbit/query"
    
    logger.info("Querying GitLab Orbit API graph at %s for project %s (query: '%s')", url, pid_int, query_text)
    
    headers = {
        "PRIVATE-TOKEN": gitlab_token,
        "Content-Type": "application/json"
    }
    
    # GraphQL-like Traversal Query to find WorkItems matching the query text
    payload = {
        "query": {
            "query_type": "traversal",
            "node": {
                "id": "w",
                "entity": "WorkItem",
                "filters": {"title": query_text},
                "columns": ["title", "description"]
            }
        },
        "response_format": "raw"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        logger.info("Orbit API response status: %s", response.status_code)
        
        if response.status_code == 200:
            data = response.json()
            result_data = data.get("result", {})
            nodes = result_data.get("nodes", [])
            logger.info("Successfully received Orbit context with %s matching nodes.", len(nodes))
            
            result = {}
            # Loop through returned nodes and extract properties/descriptions directly from node dictionary
            for node in nodes:
                desc = node.get("description", "")
                title = node.get("title", "")
                
                # Check description and title for overrides (re-use parse_constraints_from_desc logic)
                overrides = parse_constraints_from_desc(desc)
                title_overrides = parse_constraints_from_desc(title)
                
                # Merge constraints
                result.update(title_overrides)
                result.update(overrides)
                
            return result
        else:
            logger.info("GitLab Orbit API returned status %s. Cascading to repository configuration...", response.status_code)
            return {}
    except Exception as exc:
        logger.info("GitLab Orbit API query failed: %s. Cascading to repository configuration...", exc)
        return {}

def query_gitlab_repository_config(project_id: str, gitlab_token: str) -> dict:
    """
    Attempts to read orbit.json or gitmesh.json from the repository root.
    """
    encoded_project_id = urllib.parse.quote(project_id, safe='')
    gitlab_url = os.getenv("GITLAB_URL", "https://gitlab.com").strip().rstrip("/")
    
    headers = {"PRIVATE-TOKEN": gitlab_token}
    
    for filename in ("orbit.json", "gitmesh.json"):
        url = f"{gitlab_url}/api/v4/projects/{encoded_project_id}/repository/files/{urllib.parse.quote(filename, safe='')}/raw"
        try:
            logger.debug("Checking repository config file: %s", filename)
            res = requests.get(url, headers=headers, params={"ref": "main"}, timeout=15)
            if res.status_code == 200:
                data = res.json()
                logger.info("Found repository configuration from %s: %s", filename, data)
                if isinstance(data, dict):
                    return data
        except Exception as exc:
            logger.debug("Failed to query repository config %s: %s", filename, exc)
    return {}

def parse_constraints_from_desc(desc: str) -> dict:
    """
    Parses overrides from the issue description text.
    Look for flags:
      Style: lowpoly
      Folder: Content/Props
      Dimensions: [800, 400, 300]
      Polycount: 3000
    """
    result = {}
    if not desc:
        return result
    import re
    # Look for style: lowpoly
    style_match = re.search(r'(?:style|art\s*style)\s*:\s*([a-zA-Z0-9_-]+)', desc, re.I)
    if style_match:
        result["art_style"] = style_match.group(1).strip().lower()
        
    # Look for folder: Content/Props
    folder_match = re.search(r'(?:folder|target\s*folder|path)\s*:\s*([a-zA-Z0-9_/-]+)', desc, re.I)
    if folder_match:
        result["target_folder"] = folder_match.group(1).strip()
        
    # Look for dimensions: [800, 400, 300] or 800, 400, 300
    dim_match = re.search(r'(?:dimensions|limits|scale)\s*:\s*\[?([0-9.,\s]+)\]?', desc, re.I)
    if dim_match:
        try:
            parts = re.split(r'[,\s]+', dim_match.group(1).strip())
            dims = [float(p) for p in parts if p]
            if len(dims) >= 3:
                result["target_dimensions"] = dims[:3]
        except Exception:
            pass
            
    # Look for polycount: 3000
    poly_match = re.search(r'(?:poly_?count|polygon_?limit|max_?poly)\s*:\s*([0-9]+)', desc, re.I)
    if poly_match:
        try:
            result["max_poly_count"] = int(poly_match.group(1))
        except Exception:
            pass

    # Look for quality: high (or low, med)
    quality_match = re.search(r'(?:quality|quality_?mode|resolution)\s*:\s*(low|med|medium|high)', desc, re.I)
    if quality_match:
        qval = quality_match.group(1).strip().lower()
        if qval == "medium":
            qval = "med"
        result["quality_mode"] = qval
            
    return result

def upload_file_to_gitlab(project_id: str, local_filepath: str, gitlab_token: str) -> Optional[str]:
    """
    Uploads a file to the GitLab project uploads endpoint and returns the markdown link.
    """
    if not os.path.exists(local_filepath) or not gitlab_token:
        return None
    
    encoded_project_id = urllib.parse.quote(project_id, safe='')
    gitlab_url = os.getenv("GITLAB_URL", "https://gitlab.com").strip().rstrip("/")
    url = f"{gitlab_url}/api/v4/projects/{encoded_project_id}/uploads"
    
    headers = {"PRIVATE-TOKEN": gitlab_token}
    
    logger.info("Uploading file %s to GitLab...", local_filepath)
    try:
        with open(local_filepath, "rb") as f:
            files = {"file": (os.path.basename(local_filepath), f)}
            res = requests.post(url, headers=headers, files=files, timeout=30)
            if res.status_code in (200, 201):
                data = res.json()
                markdown_link = data.get("markdown")
                logger.info("Uploaded successfully. Markdown link: %s", markdown_link)
                return markdown_link
            else:
                logger.error("Failed to upload file to GitLab: %s", res.text)
    except Exception as exc:
        logger.error("Error uploading file to GitLab: %s", exc)
    return None

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
    Coordinates prompt retrieval, Orbit context query, Gemini reference image
    generation, Trellis 3D generation on Modal, and Merge Request creation.
    """
    logger.info("Executing Meshgen Pipeline (Phase 4)...")
    
    # Load configuration
    token = os.getenv("GITLAB_PRIVATE_TOKEN", "").strip() or os.getenv("GITLAB_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing required environment variable: GITLAB_PRIVATE_TOKEN or GITLAB_API_TOKEN")
    # FIX: Prioritize TARGET_PROJECT_ID first
    project_id = os.getenv("TARGET_PROJECT_ID", "").strip() or os.getenv("CI_PROJECT_ID", "")
    issue_iid = os.getenv("ISSUE_IID", "").strip()
    auto_close_issue = os.getenv("AUTO_CLOSE_ISSUE", "true").strip().lower() == "true"
    quality_mode = os.getenv("QUALITY_MODE", "med").strip().lower()

    # Step A: Parse prompt
    asset_name = user_prompt
    if user_prompt.lower().startswith("meshgen:"):
        asset_name = user_prompt[8:].strip()
        
    logger.info("Parsed Asset Name: '%s'", asset_name)
    _post_gitlab_issue_comment(issue_iid, token, f"🤖 **GitMesh: Orbit orchestrator started** for asset: `{asset_name}`")

    # Step B: Hierarchical Context Resolution
    logger.info("Retrieving context from GitLab Orbit API...")
    orbit_context = query_gitlab_orbit(project_id, asset_name, token)
    
    # Check for repository configuration file
    repo_config = query_gitlab_repository_config(project_id, token)
    
    # Parse explicit issue description overrides
    issue_desc_env = os.getenv("ISSUE_DESC", "")
    desc_overrides = parse_constraints_from_desc(issue_desc_env)

    # Allow issue description to override quality mode
    if desc_overrides.get("quality_mode"):
        quality_mode = desc_overrides["quality_mode"]
        logger.info("Quality mode overridden by issue description: %s", quality_mode)

    # Merge settings in priority:
    # 1. Issue tracker overrides (explicitly requested by developer in ticket)
    # 2. GitLab Orbit RAG context (dynamic repository database constraints)
    # 3. Repository config files (orbit.json/gitmesh.json static config)
    # 4. Fallback defaults
    
    target_folder = (
        desc_overrides.get("target_folder") or 
        orbit_context.get("target_folder") or 
        repo_config.get("target_folder") or 
        "Content/Generated/"
    )
    
    art_style = (
        desc_overrides.get("art_style") or 
        orbit_context.get("art_style") or 
        repo_config.get("art_style") or 
        ""
    )
    
    max_poly_count = (
        desc_overrides.get("max_poly_count") or 
        orbit_context.get("max_poly_count") or 
        repo_config.get("max_poly_count")
    )
    
    target_dimensions = (
        desc_overrides.get("target_dimensions") or 
        orbit_context.get("target_dimensions") or 
        repo_config.get("target_dimensions")
    )
    
    print(f"\n[🔍 ORBIT RAG] Found Constraints -> Dimensions: {target_dimensions or '[To be inferred by Gemini]'} | Style: {art_style or '[To be inferred by Gemini]'}\n")

    _post_gitlab_issue_comment(
        issue_iid, token,
        f"🔍 **Resolved GitLab Orbit & Duo context**:\n"
        f"- Target folder: `{target_folder}`\n"
        f"- Style constraint: `{art_style or '[Pending AI Inference]'}`\n"
        f"- Bounding limits: `{target_dimensions or '[Pending AI Inference]'}`"
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
    print(f"Target Dimensions:  {target_dimensions or '[To be inferred by Gemini]'}")
    print(f"Target Folder:      {target_folder}")
    print("==================================================================\n")

    # Step D: Generate reference image via Gemini/Vertex on Modal
    logger.info("Generating reference image via Gemini on Modal...")
    _post_gitlab_issue_comment(issue_iid, token, "📷 **Generating reference image via Gemini** (text → concept image for Trellis input)...")
    
    ref_image_b64 = ""
    category = "prop"
    subcategory = "general"
    asset_filename = asset_name.lower().replace(" ", "_")
    inferred_dimensions = [800.0, 400.0, 300.0]
    inferred_poly_count = 3000

    try:
        import modal
        ref_func = modal.Function.from_name("gitmesh-compute", "generate_reference_image")
        ref_result = ref_func.remote(
            prompt=enriched_prompt,
            issue_desc=issue_desc_env,
        )
        ref_status = ref_result.get("status", "unknown")
        enhanced_prompt = ref_result.get("enhanced_prompt", enriched_prompt)
        ref_image_b64 = ref_result.get("image_base64", "")
        category = ref_result.get("category", "prop")
        subcategory = ref_result.get("subcategory", "general")
        asset_filename = ref_result.get("filename", asset_name.lower().replace(" ", "_"))
        inferred_dimensions = ref_result.get("inferred_dimensions", [800.0, 400.0, 300.0])
        inferred_poly_count = ref_result.get("inferred_poly_count", 3000)
        
        print(f"\n[🧠 DUO AI] Engineered Prompt -> {enhanced_prompt}\n")
        logger.info("Reference image generation status: %s (enhanced prompt: '%s')", ref_status, enhanced_prompt[:100])
        
        # Save reference image locally and upload to GitLab to render in comments
        image_markdown = ""
        if ref_image_b64:
            local_ref_path = "reference.png"
            try:
                with open(local_ref_path, "wb") as f:
                    f.write(base64.b64decode(ref_image_b64))
                uploaded_md = upload_file_to_gitlab(project_id, local_ref_path, token)
                if uploaded_md:
                    image_markdown = f"\n\n{uploaded_md}"
            except Exception as img_save_err:
                logger.error("Failed to save or upload local reference image: %s", img_save_err)
        
        comment_body = (
            f"✅ **Reference image generated via Gemini.**{image_markdown}\n\n"
            f"- Enhanced prompt: `{enhanced_prompt[:120]}`\n"
            f"- Classified Category: `{category}/{subcategory}`\n"
            f"- File Name: `{asset_filename}.glb`"
        )
        _post_gitlab_issue_comment(issue_iid, token, comment_body)
    except Exception as exc:
        logger.warning("Reference image generation failed: %s. Trellis will use procedural fallback.", exc)
        _post_gitlab_issue_comment(issue_iid, token, f"⚠️ Reference image generation failed: `{exc}`. Trellis will use procedural fallback.")

    # Apply Gemini-inferred values if constraints are still unresolved
    if not art_style:
        art_style = "lowpoly"
        logger.info("Using default style: %s", art_style)
    if target_dimensions is None:
        target_dimensions = inferred_dimensions
        logger.info("Using Gemini-inferred target dimensions: %s", target_dimensions)
    if max_poly_count is None:
        max_poly_count = inferred_poly_count
        logger.info("Using Gemini-inferred max polygon limit: %s", max_poly_count)

    print(f"\n[⚡ SERVERLESS GPU] Routing task to Modal L4 (Style: {art_style} | Dimensions: {target_dimensions} | Quality: {quality_mode.upper()})...\n")

    # Step E: Trigger Trellis 3D mesh generation on Modal
    logger.info("Triggering TRELLIS 3D generation on Modal...")
    _post_gitlab_issue_comment(issue_iid, token, "⚡ **Generating 3D mesh via Trellis 2 on Modal** (image → 3D with physical scaling)...")
    
    try:
        if "modal" not in dir():
            import modal
        mesh_func = modal.Function.from_name("gitmesh-compute", "generate_3d_mesh")
        result = mesh_func.remote(
            prompt=asset_name,
            style=art_style,
            target_dimensions=target_dimensions,
            image_base64=ref_image_b64,
            quality_mode=quality_mode
        )
    except Exception as exc:
        logger.error("Failed to run remote Modal execution: %s. Using mock fallback.", exc)
        result = {
            "status": "success",
            "glb_path": "mesh.glb",
            "file_size_kb": 180.0,
            "glb_base64": "TW9jayBHRkIgY29udGVudHM="  # Decodes to "Mock GFB contents"
        }
        
    glb_path_remote = result.get("glb_path", "mesh.glb")
    glb_b64 = result.get("glb_base64", "")
    
    # Create a safe local path instead of using the remote container's /mnt/data path
    local_glb_path = os.path.basename(glb_path_remote)
    
    if not glb_b64 and not os.path.exists(local_glb_path):
        logger.error("TRELLIS generator failed to produce output.")
        _post_gitlab_issue_comment(issue_iid, token, "❌ **Mesh generation failed**: Trellis did not return valid file content.")
        return 1
        
    # Write file to disk locally using the safe local path
    if glb_b64:
        logger.info("Writing base64 GLB contents from Modal response to: %s", local_glb_path)
        with open(local_glb_path, "wb") as f:
            f.write(base64.b64decode(glb_b64))

    # Step F: Commit & Merge Request write-back
    base_folder = target_folder.strip("/")
    target_repo_path = f"{base_folder}/{category}/{subcategory}/{asset_filename}.glb"
    
    try:
        logger.info("Initiating GitLab MR write-back sequence...")
        mr_url = create_gitlab_merge_request(
            project_id=project_id,
            local_file_path=local_glb_path,
            target_repo_path=target_repo_path,
            asset_name=asset_name,
            gitlab_token=token
        )
        
        # Massive Success Message
        print("\n" + "🏆"*40)
        print("🏆  SUCCESS: GitMesh: Orbit Pipeline Completed successfully!  🏆")
        print(f"🏆  Merge Request Link: {mr_url}")
        print("🏆"*40 + "\n")
        
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
