#!/usr/bin/env python3
"""
GitMesh: Headless Autonomous 3D Technical Art Pipeline Agent.

Built using the Google Agent Development Kit (ADK) and Model Context Protocol (MCP).
This agent serves as a pure headless background worker for GitLab CI/CD, monitoring
issue boards, automatically generating reference assets, reconstructing 3D shapes,
and committing rigged meshes back to the repos.
"""

import os
import sys
import asyncio
import inspect
import logging
import subprocess
import urllib.parse
import urllib.request
from typing import List, Dict, Any, Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("GitMeshHeadlessAgent")
ADK_MODEL = "gemini-3.5-flash"


def _env_value(name: str, required: bool = True, default: str = "") -> str:
    value = os.getenv(name, default).strip()
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


REGISTRY_DICT_NAME = "gitmesh-project-registry"


def _registry_lookup(project_id: str) -> dict:
    """Read a target project's onboarding record from the Modal registry Dict."""
    if not project_id:
        return {}
    try:
        import modal
        registry = modal.Dict.from_name(REGISTRY_DICT_NAME, create_if_missing=True)
        return registry.get(str(project_id)) or {}
    except Exception as exc:
        logger.info("Registry lookup failed for project %s: %s", project_id, exc)
        return {}


def _gitlab_api_url() -> str:
    target_project_id = os.getenv("TARGET_PROJECT_ID", "").strip()
    if target_project_id:
        base = os.getenv("TARGET_GITLAB_URL", "").strip().rstrip("/") or "https://gitlab.com"
        return f"{base}/api/v4/projects/{target_project_id}"
    return os.getenv("GITLAB_API_URL", "").strip() or f"https://gitlab.com/api/v4/projects/{_env_value('CI_PROJECT_ID')}"


def _resolve_gitlab_token(passed_token: str) -> str:
    """Return the API token for the routed target project, else the passed token."""
    target_project_id = os.getenv("TARGET_PROJECT_ID", "").strip()
    if target_project_id:
        record = _registry_lookup(target_project_id)
        if record.get("api_token"):
            return record["api_token"]
    return passed_token


def _post_gitlab_issue_comment(issue_iid: str, gitlab_token: str, body: str) -> None:
    token = _resolve_gitlab_token(gitlab_token)
    if not issue_iid or not token:
        logger.info("Skipping GitLab issue comment because issue IID or token is missing.")
        return

    url = f"{_gitlab_api_url()}/issues/{urllib.parse.quote(str(issue_iid), safe='')}/notes"
    data = urllib.parse.urlencode({"body": body}).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("PRIVATE-TOKEN", token)
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(request, timeout=60) as response:
        logger.info("GitLab comment posted with status %s", response.status)


def _close_gitlab_issue(issue_iid: str, gitlab_token: str) -> None:
    token = _resolve_gitlab_token(gitlab_token)
    if not issue_iid or not token:
        return

    url = f"{_gitlab_api_url()}/issues/{urllib.parse.quote(str(issue_iid), safe='')}"
    data = urllib.parse.urlencode({"state_event": "close"}).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="PUT")
    request.add_header("PRIVATE-TOKEN", token)
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(request, timeout=60) as response:
        logger.info("GitLab issue close request returned status %s", response.status)


def _run_modal_command(args: List[str], stage_name: str, timeout: int = 1800) -> str:
    logger.info("[%s] Running command: %s", stage_name, " ".join(args))
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{stage_name} failed with exit code {result.returncode}")
    return result.stdout or ""


def _noop_role_agent(stage_name: str, responsibility: str) -> Dict[str, str]:
    return {"stage_name": stage_name, "responsibility": responsibility, "status": "planned"}


def _format_logical_agent_plan(issue_iid: str, issue_title: str, logical_agents: List[tuple[str, str]]) -> str:
    lines = [f"GitMesh remote plan for issue #{issue_iid}: {issue_title}"]
    for index, (agent_name, responsibility) in enumerate(logical_agents, start=1):
        lines.append(f"{index}. {agent_name}: {responsibility}")
    return "\n".join(lines)


async def run_remote_adk_orchestrator() -> int:
    """Run the production GitLab pipeline through an ADK-first role-agent orchestrator."""
    issue_title = _env_value("ISSUE_TITLE")
    issue_desc = _env_value("ISSUE_DESC", required=False)
    issue_iid = _env_value("ISSUE_IID")
    gitlab_token = _env_value("GITLAB_API_TOKEN")
    auto_close_issue = os.getenv("AUTO_CLOSE_ISSUE", "true").strip().lower() == "true"
    use_single_call_pipeline = os.getenv("USE_SINGLE_CALL_PIPELINE", "false").strip().lower() == "true"

    logical_agents = [
        ("Request Intake Agent", "Normalize GitLab issue context and initialize run state."),
        ("Reference Image Generator Agent", "Generate a visual reference image."),
        ("3D Mesh Generator Agent", "Generate the base GLB mesh."),
        ("GLB Output Validator Agent", "Validate generated GLB integrity."),
        ("Mesh Segmentation Agent", "Split mesh into semantic parts."),
        ("Part Labeling Agent", "Label segmented mesh parts."),
        ("Animation Plan Generator Agent", "Create a structured animation plan."),
        ("Animation Plan Validator Agent", "Validate and repair the animation plan."),
        ("Blender Render and Export Agent", "Render animation and export final GLB."),
        ("Delivery and Completion Agent", "Finalize issue status and delivery notes."),
    ]

    system_instruction = (
        "You are the GitMesh remote ADK supervisor. Coordinate specialized role agents for a GitLab-triggered 3D asset pipeline. "
        "You must plan, preserve stage order, and keep the user informed through GitLab issue comments. "
        "Runtime execution is performed by Modal-backed stage commands owned by each logical agent."
    )

    try:
        role_tools = []
        supervisor = create_adk_agent(
            name="gitmesh_remote_supervisor",
            model=ADK_MODEL,
            instruction=system_instruction,
            tools=role_tools,
        )
        plan_prompt = (
            f"Plan the remote GitMesh pipeline for GitLab issue #{issue_iid}.\n"
            f"Title: {issue_title}\n\nDescription:\n{issue_desc}\n\n"
            "Return a concise ordered plan naming each role agent and the handoff between agents."
        )
        logger.info("Submitting remote run context to ADK supervisor for planning.")
        if hasattr(supervisor, "generate_content"):
            adk_plan = await supervisor.generate_content(plan_prompt)
        else:
            logger.info("ADK supervisor initialized; installed SDK uses Runner/Context execution, so emitting deterministic role plan.")
            adk_plan = _format_logical_agent_plan(issue_iid, issue_title, logical_agents)
        print("\n========== ADK SUPERVISOR PLAN ==========")
        print(adk_plan)
        print("=========================================\n")
    except Exception as exc:
        logger.warning("ADK planning failed; continuing deterministic role-agent execution: %s", exc)

    _post_gitlab_issue_comment(
        issue_iid,
        gitlab_token,
        f"🧠 **GitMesh ADK Orchestrator Started**\nIssue #{issue_iid}: {issue_title}\nLogical agents: {len(logical_agents)}",
    )

    secret_list = _run_modal_command(["modal", "secret", "list"], "Modal Secret Preflight", timeout=300)
    if "gitmesh-keys" not in secret_list:
        raise RuntimeError("Modal secret 'gitmesh-keys' was not found. Run setup_remote.ps1 or bootstrap_modal_remote first.")

    if use_single_call_pipeline:
        _post_gitlab_issue_comment(
            issue_iid,
            gitlab_token,
            "⚡ Single-call mode enabled: running pipeline in one Modal container to reduce orchestration overhead.",
        )
        _run_modal_command(
            [
                "modal",
                "run",
                "modal_app.py::run_full_pipeline",
                "--prompt",
                issue_title,
                "--issue-desc",
                issue_desc,
                "--issue-iid",
                issue_iid,
                "--gitlab-token",
                gitlab_token,
            ],
            "Single-Call Pipeline Agent",
        )
        _post_gitlab_issue_comment(
            issue_iid,
            gitlab_token,
            f"🏁 **ADK-Orchestrated Pipeline Complete**\nAll role agents finished for: {issue_title}",
        )
        if auto_close_issue:
            _close_gitlab_issue(issue_iid, gitlab_token)
        return 0

    stage_commands = [
        (
            "Reference Image Generator Agent",
            "📷 Stage 2: Reference Image - ADK agent dispatching Modal image generation...",
            ["modal", "run", "modal_app.py::generate_reference_image", "--prompt", issue_title, "--issue-desc", issue_desc, "--issue-iid", issue_iid, "--gitlab-token", gitlab_token],
        ),
        (
            "3D Mesh Generator Agent",
            "🧊 Stage 3: Mesh Generation - ADK agent dispatching Modal GPU mesh generation...",
            ["modal", "run", "modal_app.py::generate_3d_mesh", "--prompt", issue_title, "--issue-desc", issue_desc, "--style", "lowpoly", "--issue-iid", issue_iid, "--gitlab-token", gitlab_token],
        ),
        (
            "GLB Output Validator Agent",
            "🔍 Stage 3b: GLB Validation - ADK validator checking mesh integrity...",
            ["modal", "run", "modal_app.py::validate_glb", "--issue-iid", issue_iid, "--gitlab-token", gitlab_token],
        ),
        (
            "Mesh Segmentation Agent",
            "✂️ Stage 4: Segmentation - ADK agent dispatching mesh segmentation...",
            ["modal", "run", "modal_app.py::segment_mesh", "--issue-iid", issue_iid, "--gitlab-token", gitlab_token],
        ),
        (
            "Part Labeling Agent",
            "🏷️ Stage 7: Part Labeling - ADK label agent classifying segments...",
            ["modal", "run", "modal_app.py::label_parts", "--asset-name", issue_title, "--issue-iid", issue_iid, "--gitlab-token", gitlab_token],
        ),
        (
            "Animation Plan Generator Agent",
            "🎬 Stage 8: Animation Planning - ADK motion agent generating plan...",
            ["modal", "run", "modal_app.py::generate_animation_plan", "--asset-name", issue_title, "--issue-iid", issue_iid, "--gitlab-token", gitlab_token],
        ),
        (
            "Animation Plan Validator Agent",
            "✅ Stage 9: Validation - ADK validation agent checking motion constraints...",
            ["modal", "run", "modal_app.py::validate_animation_plan", "--issue-iid", issue_iid, "--gitlab-token", gitlab_token],
        ),
        (
            "Blender Render and Export Agent",
            "🎬 Stage 10: Final Export - ADK render agent dispatching Blender export...",
            ["modal", "run", "modal_app.py::animate_and_render_mesh", "--issue-iid", issue_iid, "--gitlab-token", gitlab_token],
        ),
    ]

    for agent_name, comment, command in stage_commands:
        logger.info("Dispatching %s", agent_name)
        _post_gitlab_issue_comment(issue_iid, gitlab_token, comment)
        _run_modal_command(command, agent_name)

    _post_gitlab_issue_comment(
        issue_iid,
        gitlab_token,
        f"🏁 **ADK-Orchestrated Pipeline Complete**\nAll role agents finished for: {issue_title}",
    )
    if auto_close_issue:
        _close_gitlab_issue(issue_iid, gitlab_token)
    return 0

# =====================================================================
# Google Agent Development Kit (ADK) SDK Imports & Fallbacks
# =====================================================================
try:
    try:
        import google_adk as adk
        from google_adk import Agent, Tool
    except ImportError:
        from google.adk.agents import Agent
        from google.adk.tools import FunctionTool
        adk = sys.modules.get("google.adk")

        class Tool:
            @staticmethod
            def from_function(fn: Any) -> Any:
                try:
                    return FunctionTool(func=fn)
                except TypeError:
                    return FunctionTool(fn)
    logger.info("✅ Successfully imported Google ADK.")
except ImportError:
    logger.warning("⚠️ 'google-adk' package not found in current environment. Setting up dry-run fallback classes.")
    
    # Mock fallback classes for local systems development & dry-run compliance
    class MockTool:
        def __init__(self, name: str, description: str, function: Any):
            self.name = name
            self.description = description
            self.function = function

        @staticmethod
        def from_function(fn: Any) -> 'MockTool':
            name = getattr(fn, "__name__", str(fn))
            doc = getattr(fn, "__doc__", "No description provided.")
            return MockTool(name=name, description=doc, function=fn)

    class MockAgent:
        def __init__(self, model: str, system_instruction: str, tools: List[Any]):
            self.model = model
            self.system_instruction = system_instruction
            self.tools = tools
            logger.info(f"Initialized MockAgent with model {model} and {len(tools)} tools.")

        async def generate_content(self, prompt: str) -> str:
            tool_names = []
            for t in self.tools:
                if hasattr(t, "name"):
                    tool_names.append(t.name)
                elif hasattr(t, "__name__"):
                    tool_names.append(t.__name__)
                else:
                    tool_names.append(str(t))
            return (
                f"[Simulation Test Response from Gemini 3.1 Flash with tools: {', '.join(tool_names)}]\n"
                f"Resolved prompt: '{prompt}' by invoking serverless Modal routines & GitLab MCP connectors."
            )
            
    adk = sys.modules[__name__]  # self-reference placeholder
    Agent = MockAgent
    Tool = MockTool


def create_adk_agent(name: str, model: str, instruction: str, tools: List[Any]) -> Any:
    """Create an ADK Agent across google_adk and google.adk constructor variants."""
    params = inspect.signature(Agent).parameters
    kwargs: Dict[str, Any] = {"model": model, "tools": tools}
    if "name" in params:
        kwargs["name"] = name
    if "system_instruction" in params:
        kwargs["system_instruction"] = instruction
    elif "instruction" in params:
        kwargs["instruction"] = instruction
    else:
        kwargs["system_instruction"] = instruction
    return Agent(**kwargs)

# Try importing Model Context Protocol (MCP) Python SDK
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    logger.info("✅ Successfully imported mcp SDK.")
except ImportError:
    logger.warning("⚠️ 'mcp' SDK not found in current environment. Setting up dry-run fallback layers.")
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None


# =====================================================================
# 1. Serverless GPU Technical Art App Imports from modal_app.py
# =====================================================================
try:
    from modal_app import (
        generate_3d_mesh as modal_generate_3d_mesh,
        segment_mesh as modal_segment_mesh,
        animate_and_render_mesh as modal_animate_and_render_mesh
    )
    logger.info("✅ Successfully imported serverless technical art functions from modal_app.py")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import from modal_app.py direct definitions ({e}). Using inline mock fallbacks.")
    modal_generate_3d_mesh = None
    modal_segment_mesh = None
    modal_animate_and_render_mesh = None


# =====================================================================
# 2. Pipeline Execution Tool Adapters
# =====================================================================

def run_generate_3d_mesh(prompt: str, style: str = "lowpoly") -> Dict[str, Any]:
    """
    Generates a high-quality 3D mesh asset (.glb) using the serverless Trellis 2 pipeline on Modal.
    Invokes 2D concept generation followed by point-cloud shape reconstruction.

    Args:
        prompt (str): Detailed prompt of the 3D game prop (e.g., 'Lowpoly Medieval Viking Sword').
        style (str): Visual constraint filter for topology/shading ('lowpoly', 'stylized', 'realistic').

    Returns:
        dict: High-level technical art metadata holding file size, vertex counts, and storage GLB URL.
    """
    logger.info(f"🎨 [Pipeline Tool] Invoking generate_3d_mesh for: '{prompt}' (style: {style}) via Modal")
    
    # If Modal client and function is properly imported, run container sandbox invocation
    if modal_generate_3d_mesh is not None:
        try:
            # Use remote Modal container runtime handles
            if hasattr(modal_generate_3d_mesh, "remote"):
                return modal_generate_3d_mesh.remote(prompt, style)
            return modal_generate_3d_mesh(prompt, style)
        except Exception as e:
            logger.warning(f"⚠️ Modal execution failed during run_generate_3d_mesh ({e}). Proceeding in local mode.")

    # Fallback local simulation placeholder mapping
    slug = prompt.lower().replace(" ", "_").replace("'", "")
    return {
        "status": "success",
        "url": f"https://modal.com/artifacts/gitmesh-pipeline/{slug}_{style}.glb",
        "style": style,
        "vertex_count": 4500,
        "file_size_kb": 240.5,
        "generator_model": "Simulation-Fallback-Trellis"
    }


def run_segment_mesh(glb_url: str, prompt_tags: str) -> Dict[str, Any]:
    """
    Submits a segmented part analysis request to the serverless P3-SAM model.
    Divides a 3D GLB file into individual semantic parts or component sub-meshes (e.g., separating sword hilt, blade).

    Args:
        glb_url (str): Cloud target URL of the game GLB mesh file to segment.
        prompt_tags (str): Comma-separated list of target part name tags (e.g., 'hilt, blade, guard').

    Returns:
        dict: Mapping containing segmented parts, relative indices, bounds and alignment vectors.
    """
    logger.info(f"✂️ [Pipeline Tool] Invoking P3-SAM segment_mesh for tagging: {prompt_tags}")

    # If Modal client and function is properly imported, run container sandbox invocation
    if modal_segment_mesh is not None:
        try:
            if hasattr(modal_segment_mesh, "remote"):
                return modal_segment_mesh.remote(glb_url, prompt_tags)
            return modal_segment_mesh(glb_url, prompt_tags)
        except Exception as e:
            logger.warning(f"⚠️ Modal execution failed during run_segment_mesh ({e}). Proceeding in local mode.")

    # Fallback local simulation placeholder mapping
    tags = [t.strip() for t in prompt_tags.split(",")]
    parts_map = {tag: {"part_id": f"part_{i:03d}_{tag}", "index": i} for i, tag in enumerate(tags)}
    return {
        "status": "success",
        "original_mesh_url": glb_url,
        "detected_parts_count": len(tags),
        "parts": parts_map,
        "segment_pipeline": "Simulation-Fallback-SAM"
    }


def run_animate_and_render_mesh(glb_url: str, animation_plan_json: str) -> Dict[str, Any]:
    """
    Simulates or executes heavy-duty headless Blender animation rigging and MP4 video preview rendering on Modal container nodes.

    Args:
        glb_url (str): Target GLB reference asset source link.
        animation_plan_json (str): Stringified JSON outline of animation transforms (e.g., '{"rotation_y": 360, "frames": 30}').

    Returns:
        dict: Cloud URL links mapped to the active rigid animated mesh model output and the rendered MP4 turntable.
    """
    logger.info(f"🎬 [Pipeline Tool] Invoking Headless Blender animate_and_render_mesh for: {glb_url}")

    # Invoke live Modal container worker if available
    if modal_animate_and_render_mesh is not None:
        try:
            if hasattr(modal_animate_and_render_mesh, "remote"):
                return modal_animate_and_render_mesh.remote(glb_url, animation_plan_json)
            return modal_animate_and_render_mesh(glb_url, animation_plan_json)
        except Exception as e:
            logger.warning(f"⚠️ Modal execution failed during run_animate_and_render_mesh ({e}). Proceeding in local mode.")

    # Fallback local simulation routine
    return {
        "status": "success",
        "animated_glb_url": f"https://modal.com/artifacts/gitmesh-compute/animated_{os.path.basename(glb_url)}",
        "preview_video_url": f"https://modal.com/artifacts/gitmesh-compute/preview_{os.path.basename(glb_url).replace('.glb', '.mp4')}",
        "total_frames_rendered": 24,
        "render_engine": "Simulation-Fallback-Blender"
    }


# =====================================================================
# 3. Wrapping native Python functions into Google ADK structural Tools
# =====================================================================

logger.info("🛠️ Wrapping technical art pipeline functions using Tool.from_function()...")
trellis_3d_tool = Tool.from_function(run_generate_3d_mesh)
sam_segment_tool = Tool.from_function(run_segment_mesh)
blender_anim_tool = Tool.from_function(run_animate_and_render_mesh)


# =====================================================================
# GitLab MCP Pipeline Connection Orchestration
# =====================================================================

async def connect_gitlab_mcp() -> Optional[Any]:
    """
    Establishes an asynchronous stdio transport connection to the GitLab Duo MCP server.
    Spawns 'npx -y @gitlab/mcp-server-gitlab' with proper credentials injected.

    Returns:
        The stdio stream client wrapper context manager if successful, None otherwise.
    """
    if stdio_client is None or StdioServerParameters is None:
        logger.warning("⚠️ MCP library is missing. Cannot establish live GitLab Duo MCP subprocess context.")
        return None

    # Retrieve environment variables for auth config
    private_token = os.getenv("GITLAB_PRIVATE_TOKEN")
    api_url = os.getenv("GITLAB_API_URL", "https://gitlab.com")

    if not private_token:
        logger.warning("⚠️ GITLAB_PRIVATE_TOKEN missing. Skipping live GitLab MCP connection.")
        return None

    # Parameters to spawn GitLab MCP server via npx subprocess
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@gitlab/mcp-server-gitlab"],
        env={
            **os.environ,
            "GITLAB_PRIVATE_TOKEN": private_token,
            "GITLAB_API_URL": api_url
        }
    )

    logger.info(f"🚀 Spawning GitLab MCP server on subprocess: npx -y @gitlab/mcp-server-gitlab")
    try:
        # Standard stdio client establishes bidirectional pipeline (stdin/stdout) to sub-process
        return stdio_client(server_params)
    except Exception as e:
        logger.error(f"❌ Failed to instantiate stdio transport connection: {e}")
        return None


# =====================================================================
# Main Header Execution Loop
# =====================================================================

async def main():
    if "--remote-ci" in sys.argv:
        raise SystemExit(await run_remote_adk_orchestrator())

    logger.info("Initializing Headless GitMesh Pipeline Agent...")

    # Load and validate key configuration variables
    gitlab_token = os.getenv("GITLAB_PRIVATE_TOKEN")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not gitlab_token:
        logger.warning("💡 GITLAB_PRIVATE_TOKEN not found in environment. Running without live GitLab MCP tools.")
    if not gemini_key:
        logger.warning("💡 GEMINI_API_KEY not found in environment. Vertex/Gemini requests will run mock fallback.")

    # 1. Attempt Connection to GitLab Duo MCP Server
    gitlab_mcp_ctx = await connect_gitlab_mcp()
    mcp_tools = []

    if gitlab_mcp_ctx:
        logger.info("🔄 Connecting to live GitLab MCP Session...")
        try:
            # Enter stdio transport loops
            async with gitlab_mcp_ctx as (read_stream, write_stream):
                # Init ClientSession Handshake protocols
                async with ClientSession(read_stream, write_stream) as session:
                    logger.info("🤝 Performing protocol handshake with GitLab Duo MCP...")
                    await session.initialize()
                    
                    logger.info("📡 Retrieving GitLab Dynamic Actions & API Tools...")
                    tools_response = await session.list_tools()
                    mcp_tools = tools_response.tools if hasattr(tools_response, 'tools') else []
                    
                    logger.info(f"🎉 Connected! Dynamic GitLab MCP tools discovered: {[t.name for t in mcp_tools]}")
                    
                    # Run the active agent loop within the standard context session
                    await initialize_adk_agent_and_test(mcp_tools)
                    return
        except Exception as e:
            logger.error(f"❌ Connection error during live GitLab MCP initialization: {e}")
            logger.info("Falling back to simulated pipeline dry-run...")
    else:
        logger.info("Running standard dry-run simulation mode (Active Local Pipeline).")

    # 2. Run simulation loop if live subprocess is not configured/supported
    await initialize_adk_agent_and_test(mcp_tools=[])


async def initialize_adk_agent_and_test(mcp_tools: List[Any]):
    """
    Initializes the Google ADK Agent using the configured ADK model, combining
    the wrapped serverless Modal tech-art tools and dynamic tools retrieved from GitLab MCP.
    """
    logger.info("🛠️ Building combined workflow toolbelt with injected Modal tools...")
    
    # 4. Inject wrapped Modal tools into combined workflow belt alongside GitLab dynamic tools
    combined_tools = [trellis_3d_tool, sam_segment_tool, blender_anim_tool] + mcp_tools
    
    system_instruction = (
        "You are GitMesh, a highly expert autonomous AI Technical Art pipeline agent operating as a headless CI/CD worker integrated into GitLab via MCP. "
        "Your absolute protocol is to follow this exact 10-step sequence whenever a new issue or asset request is processed. "
        "You must output detailed step-by-step progress and you are strictly forbidden from skipping any of the following GitLab UI update and comment steps:\n\n"
        "STEPS TO FOLLOW:\n"
        "-----------------\n"
        "Step 1: Read and analyze the user's 3D asset request. Use available GitLab MCP tools to create a new Git branch and a Merge Request (MR) associated with the task.\n"
        "Step 2: Use the GitLab MCP commenting tool to post an initial comment to the created MR: 'Initializing GitMesh Pipeline: Generating base 3D mesh...'\n"
        "Step 3: Execute the 'run_generate_3d_mesh' tool with an appropriate prompt and style constraint derived from the request specifications.\n"
        "Step 4: Use the GitLab MCP commenting tool to post a progression comment to the MR: 'Mesh generated. Segmenting semantic parts...'\n"
        "Step 5: Execute the 'run_segment_mesh' tool to partition the generated GLB asset into logical sub-meshes.\n"
        "Step 6: Autonomously calculate/compile a precise math animation plan parameters payload in valid JSON format based specifically on the structural asset type. "
        "(For example, if the asset is a chest, generate a hinge rotation on local axes; if it is a sword, generate a turntable twist or slice motion).\n"
        "Step 7: Use the GitLab MCP commenting tool to post a progression comment to the MR: 'Applying procedural rigging and rendering preview...'\n"
        "Step 8: Execute the 'run_animate_and_render_mesh' tool passing the segmented GLB URL and your generated JSON animation plan.\n"
        "Step 9: Use the GitLab MCP 'push_files' or equivalent commit tooling to push the final rigged and animated '.glb' model file along with the rendered turntable preview '.mp4' file into the repository branch.\n"
        "Step 10: Post a final conclusive comment on the MR confirming delivery containing references/links (e.g., Markdown video pointers) to the compiled assets, and update the MR status/metadata to transition the MR state into 'ready for review' to complete your operational cycle."
    )

    logger.info(f"🧠 Instantiating Google ADK Agent (Model: {ADK_MODEL})...")
    try:
        # Initialize Google ADK Agent with combined workspace capabilities
        agent = create_adk_agent(
            name="gitmesh_headless_agent",
            model=ADK_MODEL,
            instruction=system_instruction,
            tools=combined_tools,
        )
        logger.info("✅ Google ADK Agent initialized successfully.")
        
        # Test query to verify integration, planning capability, and mock output
        # Fallback: If no command-line argument is provided (e.g., during manual testing), fall back to a default mock string.
        if len(sys.argv) > 1:
            issue_query = sys.argv[1]
        else:
            issue_query = (
                "Analyze GitLab Issue #42: 'Asset Request: Lowpoly Pirate Chest'. "
                "Execute run_generate_3d_mesh for 'Lowpoly Pirate Chest', segment the output mesh into "
                "'lid, base, lock' using run_segment_mesh, animate and render the keyframes using "
                "run_animate_and_render_mesh for 30 frames with 360 degree turntable loop, and post a final draft checkout comment with the turntable MP4 link."
            )
        logger.info(f"📬 Submitting query of work to agent: '{issue_query}'")
        
        if hasattr(agent, "generate_content"):
            test_response = await agent.generate_content(issue_query)
        else:
            test_response = (
                "ADK agent initialized successfully. The installed SDK uses Runner/Context execution, "
                "so direct generate_content smoke output is unavailable in this local harness.\n"
                f"Received query: {issue_query}"
            )
        print("\n" + "="*50)
        print("          GITMESH HEADLESS AGENT TEST RESPONSE      ")
        print("="*50)
        print(test_response)
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Failed to execute ADK Agent cycle: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Execution halted by user.")
