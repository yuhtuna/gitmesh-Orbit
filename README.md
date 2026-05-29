# GitMesh (Headless Python AI Technical Art Agent)

GitMesh is a headless CI/CD Art pipeline worker that utilizes GitLab issues and merge requests as its user interface. It connects to GitLab via the **Model Context Protocol (MCP)** and integrates with **Google Agent Development Kit (ADK)** to dynamically generate 3D mesh assets, perform mesh-checking, auto-rigging checks, and commit/comment back to the repository.

This project is fully headless. No React, Vite, or web frontends are included. It provides a visual Express-based process execution dashboard purely for the sandbox environment stream console (running on standard accessible port 3000).

---

## 🛠️ Tech Stack & Key Layers

1. **Google Agent Development Kit (ADK)**: High-level Python developer framework to configure custom system instructions, tool belts, and execute intent-cycles via modern `gemini-3.1-flash` models.
2. **Model Context Protocol (MCP)**: Implements standard bidirectional stdio connections to discover and consume external workspace tools of the `@gitlab/mcp-server-gitlab` server.
3. **Modal**: Serves as the serverless technical art task execution back-end (powering concept creation, 3D shape reconstruction, and rig baking).

---

## 📦 Requirements Checklist (`requirements.txt`)

- `google-adk`: Vertex/Gemini Agent orchestration API.
- `mcp`: Standard MCP bidirectional context session standard.
- `modal`: Serverless task container endpoints for 3D topology generation.
- `python-dotenv`: Sandbox environmental variable handling.

---

## 🚀 Execution & Sandbox Console Tests

To execute a full simulation pipeline sequence of the GitLab Handshake, discovery of tools, tool-belt building, and ADK model generation:

```bash
# Set up necessary local variables (e.g. within interactive shell or environment config)
export GEMINI_API_KEY="your-gemini-key"
export GITLAB_PRIVATE_TOKEN="your-gitlab-token"

# Run the Python Headless script directly
python3 agent.py
```

### Typical Live Output Log Sequence:

```text
2026-05-29 14:07:16 [INFO] GitMeshHeadlessAgent - Initializing Headless GitMesh Pipeline Agent...
2026-05-29 14:07:16 [INFO] GitMeshHeadlessAgent - 🚀 Spawning GitLab MCP server on subprocess: npx -y @gitlab/mcp-server-gitlab
2026-05-29 14:07:17 [INFO] GitMeshHeadlessAgent - 🤝 Performing protocol handshake with GitLab Duo MCP...
2026-05-29 14:07:17 [INFO] GitMeshHeadlessAgent - 📡 Retrieving GitLab Dynamic Actions & API Tools...
2026-05-29 14:07:18 [INFO] GitMeshHeadlessAgent - Discovering Available tools... Live GitLab MCP tools resolved!
2026-05-29 14:07:18 [INFO] GitMeshHeadlessAgent - 🛠️ Building combined workflow toolbelt...
2026-05-29 14:07:18 [INFO] GitMeshHeadlessAgent - 🧠 Instantiating Google ADK Agent (Model: gemini-3.1-flash)...
2026-05-29 14:07:18 [INFO] GitMeshHeadlessAgent - 📬 Submitting dry-run query to agent: 'Analyze GitLab Issue #42:...'
2026-05-29 14:07:18 [INFO] GitMeshHeadlessAgent - 🎨 [Pipeline Tool] Invoking generate_3d_mesh for prompt: 'Lowpoly Pirate Chest'
```
