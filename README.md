# GitMesh 🚀 (Headless Autonomous 3D Technical Art Pipeline Agent)

GitMesh is a fully headless, autonomous **CO-OP CI/CD agent** engineered to break the single greatest bottleneck in game development: manual 3D asset generation, mesh cleanup, semantic splitting, skeletal rigging, and visual turntable proofing. 

By utilizing a ChatOps-driven workflow, **GitLab's Issue Board and Merge Requests (MRs)** serve as the entire user interface. Technical artists, game designers, and developers request assets inside standard tickets; GitMesh automatically intercepts, triggers massive high-performance remote GPU pipelines, and commits rigged game-ready assets alongside proof-of-work streaming renders back to the repository branch—completely unsupervised.

---

## 🚀 How It Works (The 10-Step Pipeline)

When a developer labels a GitLab issue `asset:generate`, the **Google Agent Development Kit (ADK)**-powered brain takes total control, executing a flawless, sequential 10-step development pipeline:

```mermaid
graph TD
    subgraph GitLab [1. UI / ChatOps Interface]
        A["GitLab Issue Board (Label: asset:generate)"] -->|1. Intercept Issue| C
        B["GitLab Merge Request Workspace"] -.-|9. Commits Rigged GLB & MP4| H
        B -.-|10. Posts Delivery Link| I["Final Review / Merge ✅"]
    end

    subgraph Brain [2. Orchestration Brain - Google Agent SDK]
        C["Google ADK Agent (Gemini 3.1 Flash)"] <-->|Bidirectional Action Discovery| D["Model Context Protocol (MCP) Server"]
        C -->|2, 4, 7. Progression Status Comments| B
    end

    subgraph Compute [3. Technical Art Compute Layer - Modal serverless GPU]
        C -->|3. Generate Base Mesh| E["Trellis 2 (A10G GPU)<br/>Sparse Reconstructor"]
        C -->|5. Partition Submeshes| F["P3-SAM 3D (A10G GPU)<br/>Segment Anything Model"]
        C -->|8. Animate & Render| G["Headless Blender (Core Container)<br/>bpy Procedural Rigging"]
        E -->|Initial .glb| F
        F -->|Spatially Isolated Segments| G
        G -->|Turntable Engine Output| H["Rigged GLB + Turntable MP4 Preview"]
    end

    style C fill:#4285F4,stroke:#333,stroke-width:2px,color:#fff
    style D fill:#34A853,stroke:#333,stroke-width:1px,color:#fff
    style E fill:#EA4335,stroke:#333,stroke-width:1px,color:#fff
    style F fill:#FBBC05,stroke:#333,stroke-width:1px,color:#fff
    style G fill:#FF9F1C,stroke:#333,stroke-width:1px,color:#fff
```

### The Breakdown:
1. **Issue Analysis**: The agent intercepts a GitLab issue containing physical prop parameters (e.g., *"Lowpoly Pirate Chest, oak wood"*), automatically instantiates a new local Git branch, and opens a Merge Request.
2. **Handshake Post**: The agent comments on the MR: `"Initializing GitMesh Pipeline: Generating base 3D mesh..."` 
3. **Sparse Reconstruction (Trellis 2)**: Triggers serverless GPU routines on Modal to reconstruct a dense 3D point cloud and generate an initial clean `.glb` mesh envelope.
4. **Interim Post**: Comments on the MR: `"Mesh completed. Segmenting semantic parts..."`
5. **Semantic Part Segmentation (P3-SAM)**: Sends the mesh raw vertex buffers to **P3-SAM** (Segment Anything Model 3D) to automatically group discrete components (e.g., isolating a chest's *lid* from its *base*).
6. **Kinetic Intelligence Core**: The agent's LLM brain parses the segmented components with spatial coordinates and crafts custom mechanical movement plans in standard JSON.
7. **Animation Handshake**: Comments on MR: `"Applying procedural rigging and rendering preview..."`
8. **Headless Blender Rigging (`bpy` + Workbench)**: Deploys a headless Blender container on Modal to write bone weights and keyframe the turntable loops.
9. **Turntable Rendering**: Renders a standard turntable movie (`.mp4`) at 60 FPS using Blender's rapid Workbench engine.
10. **Delivery & PR Checkout**: Pushes the finished `.glb` and preview `.mp4` into the Git repository branch, uploads structural markdown video embeds directly inside the MR comments, and updates MR status metadata to `ready for review`.

---

## 🏛️ System Architecture

GitMesh bridges cloud-native enterprise developer interfaces (GitLab) with state-of-the-art serverless GPU clusters using structured AI coordination:

- **Google Agent Development Kit & Gemini API Key Server**: Orchestrates the multi-layered decisions, tool selection, and code injection sequences.
- **Model Context Protocol (MCP)**: Standardizes tool declarations, executing low-level shell commands to spin up the `@gitlab/mcp-server-gitlab` bridge and communicate via bidirectional stdio streams.
- **Serverless Task Container Blocks (Modal)**: Offloads heavy memory/processing pipelines into cold-start optimized micro-containers running on state-of-the-art neural cores on-demand.

---

## 🛠️ The Tech Stack

- **google-adk (Google Agent Development Kit)**: High-level Python developer SDK mapping system instructions into safe tool belts, managing recursive function calls, and carrying out multi-step code and design tasks with `gemini-3.1-flash`.
- **Model Context Protocol (MCP)**: Universal context gateway standard allowing the Gemini developer brain to discover, call, and coordinate standard Git APIs securely over the `@gitlab/mcp-server-gitlab` dynamic tool schema.
- **Modal Serverless Platforms**:
  - **Trellis 2 (3D Generation)**: Runs serverless inference over state-of-the-art transformer 3D geometry builders on fast A10G GPUs.
  - **P3-SAM**: Runs high-accuracy part-to-semantic segmentation models.
  - **Blender headless**: Standard Debian environments running custom `bpy` tasks to rig objects programmatically.
- **FastAPI / Python 3.11**: Event-driven client core orchestrating background tasks, handling continuous streams of webhook events, and driving state machine processes.

---

## 🚀 Sandbox Simulation Runs

To spin up and simulate the local pipeline dry-run, tool-belt synthesis, and mock agent cycle:

```bash
# Export standard API tokens
export GEMINI_API_KEY="your-gemini-key"
export GITLAB_PRIVATE_TOKEN="your-gitlab-token"

# Run the central orchestrator
python3 agent.py
```

---

## 🔮 V2 Roadmap (Hackathon Future Pitch)

During game production, 3D props are completely empty without **organic sound design** to accompany their visual kinetic animation cues (e.g. wood creaking during a trunk-lid opening, steel echoing during sword swings).

- **AudioLDM 2 Integration on Modal**: Add an extra orchestration tool `generate_audio_fx_for_part`.
- **Procedural Sound Trigger Maps**: GitMesh's LLM brain will analyze the animation plan and auto-generate precise SFX (mp3 files) synchronized exactly with Blender keyframe limits.
- **Dynamic GLTF Audio Ext**: Output unified spatial objects directly packaged with audio triggers, delivering a fully interactive visual-audio pipeline directly out of CI/CD.
