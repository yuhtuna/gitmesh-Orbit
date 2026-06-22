# GitMesh: Orbit 🪐
> **Your context-aware AI Technical Artist, powered by GitLab Orbit.**

GitMesh: Orbit is a 3D asset generation pipeline built as a GitLab Duo Custom Skill. It connects your repository's spatial constraints and styling guidelines directly to a serverless GPU-powered 3D engine, closing the DevOps loop by automatically committing generated and scaled 3D assets via Merge Requests.

---

## 🛑 The Problem
Standard 3D generative AI pipelines suffer from a **context blind spot**. When prompted to generate game assets (e.g., *"Create a medieval chest"*), they run blindly without understanding:
1. **Target Subdirectories:** Where should the asset go inside your project hierarchy?
2. **Art Style Constraints:** Should the model be low-poly, voxel, stylized, or realistic?
3. **Physics/Scale Limits:** What coordinate limits (extents) does your target game engine (Unity, Unreal, Godot) expect for this prop?

Without this metadata, developers must manually download, scale, convert, and organize generated models, breaking the automation workflow.

---

## ⚡ The Solution: How It Works

GitMesh: Orbit bridges this context gap by dividing roles into a **Brain** (GitLab Duo and Orbit knowledge graph) and the **Muscle** (our deterministic serverless engine):

```mermaid
graph TD
    A[GitLab Duo Webhook / Chat] -->|1. Intent parsed| B(agent.py Orchestrator)
    B -->|2. Query context| C{GitLab Orbit RAG API}
    C -->|3. Subfolder, style & scale constraints| B
    B -->|4. Trigger remote TRELLIS 3D| D[modal_app.py Serverless GPU]
    D -->|5. Image-to-3D GLB| D
    D -->|6. trimesh scaling| E[Scaled Asset GLB]
    E -->|7. Return Base64| B
    B -->|8. Create MR| F[GitLab Repository & Merge Request]
```

### 1. Trigger
GitLab Duo monitors issue boards and chat triggers for the keyword `Meshgen:`. It parses the high-level request intent (e.g., `"Meshgen: Viking Sword"`). You can also use the `/meshgen` slash command in GitLab Duo Chat.

### 2. Context (The Brain)
The orchestrator (`agent.py`) queries the **GitLab Orbit API** (`/orbit/nodes`) using the asset name. Orbit extracts:
- `target_folder`: The directory where the asset belongs (e.g., `Assets/Props/Weapons/`).
- `art_style`: Visual constraint guidelines (e.g., `lowpoly`).
- `target_dimensions`: Physical X/Y/Z coordinate bounds required by the project.

### 3. Generation & Scaling (The Muscle)
The orchestrator triggers the serverless **Trellis 2 pipeline on Modal** (using high-performance NVIDIA L4 GPUs):
- **Gemini** generates a high-fidelity reference concept image from the enriched prompt.
- **Trellis 2** reconstructs 3D meshes from the reference image.
- **`trimesh` post-processing** calculates the generated mesh bounding box, computes uniform scaling factors to fit inside your `target_dimensions`, and rescales the geometry.

### 4. Write-Back (DevOps Loop)
The orchestrator receives the base64-encoded scaled `.glb` file, pushes it to a new branch at the correct folder path (e.g., `Content/Generated/weapon/pistol/laser_pistol.glb`), and automatically creates a Merge Request targeting `main`.

---

## 🎚️ Quality Modes

GitMesh supports three quality tiers that control the fidelity of the generated 3D mesh. Set `QUALITY_MODE` as a CI/CD variable or include `Quality: high` in the issue description.

| Mode | Diffusion Steps | Mesh Simplification | Texture Size | GPU Time | Best For |
|------|----------------|---------------------|-------------|----------|----------|
| **low** | 8 | 98% decimation | 512px | ~45s | Quick previews, rapid iteration |
| **med** *(default)* | 12 | 95% decimation | 1024px | ~90s | Standard production assets |
| **high** | 20 | 50% decimation | 2048px | ~150s | Hero assets, demo showcases |

### Setting Quality via Issue Description
Add a `Quality:` flag in your issue body to override the pipeline default:
```
MeshGen: A medieval treasure chest

Quality: high
Style: lowpoly
Folder: Content/Props/Furniture
Dimensions: [800, 500, 500]
```

---

## 🤖 GitLab Duo Chat Integration

GitMesh registers as a **GitLab Duo Custom Chat Skill** with the `/meshgen` slash command.

```
/meshgen a retro sci-fi laser rifle
```

This triggers the full pipeline directly from GitLab Duo Chat or your IDE extension — no issue creation needed.

---

## 🚀 Setup & Installation

Follow these instructions to set up GitMesh: Orbit for your own GitLab projects.

### 1. Prerequisites
- **Python 3.10+** installed locally or in your CI/CD environment.
- **GitLab Project:** A target repository where you want 3D assets to be generated.
- **Modal Account:** Sign up at [Modal.com](https://modal.com) for serverless GPU access.
- **Google Cloud:** A GCP project with Vertex AI API enabled (for Gemini image generation).

### 2. Clone and Install
Clone this repository and install the required dependencies:
```bash
git clone https://github.com/yuhtuna/gitmesh-Orbit.git
cd gitmesh-Orbit
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory (or set these as CI/CD variables in GitLab) and populate the following:

```env
# GitLab Configuration
GITLAB_PRIVATE_TOKEN=glpat-your_personal_access_token_here
GITLAB_PROJECT_ID=your_project_id
GITLAB_TRIGGER_TOKEN=your_trigger_token
GITLAB_WEBHOOK_SECRET=your_webhook_secret

# Modal Cloud GPUs (Auto-generated if you run `modal setup`)
MODAL_TOKEN_ID=your_modal_token_id
MODAL_TOKEN_SECRET=your_modal_token_secret

# Google Cloud (for Gemini reference image generation)
GCP_PROJECT_ID=your_gcp_project_id
GCP_SERVICE_ACCOUNT_JSON='{ ... your service account JSON ... }'
```

### 4. Deploy the Compute Engine (Modal)
Deploy the serverless compute functions and webhook listener to Modal. The `setup_remote.ps1` script handles everything:
```powershell
# Full automated setup (deploys Modal apps, creates GitLab webhook, syncs CI/CD variables)
.\setup_remote.ps1

# Or deploy individually:
modal deploy modal_app.py      # GPU compute functions
modal deploy gitlab_webhook.py # Webhook listener
```

### 5. Running the Pipeline
**Option A: Via GitLab Issue (recommended)**
Create an issue titled `MeshGen: <your asset prompt>` in your GitLab project. The webhook triggers the pipeline automatically.

**Option B: Via GitLab Duo Chat**
Type `/meshgen <your asset prompt>` in GitLab Duo Chat.

**Option C: Local testing**
```bash
python agent.py "Meshgen: Viking Broadsword"
```

**What happens next?**
1. The agent queries your repository's Orbit API for context.
2. Gemini generates an optimized reference concept image.
3. The serverless GPU generates and scales the 3D mesh.
4. The agent commits the new asset to the correct folder and returns a **GitLab Merge Request URL**.
5. The originating issue is automatically closed.

---

## 📂 Project Structure

| File | Purpose |
|------|---------|
| `agent.py` | Orchestrator — coordinates the full pipeline from prompt to MR |
| `modal_app.py` | Serverless GPU compute — Gemini image gen + Trellis 3D reconstruction |
| `gitlab_webhook.py` | FastAPI webhook on Modal — receives issue/Duo events, triggers CI pipeline |
| `metadata.json` | GitLab Duo Custom Skill manifest (defines `/meshgen` command) |
| `SKILL.md` | System prompt layer for GitLab Duo LLM integration |
| `.gitlab-ci.yml` | CI/CD pipeline definition with quality presets |
| `setup_remote.ps1` | One-click bootstrap script for Modal + GitLab configuration |

---

## 🏆 Built for GitLab Transcend Hackathon

GitMesh: Orbit demonstrates the power of combining **GitLab Orbit** (spatial context RAG), **GitLab Duo** (AI-powered developer experience), and **serverless GPU infrastructure** to create a fully automated 3D asset pipeline that understands your repository.
