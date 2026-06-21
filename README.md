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
GitLab Duo monitors issue boards and chat triggers for the keyword `Meshgen:`. It parses the high-level request intent (e.g., `"Meshgen: Viking Sword"`).

### 2. Context (The Brain)
The orchestrator (`agent.py`) queries the **GitLab Orbit API** (`/orbit/nodes`) using the asset name. Orbit extracts:
- `target_folder`: The directory where the asset belongs (e.g., `Assets/Props/Weapons/`).
- `art_style`: Visual constraint guidelines (e.g., `lowpoly`).
- `target_dimensions`: Physical X/Y/Z coordinate bounds required by the project.

### 3. Generation & Scaling (The Muscle)
The orchestrator triggers the serverless **Trellis 2 pipeline on Modal** (using high-performance NVIDIA GPUs):
- **Trellis 2** reconstructs high-fidelity 3D meshes from reference concepts.
- **`trimesh` post-processing** calculates the generated mesh bounding box, computes uniform scaling factors to fit inside your `target_dimensions`, and rescales the geometry.

### 4. Write-Back (DevOps Loop)
The orchestrator receives the base64-encoded scaled `.glb` file, pushes it to a new branch, and automatically creates a Merge Request targeting `main`.

---

## 🚀 Setup & Installation

Follow these instructions to set up GitMesh: Orbit for your own GitLab projects.

### 1. Prerequisites
- **Python 3.10+** installed locally or in your CI/CD environment.
- **GitLab Project:** A target repository where you want 3D assets to be generated.
- **Modal Account:** Sign up at [Modal.com](https://modal.com) for serverless GPU access.

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
GITLAB_PROJECT_ID=your_namespace/your_project_name # e.g. username/my-game-repo

# Modal Cloud GPUs (Auto-generated if you run `modal setup`)
MODAL_TOKEN_ID=your_modal_token_id
MODAL_TOKEN_SECRET=your_modal_token_secret
```

### 4. Deploy the Compute Engine (Modal)
Deploy the serverless compute functions to Modal. This only needs to be done once, or whenever you update `modal_app.py`.
```bash
modal deploy modal_app.py
```

### 5. Running the Pipeline
You can trigger the orchestrator locally for testing, or set it up in your CI/CD pipeline triggered by GitLab Duo.

To run locally:
```bash
python agent.py "Meshgen: Viking Broadsword"
```

**What happens next?**
1. The agent queries your repository's Orbit API.
2. The serverless GPU generates and scales the mesh.
3. The agent commits the new asset and returns a **GitLab Merge Request URL** in your terminal!
