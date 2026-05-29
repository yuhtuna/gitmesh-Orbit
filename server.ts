import express from "express";
import path from "path";
import fs from "fs/promises";
import { spawn } from "child_process";
import { createServer as createViteServer } from "vite";

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Enable JSON request body parsing
  app.use(express.json());

  // API Route: Check system status
  app.get("/api/health", (req, res) => {
    res.json({
      status: "ok",
      timestamp: new Date().toISOString(),
      python: "3.11+",
    });
  });

  // API Route: Retrieve agent.py content
  app.get("/api/agent-code", async (req, res) => {
    try {
      const codePath = path.join(process.cwd(), "agent.py");
      const content = await fs.readFile(codePath, "utf-8");
      res.json({ content });
    } catch (err: any) {
      res.status(500).json({ error: "Failed to read agent.py file: " + err.message });
    }
  });

  // API Route: Save edited agent.py content
  app.post("/api/save-agent-code", async (req, res) => {
    try {
      const { content } = req.body;
      if (typeof content !== "string") {
        return res.status(400).json({ error: "Invalid content payload" });
      }
      const codePath = path.join(process.cwd(), "agent.py");
      await fs.writeFile(codePath, content, "utf-8");
      res.json({ success: true, message: "agent.py updated successfully" });
    } catch (err: any) {
      res.status(500).json({ error: "Failed to write agent.py file: " + err.message });
    }
  });

  // API Route: Spawn agent.py or run full-fidelity simulation
  app.post("/api/run-agent", async (req, res) => {
    const { simulate = false, issueTitle = "Lowpoly chest", animationType = "idle" } = req.body;

    if (simulate) {
      // Simulate real-time streaming response for maximum interactive experience
      res.setHeader("Content-Type", "text/plain");
      res.setHeader("Transfer-Encoding", "chunked");

      const logs = [
        `[${new Date().toISOString()}] INFO - GitMesh - Starting GitMesh Pipeline Agent - Phase 1 Initialization`,
        `[${new Date().toISOString()}] INFO - GitMesh - Running environment diagnostics checks...`,
        `[${new Date().toISOString()}] INFO - GitMesh - 🔑 GEMINI_API_KEY detected in configuration secrets.`,
        `[${new Date().toISOString()}] INFO - GitMesh - 🛠️ GITLAB_PRIVATE_TOKEN detected in configuration secrets.`,
        `[${new Date().toISOString()}] INFO - GitMesh - Swapping handshakes and initializing protocol details...`,
        `[${new Date().toISOString()}] INFO - GitMesh - Connecting to GitLab MCP stream and retrieving active tools...`,
        `[${new Date().toISOString()}] INFO - GitMesh - 🎉 Session established! Retrieving GitLab capability tools...`,
        `[${new Date().toISOString()}] INFO - GitMesh - 📡 Found 8 GitLab MCP action tools available:`,
        `[${new Date().toISOString()}] INFO - GitMesh -   [1] Tool Name: 'create_issue' - Create a new issue in a GitLab project...`,
        `[${new Date().toISOString()}] INFO - GitMesh -   [2] Tool Name: 'search_merge_requests' - Search for MRs...`,
        `[${new Date().toISOString()}] INFO - GitMesh -   [3] Tool Name: 'create_commit' - Commit files onto a branch...`,
        `[${new Date().toISOString()}] INFO - GitMesh -   [4] Tool Name: 'create_merge_request' - Open an MR...`,
        `[${new Date().toISOString()}] INFO - GitMesh -   [5] Tool Name: 'post_issue_comment' - Add comments to issues...`,
        `[${new Date().toISOString()}] INFO - GitMesh -   [6] Tool Name: 'get_file_contents' - Read repo contents...`,
        `[${new Date().toISOString()}] INFO - GitMesh -   [7] Tool Name: 'edit_file_content' - Update content...`,
        `[${new Date().toISOString()}] INFO - GitMesh -   [8] Tool Name: 'get_job_logs' - Retrieve GitLab CI/CD runner details...`,
        `[${new Date().toISOString()}] INFO - GitMesh - 🛠️ Complete GitMesh Toolbelt Assembled: 11 total tools bindable.`,
        `[${new Date().toISOString()}] INFO - GitMesh - Injecting dynamic GitLab + Pipeline tools list into Gemini 3.1 Flash Brain...`,
        `[${new Date().toISOString()}] INFO - GitMesh - 🧠 Gemini 3.1 Agent Skeleton pre-configured & fully prepared for automated loops!`,
        `\n--- GITMESH AUTONOMOUS PIPELINE PIPES SIMULATION ---`,
        `[Pipeline Stage 1] Checking GitLab Issue #141 (Requested: '${issueTitle}')`,
        `[Pipeline Stage 2] Triggering 'imagen_tool' -> Art generated: 'storage://ref_${issueTitle.toLowerCase().replace(/\s+/g, "_")}.png'`,
        `[Pipeline Stage 3] Triggering 'fal_trellis_tool' -> 3D point cloud -> Reconstructed raw mesh 'mesh_${issueTitle.toLowerCase().replace(/\s+/g, "_")}_raw.glb'`,
        `[Pipeline Stage 4] Triggering 'modal_blender_tool' -> Rigged animation '${animationType}' configured. Clean UV mapping, skeletal keyframes baked.`,
        `[Pipeline Stage 5] Committing asset to GitLab repo 'game-assets/props/${issueTitle.toLowerCase().replace(/\s+/g, "_")}.glb'`,
        `[Pipeline Stage 6] Posting status comment with render preview directly on MR-141.`,
        `--- PIPELINE SIMULATION COMPLETED ---\n`,
        `[${new Date().toISOString()}] SUCCESS - Phase 1 Skeleton initialized and verification validation test passed successfully.`
      ];

      for (const log of logs) {
        res.write(log + "\n");
        await new Promise((resolve) => setTimeout(resolve, 300));
      }
      res.end();
      return;
    }

    // Try executing actual python3 script
    res.setHeader("Content-Type", "text/plain");
    const pythonProc = spawn("python3", ["agent.py"], {
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1"
      }
    });

    pythonProc.stdout.on("data", (data) => {
      res.write(data.toString());
    });

    pythonProc.stderr.on("data", (data) => {
      res.write(data.toString());
    });

    pythonProc.on("close", (code) => {
      res.write(`\n--- Process exited with code ${code} ---\n`);
      if (code !== 0) {
        res.write(`[SUGGESTION] Python environment may lack dependencies ('mcp' or 'google-genai'). You can toggle "Simulation Mode" to preview full flow features safely!`);
      }
      res.end();
    });
  });

  // Vite Integration: Serve assets
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app._router.stack.forEach((r: any) => {
      if (r.route && r.route.path === "*") {
        // Remove existing error routes or catch-alls
        app._router.stack = app._router.stack.filter((item: any) => item !== r);
      }
    });
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[GitMesh Server] Server running on http://localhost:${PORT} in ${process.env.NODE_ENV || 'dev'} mode.`);
  });
}

startServer().catch(err => {
  console.error("Failed to start server:", err);
});
