import express from "express";
import path from "path";
import fs from "fs/promises";
import { spawn } from "child_process";

const app = express();
const PORT = 3000;

app.use(express.json());

// Main HTML view to monitor and interact with the headless Python Agent
app.get("/", async (req, res) => {
  let agentCode = "";
  let requirementsContent = "";
  try {
    agentCode = await fs.readFile(path.join(process.cwd(), "agent.py"), "utf-8");
  } catch (e: any) {
    agentCode = `Failed to read agent.py: ${e.message}`;
  }

  try {
    requirementsContent = await fs.readFile(path.join(process.cwd(), "requirements.txt"), "utf-8");
  } catch (e: any) {
    requirementsContent = `Failed to read requirements.txt: ${e.message}`;
  }

  const html = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GitMesh — Headless AI Agent Pipeline</title>
  <style>
    :root {
      --bg-color: #0b0d12;
      --card-bg: #131722;
      --border-color: #262c3d;
      --text-main: #f3f4f6;
      --text-muted: #8b9bb4;
      --accent-color: #fc6d26; /* GitLab Orange */
      --accent-glow: rgba(252, 109, 38, 0.15);
      --terminal-green: #10b981;
      --font-mono: "Fira Code", "JetBrains Mono", SFMono-Regular, Consolas, monospace;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg-color);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.5;
      padding: 2rem;
      max-width: 1200px;
      margin: 0 auto;
    }

    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2rem;
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 1.5rem;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .brand-logo {
      width: 48px;
      height: 48px;
      background: linear-gradient(135deg, var(--accent-color), #e24329);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: bold;
      color: white;
      font-size: 1.5rem;
      box-shadow: 0 4px 12px var(--accent-glow);
    }

    .brand-text h1 {
      font-size: 1.5rem;
      font-weight: 700;
      letter-spacing: -0.025em;
    }

    .brand-text p {
      color: var(--text-muted);
      font-size: 0.875rem;
    }

    .badge {
      background-color: rgba(16, 185, 129, 0.1);
      color: var(--terminal-green);
      font-family: var(--font-mono);
      font-size: 0.75rem;
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      border: 1px solid rgba(16, 185, 129, 0.2);
    }

    .grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 1.5rem;
    }

    @media (min-width: 768px) {
      .grid {
        grid-template-columns: 350px 1fr;
      }
    }

    .sidebar {
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }

    .card {
      background-color: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 1.25rem;
    }

    .card-title {
      font-size: 0.875rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 1rem;
      font-weight: 600;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 100%;
      background: linear-gradient(135deg, var(--accent-color), #e24329);
      color: white;
      border: none;
      border-radius: 6px;
      padding: 0.75rem 1rem;
      font-weight: 600;
      font-size: 0.875rem;
      cursor: pointer;
      transition: opacity 0.2s, transform 0.1s;
    }

    .btn:hover {
      opacity: 0.9;
    }

    .btn:active {
      transform: scale(0.98);
    }

    .terminal-container {
      display: flex;
      flex-direction: column;
      height: 400px;
      margin-top: 1rem;
    }

    .terminal-header {
      background-color: #1a1f2c;
      border: 1px solid var(--border-color);
      border-bottom: none;
      border-top-left-radius: 6px;
      border-top-right-radius: 6px;
      padding: 0.5rem 1rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: var(--font-mono);
      font-size: 0.75rem;
      color: var(--text-muted);
    }

    .terminal-body {
      background-color: #05070a;
      border: 1px solid var(--border-color);
      border-bottom-left-radius: 6px;
      border-bottom-right-radius: 6px;
      padding: 1rem;
      flex-grow: 1;
      overflow-y: auto;
      font-family: var(--font-mono);
      font-size: 0.8125rem;
      color: #34d399;
      white-space: pre-wrap;
    }

    pre {
      font-family: var(--font-mono);
      font-size: 0.8125rem;
      background-color: #05070a;
      padding: 1rem;
      border-radius: 6px;
      border: 1px solid var(--border-color);
      overflow-x: auto;
      color: #93c5fd;
      max-height: 500px;
    }

    .env-var {
      display: flex;
      justify-content: space-between;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      padding: 0.375rem 0;
      font-size: 0.8125rem;
    }

    .env-name {
      font-family: var(--font-mono);
      color: #fca5a5;
    }

    .env-status {
      font-family: var(--font-mono);
    }

    .gitlab-logo {
      color: var(--accent-color);
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-logo">GM</div>
      <div class="brand-text">
        <h1>GitMesh Headless Console</h1>
        <p>CI/CD Automated Tech-Art Pipeling Agent System</p>
      </div>
    </div>
    <div class="badge">● Headless Python Mode</div>
  </header>

  <div class="grid">
    <div class="sidebar">
      <div class="card">
        <div class="card-title">Diagnostics</div>
        <div class="env-var">
          <span class="env-name">Environment</span>
          <span class="env-status" style="color: var(--terminal-green)">Cloud Sandboxed</span>
        </div>
        <div class="env-var">
          <span class="env-name">Python Engine</span>
          <span class="env-status" style="color: #60a5fa">python3 (v3.10+)</span>
        </div>
        <div class="env-var">
          <span class="env-name">GitLab MCP Server</span>
          <span class="env-status" style="color: #c084fc">stdio_client (Duo)</span>
        </div>
      </div>

      <div class="card">
        <div class="card-title">Interactive Trigger</div>
        <p style="font-size: 0.8125rem; color: var(--text-muted); margin-bottom: 1rem;">
          Trigger the headless Python agent subprocess directly in the sandboxed container. Read input, verify handshake details, and monitor standard output pipes.
        </p>
        <button class="btn" id="runBtn" onclick="runAgentPipe()">⚡ Trigger agent.py Execution</button>
      </div>

      <div class="card">
        <div class="card-title">requirements.txt</div>
        <pre style="max-height: 150px; padding: 0.5rem; font-size: 0.75rem;">${requirementsContent.trim()}</pre>
      </div>
    </div>

    <div class="main-content">
      <div class="card" style="margin-bottom: 1.5rem;">
        <div class="card-title">Live Execution Console Trace</div>
        <div class="terminal-container">
          <div class="terminal-header">
            <span>stdout & stderr buffer</span>
            <span id="termStatus" style="color: var(--text-muted)">idle</span>
          </div>
          <div class="terminal-body" id="consoleBody">Click "Trigger agent.py Execution" above to test initialized toolsets, dynamic GitLab handshakes, and Gemini capabilities.</div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">agent.py Skeleton Code</div>
        <pre>${escapeHtml(agentCode)}</pre>
      </div>
    </div>
  </div>

  <script>
    function escapeHtml(text) {
      return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    async function runAgentPipe() {
      const btn = document.getElementById("runBtn");
      const statusSpan = document.getElementById("termStatus");
      const consoleBody = document.getElementById("consoleBody");

      btn.disabled = true;
      btn.innerText = "Executing...";
      statusSpan.innerText = "running...";
      statusSpan.style.color = "var(--accent-color)";
      consoleBody.textContent = ">>> Launching process: python3 agent.py\\n\\n";

      try {
        const response = await fetch("/api/run-headless-agent", { method: "POST" });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunkStr = decoder.decode(value);
          consoleBody.textContent += chunkStr;
          consoleBody.scrollTop = consoleBody.scrollHeight;
        }
      } catch (err) {
        consoleBody.textContent += "\\n\\n[Connection Error] " + err.message;
      } finally {
        btn.disabled = false;
        btn.innerText = "⚡ Trigger agent.py Execution";
        statusSpan.innerText = "exited";
        statusSpan.style.color = "var(--text-muted)";
      }
    }
  </script>
</body>
</html>
  `;
  res.send(html);
});

// Post script route: Spawns the clean subprocess
app.post("/api/run-headless-agent", (req, res) => {
  res.setHeader("Content-Type", "text/plain");

  const pythonProc = spawn("python3", ["agent.py"], {
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1",
    },
  });

  pythonProc.stdout.on("data", (data) => {
    res.write(data.toString());
  });

  pythonProc.stderr.on("data", (data) => {
    res.write(data.toString());
  });

  pythonProc.on("close", (code) => {
    res.write(`\n--- Headless Agent Terminated with Process Exit Code: ${code} ---`);
    res.end();
  });
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`[GitMesh Headless Express Wrapper] Monitor running on http://localhost:${PORT}`);
});

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
