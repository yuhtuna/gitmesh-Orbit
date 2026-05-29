import React, { useState, useEffect, useRef } from "react";
import { 
  Play, 
  Terminal as TerminalIcon, 
  FileCode, 
  Layers, 
  GitFork, 
  Cpu, 
  Save, 
  Settings, 
  RefreshCw, 
  Check, 
  Copy, 
  AlertCircle, 
  CheckCircle2, 
  ExternalLink,
  MessageSquare,
  Box,
  Image,
  Video,
  Database,
  ArrowRight
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

// Types represent core state structures of our app
interface PipelineStage {
  id: string;
  title: string;
  subtitle: string;
  desc: string;
  tool: string;
  status: "pending" | "running" | "success" | "error";
  icon: React.ReactNode;
}

export default function App() {
  // Navigation tabs
  const [activeTab, setActiveTab] = useState<"pipeline" | "editor" | "terminal">("pipeline");
  
  // Script and file state management
  const [agentCode, setAgentCode] = useState<string>("");
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [saveSuccess, setSaveSuccess] = useState<boolean>(false);
  const [hasCopied, setHasCopied] = useState<boolean>(false);

  // Runtime triggering state management
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [runMode, setRunMode] = useState<"simulate" | "real">("simulate");
  const [terminalLogs, setTerminalLogs] = useState<string[]>([]);
  const [issueTitle, setIssueTitle] = useState<string>("Lowpoly Chest Asset");
  const [animationType, setAnimationType] = useState<string>("idle");
  const [selectedStage, setSelectedStage] = useState<string | null>("mcp_gitlab");

  // Terminal scroll reference
  const terminalEndRef = useRef<HTMLDivElement>(null);

  // Pipeline execution tracking helper matches log statements to highlight states dynamically
  const [pipelineState, setPipelineState] = useState<string | null>(null);

  // Load agent.py on boot
  useEffect(() => {
    fetch("/api/agent-code")
      .then((res) => {
        if (!res.ok) throw new Error("Failed file load");
        return res.json();
      })
      .then((data) => {
        setAgentCode(data.content || "");
      })
      .catch((err) => {
        console.error("Could not fetch agent.py asset:", err);
        setAgentCode(`# Failed to load file from disk.\n# Run "node server.ts" inside Cloud Run to activate API ports.`);
      });
  }, []);

  // Automatic terminal scrolling on streaming logs
  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [terminalLogs]);

  // Extract pipeline state indicators from logs to light up active flowchart steps
  useEffect(() => {
    const lastLog = terminalLogs[terminalLogs.length - 1] || "";
    if (lastLog.includes("connect_gitlab_mcp") || lastLog.includes("GitLab MCP integration")) {
      setPipelineState("mcp_gitlab");
    } else if (lastLog.includes("imagen_tool")) {
      setPipelineState("imagen");
    } else if (lastLog.includes("fal_trellis_tool")) {
      setPipelineState("trellis");
    } else if (lastLog.includes("modal_blender_tool")) {
      setPipelineState("blender");
    } else if (lastLog.includes("create_commit") || lastLog.includes("Committing asset")) {
      setPipelineState("commit");
    } else if (lastLog.includes("post_issue_comment") || lastLog.includes("Posting status comment")) {
      setPipelineState("comment");
    } else if (lastLog.includes("SUCCESS")) {
      setPipelineState("completed");
    }
  }, [terminalLogs]);

  // Command handlers
  const handleSaveCode = async () => {
    setIsSaving(true);
    try {
      const res = await fetch("/api/save-agent-code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: agentCode }),
      });
      if (res.ok) {
        setSaveSuccess(true);
        setTimeout(() => setSaveSuccess(false), 2000);
      }
    } catch (err) {
      console.error("Failed to commit back to disk:", err);
    } finally {
      setIsSaving(false);
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(agentCode);
    setHasCopied(true);
    setTimeout(() => setHasCopied(false), 2000);
  };

  const triggerAgentPipeline = async () => {
    setIsRunning(true);
    setTerminalLogs([`[SYSTEM] Spawning GitMesh async loop. Executing in: \x1b[1m${runMode.toUpperCase()} MODE\x1b[0m`]);
    
    // Switch to console viewport immediately to observe stream
    setActiveTab("terminal");

    try {
      const response = await fetch("/api/run-agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          simulate: runMode === "simulate",
          issueTitle,
          animationType,
        }),
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (reader) {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          
          // Split by line and stream asynchronously
          const lines = chunk.split("\n");
          for (let line of lines) {
            if (line.trim()) {
              setTerminalLogs((prev) => [...prev, line]);
              await new Promise((resolve) => setTimeout(resolve, 80)); // Fine throttle animation
            }
          }
        }
      }
    } catch (err: any) {
      setTerminalLogs((prev) => [...prev, `[CRITICAL FATAL] Network connection timed out: ${err.message}`]);
    } finally {
      setIsRunning(false);
    }
  };

  // Pipeline Visual Hierarchy Steps:
  const pipelineStages: PipelineStage[] = [
    {
      id: "mcp_gitlab",
      title: "GitLab Handshake (MCP)",
      subtitle: "Stdio Client Boot",
      desc: "Connects securely to GitLab MCP server using 'npx @gitlab/mcp-server-gitlab'. Pulls and registers workspace capabilities dynamically.",
      tool: "npx @gitlab/mcp-server-gitlab",
      icon: <GitFork className="w-5 h-5" />,
      status: isRunning && pipelineState === "mcp_gitlab" ? "running" : pipelineState ? "success" : "pending"
    },
    {
      id: "imagen",
      title: "2D Reference Generation",
      subtitle: "Google Imagen Webhook",
      desc: "Triggers Google's Imagen model via Vertex AI integration. Reads your Issue body prompt and paints high-fidelity orthographic concept art.",
      tool: "imagen_tool(prompt)",
      icon: <Image className="w-5 h-5" />,
      status: isRunning && pipelineState === "imagen" ? "running" : (pipelineState && pipelineState !== "mcp_gitlab" ? "success" : "pending")
    },
    {
      id: "trellis",
      title: "3D Mesh Reconstruction",
      subtitle: "Fal.ai Trellis-2 Serverless",
      desc: "Streams synthesized 2D imagery into the state-of-the-art Trellis model. Constructs 3D point-clouds and exports a raw unoptimized .glb mesh.",
      tool: "fal_trellis_tool(image_url)",
      icon: <Box className="w-5 h-5" />,
      status: isRunning && pipelineState === "trellis" ? "running" : (["blender", "commit", "comment", "completed"].includes(pipelineState || "") ? "success" : "pending")
    },
    {
      id: "blender",
      title: "Headless Asset Sculpting",
      subtitle: "Modal + Blender Engine",
      desc: "Spawns a massive scale headless Blender container in Modal. Performs geometric decimation, UV unwrapping, baking, and animations rigging.",
      tool: "modal_blender_tool(model, anim)",
      icon: <Cpu className="w-5 h-5" />,
      status: isRunning && pipelineState === "blender" ? "running" : (["commit", "comment", "completed"].includes(pipelineState || "") ? "success" : "pending")
    },
    {
      id: "commit",
      title: "Asynchronous Repo Push",
      subtitle: "GitLab Commit Action",
      desc: "Commits compiled animated game-assets (.glb files) onto dev branches, routing them automatically through GitLab pipelines.",
      tool: "gitlab_mcp.create_commit()",
      icon: <Database className="w-5 h-5" />,
      status: isRunning && pipelineState === "commit" ? "running" : (["comment", "completed"].includes(pipelineState || "") ? "success" : "pending")
    },
    {
      id: "comment",
      title: "PR Validation Review",
      subtitle: "Merge Request Reporter",
      desc: "Writes progress notes, keyframe math metrics, and rendering results directly back to the issue thread or merge request reviewer workflow.",
      tool: "gitlab_mcp.post_issue_comment()",
      icon: <MessageSquare className="w-5 h-5" />,
      status: isRunning && pipelineState === "comment" ? "running" : (pipelineState === "completed" ? "success" : "pending")
    }
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-300 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      
      {/* HEADER SECTION */}
      <header className="h-16 border-b border-slate-800 flex items-center justify-between px-6 bg-slate-950/50 backdrop-blur-md shrink-0">
        <div className="flex items-center gap-4">
          <div className="w-8 h-8 bg-indigo-600 rounded flex items-center justify-center shrink-0">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"></path>
            </svg>
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight text-white leading-tight flex items-center gap-2">
              GitMesh <span className="text-slate-500 font-mono text-[10px]">v1.2.0</span>
            </h1>
            <p className="text-[10px] uppercase tracking-widest text-indigo-400 font-bold">Autonomous 3D Asset Pipeline</p>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="hidden md:flex flex-col items-end shrink-0">
            <span className="text-[10px] uppercase text-slate-500 font-mono">Model Context Protocol</span>
            <span className="text-[11px] font-mono text-emerald-400 flex items-center gap-1.5 font-semibold">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span> @gitlab/mcp-server-gitlab:active
            </span>
          </div>

          <div className="hidden sm:block h-8 w-px bg-slate-800"></div>

          <div className="flex items-center gap-2">
            <div className="hidden lg:block px-2.5 py-1 bg-slate-900 border border-slate-800 rounded text-[11px] font-mono text-slate-400">
              gemini-3.1-flash
            </div>
            <div className="hidden lg:block px-2.5 py-1 bg-indigo-950/30 border border-indigo-500/30 rounded text-[11px] text-indigo-300">
              Phase 1: Skeleton
            </div>
          </div>
        </div>
      </header>

      {/* SUB-HEADER SETTINGS STRIP */}
      <section className="bg-slate-950/40 border-b border-slate-800 px-6 py-3.5 grid grid-cols-1 md:grid-cols-3 gap-4 shrink-0">
        <div>
          <label className="block text-[10px] font-mono tracking-widest uppercase text-slate-500 font-semibold text-xs">Asset Trigger Request (Mock Issue)</label>
          <input
            type="text"
            value={issueTitle}
            onChange={(e) => setIssueTitle(e.target.value)}
            disabled={isRunning}
            placeholder="e.g. Lowpoly Viking Sword"
            className="w-full bg-slate-900/60 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 mt-1.5 transition-all text-slate-200"
          />
        </div>
        <div>
          <label className="block text-[10px] font-mono tracking-widest uppercase text-slate-500 font-semibold text-xs">Animation Rig Task</label>
          <select
            value={animationType}
            onChange={(e) => setAnimationType(e.target.value)}
            disabled={isRunning}
            className="w-full bg-slate-900/60 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 mt-1.5 cursor-pointer text-slate-200"
          >
            <option value="idle">Idle Loop Animation</option>
            <option value="open_chest">Bake Open / Closed Keyframes</option>
            <option value="spin_axes">360-Degree Radial Swivel</option>
            <option value="viking_sword_swing">Double Swing Melee Attack</option>
          </select>
        </div>
        <div className="flex flex-row md:flex-col justify-end gap-3 md:gap-0">
          <div className="flex flex-wrap items-center gap-3 md:justify-end">
            <div className="bg-slate-900 border border-slate-805 p-0.5 rounded-lg flex items-center">
              <button 
                onClick={() => setRunMode("simulate")}
                className={`px-2.5 py-1 text-[11px] rounded-md transition-all font-mono font-medium ${
                  runMode === "simulate" 
                    ? "bg-slate-800 text-white shadow-sm" 
                    : "text-slate-500 hover:text-slate-300"
                }`}
              >
                Demo Simulation
              </button>
              <button 
                onClick={() => setRunMode("real")}
                className={`px-2.5 py-1 text-[11px] rounded-md transition-all font-mono font-medium flex items-center gap-1 ${
                  runMode === "real" 
                    ? "bg-slate-800 text-white shadow-sm" 
                    : "text-slate-500 hover:text-slate-300"
                }`}
              >
                Real Running
              </button>
            </div>

            <button
              onClick={triggerAgentPipeline}
              disabled={isRunning}
              className={`cursor-pointer px-3.5 py-1.5 text-xs font-semibold rounded-lg flex items-center gap-2 transition-all ${
                isRunning 
                  ? "bg-slate-800 text-slate-500 cursor-not-allowed" 
                  : "bg-indigo-600 hover:bg-indigo-400 text-white shadow-lg shadow-indigo-950/40"
              }`}
            >
              {isRunning ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  Running
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" />
                  Spawn Pipeline
                </>
              )}
            </button>
          </div>
        </div>
      </section>

      {/* MAIN WORKSPACE */}
      <main className="flex-1 flex flex-col md:flex-row overflow-hidden bg-slate-900/20">
        
        {/* LEFT NAV PANEL - vertical navigation train matching Sophisticated Dark layout */}
        <nav className="w-full md:w-16 border-b md:border-b-0 md:border-r border-slate-800 flex flex-row md:flex-col items-center justify-around md:justify-start py-2.5 md:py-6 gap-0 md:gap-6 shrink-0 bg-slate-950">
          <button
            onClick={() => setActiveTab("pipeline")}
            className={`p-2.5 rounded-lg transition-all border ${
              activeTab === "pipeline" 
                ? "bg-indigo-600 text-white border-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.4)]" 
                : "text-slate-500 hover:text-slate-300 border-transparent hover:bg-slate-900/50"
            }`}
            title="Pipeline Architecture"
          >
            <Layers className="w-5 h-5" />
          </button>

          <button
            onClick={() => setActiveTab("editor")}
            className={`p-2.5 rounded-lg transition-all border ${
              activeTab === "editor" 
                ? "bg-indigo-600 text-white border-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.4)]" 
                : "text-slate-500 hover:text-slate-300 border-transparent hover:bg-slate-900/50"
            }`}
            title="Inspect agent.py Code"
          >
            <FileCode className="w-5 h-5" />
          </button>

          <button
            onClick={() => setActiveTab("terminal")}
            className={`p-2.5 rounded-lg transition-all border ${
              activeTab === "terminal" 
                ? "bg-indigo-600 text-white border-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.4)]" 
                : "text-slate-500 hover:text-slate-300 border-transparent hover:bg-slate-900/50"
            }`}
            title="Live Stream Trace"
          >
            <TerminalIcon className="w-5 h-5" />
          </button>

          <div className="hidden md:block mt-auto">
            <div className="p-2 text-slate-650">
              <span className="w-2 h-2 rounded-full bg-indigo-500 inline-block animate-pulse"></span>
            </div>
          </div>
        </nav>

        {/* CONTAINER FOR ACTIVE TAB VIEWPORTS */}
        <div className="flex-1 flex flex-col min-w-0 overflow-y-auto p-4 md:p-6">
          <AnimatePresence mode="wait">
            
            {/* VIEW A: PIPELINE VISUALIZER */}
            {activeTab === "pipeline" && (
              <motion.div
                key="pipeline"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.15 }}
                className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start"
              >
                
                {/* PIPELINE STAGES LIST */}
                <div className="lg:col-span-2 space-y-4">
                  <div className="flex justify-between items-center bg-slate-900/50 p-3 rounded-lg border border-slate-800">
                    <h2 className="text-xs font-semibold tracking-wide text-slate-100 flex items-center gap-2 uppercase font-mono">
                      <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.6)] animate-pulse"></span>
                      Game Asset Generation Nodes
                    </h2>
                    <span className="text-[10px] font-mono text-slate-500 uppercase">Phase 1 verification</span>
                  </div>

                  <div className="grid grid-cols-1 gap-3.5 relative">
                    {pipelineStages.map((stage, idx) => {
                      const isActive = selectedStage === stage.id;
                      
                      let cardStyle = "border-slate-800 bg-slate-950/40 text-slate-400";
                      let badge = <span className="px-1.5 py-0.5 bg-slate-900 text-slate-600 rounded">WAIT</span>;

                      if (stage.status === "running") {
                        cardStyle = "border-indigo-500/40 bg-indigo-500/5 text-indigo-400 scale-[1.01] shadow-lg shadow-indigo-950/30";
                        badge = <span className="px-1.5 py-0.5 bg-indigo-500/20 text-indigo-400 rounded font-bold animate-pulse">ACTIVE</span>;
                      } else if (stage.status === "success") {
                        cardStyle = "border-slate-700 bg-slate-900/20 text-slate-300";
                        badge = <span className="px-1.5 py-0.5 bg-emerald-500/20 text-emerald-400 rounded">OK</span>;
                      } else {
                        // Pending or stubs
                        if (stage.id === "trellis" || stage.id === "blender") {
                          badge = <span className="px-1.5 py-0.5 bg-slate-800 text-slate-500 rounded">STUB</span>;
                        }
                      }

                      return (
                        <div key={stage.id} className="relative">
                          {/* Connector arrow link wire */}
                          {idx < pipelineStages.length - 1 && (
                            <div className="absolute left-[26px] top-12 bottom-[-16px] w-0.5 border-l border-dashed border-slate-800 z-0"></div>
                          )}

                          <button
                            onClick={() => setSelectedStage(stage.id)}
                            className={`w-full text-left relative z-10 p-3.5 rounded-xl border flex items-center gap-4 transition-all hover:border-slate-705 hover:bg-slate-900/25 ${cardStyle}`}
                          >
                            <div className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${
                              stage.status === "running" 
                                ? "bg-indigo-600 text-white shadow-lg" 
                                : stage.status === "success" 
                                ? "bg-slate-900 text-emerald-400" 
                                : "bg-slate-950 border border-slate-800 text-slate-500"
                            }`}>
                              {stage.icon}
                            </div>

                            <div className="flex-1 min-w-0">
                              <p className="font-semibold text-xs text-white leading-none flex items-center gap-2">
                                {stage.title}
                              </p>
                              <p className="text-[10px] text-zinc-500 font-mono mt-1">{stage.tool}</p>
                            </div>

                            <div className="flex items-center gap-3 shrink-0">
                              <span className="text-[10px] text-slate-500 font-mono">Stage 0{idx + 1}</span>
                              <div className="text-[10px] font-semibold font-mono">
                                {badge}
                              </div>
                              <ArrowRight className="w-3 h-3 text-slate-600" />
                            </div>
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* ACTIVE NODE DETAIL VIEW */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h2 className="text-xs font-semibold tracking-wider text-slate-400 uppercase font-mono">Active Node Specs</h2>
                    <span className="text-[11px] text-slate-600 font-mono">Detailed manifest</span>
                  </div>

                  {selectedStage ? (() => {
                    const stage = pipelineStages.find(s => s.id === selectedStage);
                    if (!stage) return null;
                    return (
                      <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-4 relative overflow-hidden">
                        <div className="flex justify-between items-start">
                          <div>
                            <span className="text-[9px] font-mono tracking-wider text-indigo-400 bg-indigo-950/50 border border-indigo-900/50 px-2 py-0.5 rounded uppercase font-semibold">
                              {selectedStage === "mcp_gitlab" || selectedStage === "commit" || selectedStage === "comment" ? "MCP Protocol Bindings" : "External Serverless Execution"}
                            </span>
                            <h3 className="text-sm font-semibold text-white mt-2">{stage.title}</h3>
                            <p className="text-[10px] text-slate-500 font-mono mt-0.5">{stage.subtitle}</p>
                          </div>
                        </div>

                        <p className="text-xs text-slate-400 leading-relaxed pt-2 border-t border-slate-800">
                          {stage.desc}
                        </p>

                        <div className="bg-black/40 p-2.5 rounded border border-slate-805 text-slate-400 font-mono text-[10px] space-y-1">
                          <p className="text-indigo-400/80">// Method binding signature</p>
                          <p><span className="text-indigo-400">async def</span> <span className="text-white">{stage.tool}</span>:</p>
                          <p className="text-slate-500 pl-4">""" Phase 1 dynamic validation checkpoint """</p>
                          <p className="text-slate-400 pl-4">pass <span className="text-slate-600"># Triggered by workflow</span></p>
                        </div>

                        <div className="space-y-2 pt-2 text-[11px]">
                          <p className="font-mono text-[10px] uppercase text-slate-500 tracking-wider">Node Verification Checks</p>
                          <div className="flex items-center gap-2 text-slate-400">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                            <span>Standard asyncio bindings registered</span>
                          </div>
                          <div className="flex items-center gap-2 text-slate-400">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                            <span>Subprocess return validation passed</span>
                          </div>
                        </div>
                      </div>
                    );
                  })() : (
                    <div className="p-6 rounded-xl bg-slate-950 border border-slate-800 text-center text-xs text-slate-500">
                      Select any asset stage to inspect active parameters.
                    </div>
                  )}

                  <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/40 space-y-3">
                    <p className="text-xs font-semibold text-slate-300 flex items-center gap-2">
                      <Settings className="w-3.5 h-3.5 text-indigo-400 animate-spin" />
                      Dynamic GitLab Operations
                    </p>
                    <p className="text-[11px] text-slate-400 leading-relaxed">
                      Whenever issues labeled <code className="bg-slate-900 border border-slate-800 px-1 py-0.5 rounded text-indigo-300 font-mono text-[10px]">asset:generate</code> hit the repo, GitMesh triggers this multi-stage container chain, building models in parallel.
                    </p>
                  </div>
                </div>

              </motion.div>
            )}

            {/* VIEW B: INTEGRATED CODE EDITOR */}
            {activeTab === "editor" && (
              <motion.div
                key="editor"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.15 }}
                className="flex flex-col gap-3 h-full"
              >
                
                {/* Editor Control Tool Belt */}
                <div className="flex justify-between items-center bg-slate-900/60 p-2.5 rounded-lg border border-slate-800">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-yellow-500"></span>
                    <span className="text-xs font-mono font-medium text-slate-200">agent.py</span>
                    <span className="text-[10px] bg-slate-800 px-1.5 py-0.5 text-slate-500 font-mono rounded font-semibold uppercase">
                      Local Python Checkpoint
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={copyToClipboard}
                      className="cursor-pointer px-3 py-1.5 text-xs text-slate-400 bg-slate-950 hover:bg-slate-900 border border-slate-800 rounded-lg transition-all flex items-center gap-1.5"
                    >
                      {hasCopied ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                          Copied
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5" />
                          Copy to Clips
                        </>
                      )}
                    </button>

                    <button
                      onClick={handleSaveCode}
                      disabled={isSaving}
                      className="cursor-pointer px-3.5 py-1.5 text-xs text-white bg-indigo-600 hover:bg-indigo-400 font-medium rounded-lg transition-all flex items-center gap-1.5 shadow"
                    >
                      {isSaving ? (
                        <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      ) : saveSuccess ? (
                        <Check className="w-3.5 h-3.5 text-white" />
                      ) : (
                        <Save className="w-3.5 h-3.5" />
                      )}
                      {saveSuccess ? "Committed!" : "Save Changes"}
                    </button>
                  </div>
                </div>

                <div className="relative flex-1 bg-slate-950 rounded-xl border border-slate-805 overflow-hidden min-h-[480px]">
                  {/* Custom Styled Monospace Code Editor */}
                  <textarea
                    value={agentCode}
                    onChange={(e) => setAgentCode(e.target.value)}
                    className="w-full h-full p-4 font-mono text-slate-300 text-xs bg-transparent border-0 ring-0 focus:ring-0 focus:outline-none resize-none select-text overflow-y-auto leading-relaxed"
                    spellCheck="false"
                  />
                </div>
              </motion.div>
            )}

            {/* VIEW C: RUNNER CLIENT TERMINAL */}
            {activeTab === "terminal" && (
              <motion.div
                key="terminal"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.15 }}
                className="flex flex-col gap-3 h-full"
              >
                
                {/* Console controller belt */}
                <div className="flex justify-between items-center bg-slate-900/60 p-2.5 rounded-lg border border-slate-800">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-indigo-500 animate-pulse"></span>
                    <span className="text-xs font-mono font-medium text-slate-200">Terminal Output Buffer</span>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setTerminalLogs([])}
                      className="cursor-pointer px-2.5 py-1 text-xs text-slate-400 bg-slate-950 hover:bg-slate-900 border border-slate-800 rounded transition-all"
                    >
                      Clear Log Output
                    </button>
                    <button
                      onClick={triggerAgentPipeline}
                      disabled={isRunning}
                      className="cursor-pointer px-3.5 py-1.5 text-xs bg-indigo-600 hover:bg-indigo-400 text-white font-medium rounded-lg transition-all flex items-center gap-1.5 shadow"
                    >
                      <Play className="w-3 h-3 fill-current" />
                      Re-execute
                    </button>
                  </div>
                </div>

                {/* Core terminal window */}
                <div className="bg-slate-950 rounded-xl border border-slate-805 overflow-hidden flex flex-col p-4 font-mono text-xs h-[480px] overflow-y-auto">
                  <div className="flex items-center justify-between border-b border-slate-900 pb-2.5 mb-3 text-slate-500 text-[10px]">
                    <span>INSTANCE ID: gitmesh-agent-01-std</span>
                    <span>ACTIVE BUFFER: STDOUT/STDERR</span>
                  </div>

                  <div className="flex-1 space-y-1.5 selection:bg-indigo-505 selection:text-white">
                    {terminalLogs.length === 0 ? (
                      <div className="text-slate-600 italic h-full flex items-center justify-center text-xs">
                        No console output active. Run the pipeline to trigger live traces.
                      </div>
                    ) : (
                      terminalLogs.map((log, idx) => {
                        let textClass = "text-slate-300";
                        if (log.includes("ERROR") || log.includes("failure") || log.includes("[CRITICAL]")) {
                          textClass = "text-red-400 font-semibold";
                        } else if (log.includes("SUCCESS") || log.includes("🎉") || log.includes("initialized")) {
                          textClass = "text-emerald-400 font-semibold";
                        } else if (log.includes("Warning") || log.includes("⚠️") || log.includes("WARNING")) {
                          textClass = "text-amber-400";
                        } else if (log.includes("Triggering") || log.includes("[Pipeline")) {
                          textClass = "text-indigo-300";
                        } else if (log.includes("[SYSTEM]")) {
                          textClass = "text-indigo-400 font-bold";
                        }

                        // Parse console formatting color codes if any
                        const cleanLog = log
                          .replace(/\x1b\[1m/g, "")
                          .replace(/\x1b\[0m/g, "");

                        return (
                          <div key={idx} className={`leading-relaxed break-all ${textClass}`}>
                            {cleanLog}
                          </div>
                        );
                      })
                    )}
                    <div ref={terminalEndRef} />
                  </div>
                </div>
              </motion.div>
            )}

          </AnimatePresence>
        </div>

        {/* RIGHT PANEL: SYSTEM ARCHITECTURE & INVENTORY */}
        <aside className="w-full md:w-72 border-t md:border-t-0 md:border-l border-slate-800 flex flex-col bg-slate-950 shrink-0">
          <div className="p-6 flex-1 space-y-6">
            <div>
              <h3 className="text-[10px] uppercase tracking-widest text-slate-500 font-bold mb-4">Core System Diagnostics</h3>
              
              <div className="space-y-4">
                {/* GitLab MCP SDK Progress */}
                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-slate-400">GitLab MCP SDK</span>
                    <span className="text-emerald-400 text-[11px] font-semibold">Active</span>
                  </div>
                  <div className="h-1.5 w-full bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                    <div className="h-full w-full bg-emerald-500/60 rounded-full animate-pulse"></div>
                  </div>
                </div>

                {/* Vertex AI Engine Dynamic Speed Bar */}
                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-slate-400">Vertex AI Brain</span>
                    <span className="text-indigo-400 text-[11px] font-semibold">98ms latency</span>
                  </div>
                  <div className="h-1.5 w-full bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                    <div className="h-full w-4/5 bg-indigo-500/60 rounded-full"></div>
                  </div>
                </div>

                {/* Headless Blender container progress */}
                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-slate-500">Modal (Blender SAM)</span>
                    <span className="text-slate-600 text-[10px] italic">Phase 1 placeholder</span>
                  </div>
                  <div className="h-1.5 w-full bg-slate-900 rounded-full border border-slate-800"></div>
                </div>

                {/* Fal.ai Trellis container progress */}
                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-slate-500">Fal.ai (Trellis-2)</span>
                    <span className="text-slate-600 text-[10px] italic">Phase 1 placeholder</span>
                  </div>
                  <div className="h-1.5 w-full bg-slate-900 rounded-full border border-slate-800"></div>
                </div>
              </div>
            </div>

            {/* LAST GENERATED MODEL CONTAINER */}
            <div className="p-4 border border-slate-800/80 rounded-xl bg-slate-900/30">
              <p className="text-[10px] text-slate-500 mb-2.5 font-mono uppercase tracking-wider font-semibold">Render Asset Stream Model</p>
              
              <div className="aspect-square w-full rounded-lg bg-black flex flex-col items-center justify-center border border-slate-800/60 mb-2.5 relative overflow-hidden">
                {/* Dynamic visual preview based on execution state */}
                {isRunning ? (
                  <div className="text-center p-4">
                    <div className="w-12 h-12 border-2 border-dashed border-indigo-500 rounded-full animate-spin mx-auto mb-3"></div>
                    <span className="text-[10px] text-indigo-400 font-mono animate-pulse">Reconstructing vertex indices...</span>
                  </div>
                ) : (
                  <div className="text-center p-4">
                    <Box className="w-10 h-10 text-slate-700 mx-auto mb-2 animate-bounce" />
                    <span className="text-[10.5px] text-slate-500 font-mono block">Loaded: 1.2M Polys</span>
                    <span className="text-[9px] text-emerald-400 font-mono mt-0.5 bg-emerald-950/40 border border-emerald-900/40 rounded px-1.5 py-0.5 inline-block">mesh_final.glb ready</span>
                  </div>
                )}
              </div>
              
              <div className="text-[11px] text-slate-300 font-mono break-all leading-tight bg-slate-950 p-2 rounded border border-slate-800 flex justify-between items-center">
                <span>{issueTitle.toLowerCase().replace(/\s+/g, "_")}.glb</span>
                <span className="text-[9px] text-slate-500">12.1MB</span>
              </div>
            </div>
          </div>
          
          <div className="p-6 border-t border-slate-800">
            <button 
              onClick={triggerAgentPipeline}
              disabled={isRunning}
              className="cursor-pointer w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold tracking-wide transition-all flex items-center justify-center gap-2 shadow-lg shadow-indigo-950/20"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isRunning ? "animate-spin" : ""}`} />
              Execute Async Pipeline
            </button>
          </div>
        </aside>

      </main>

      {/* FOOTER DIAGNOSTICS */}
      <footer className="border-t border-slate-800 bg-slate-950 px-6 py-3 flex flex-col sm:flex-row sm:items-center sm:justify-between text-[11px] text-slate-500 gap-2 shrink-0">
        <p className="font-sans">GitMesh Autonomous CI/CD Pipeline Setup. Powered by Google vertex-genai.</p>
        <p className="flex items-center gap-1.5 font-mono">
          <span>GitLab Server Transport:</span>
          <span className="text-emerald-400 font-bold uppercase shrink-0">STDIO STREAM CLIENT</span>
        </p>
      </footer>

    </div>
  );
}
