# ADK Migration Tracker

## Goal
Move from direct CI stage orchestration to Google ADK-first orchestration for remote GitLab execution, while preserving a safe rollback path.

## Agent and Runtime Inventory

- Google ADK agent runtime:
  - File: agent.py
  - Current role: Local orchestration and ADK workflow testing
  - Target role: Primary remote orchestrator entrypoint

- Modal webhook runtime:
  - File: gitlab_webhook.py
  - Current role: Receives GitLab Issue Open events and triggers pipeline
  - Target role: Keep as trigger gateway (no major change)

- Modal compute runtime:
  - File: modal_app.py
  - Current role: Executes pipeline stages (image, mesh, segmentation, labels, animation, export)
  - Target role: Keep as tool/function execution backend

- GitLab CI orchestration runtime:
  - File: .gitlab-ci.yml
  - Current role: Directly calls modal_app.py functions stage by stage
  - Target role: Calls ADK orchestrator as primary path, direct chain as fallback during transition

- Remote bootstrap runtime:
  - File: setup_remote.ps1
  - Current role: Sync secrets, deploy Modal apps, set GitLab vars/hooks
  - Target role: Keep as deployment/bootstrap utility

## Agent and Function Matrix

Note on counts:

- Physical ADK agent instances right now: 1 (`agent.py`)
- Logical pipeline agents (role-based decomposition): many (listed below)
- Reason you saw only 3 earlier: `agent.py` currently exposes only 3 wrapper tools, not the full role-based agent map yet.

## Logical Multi-Agent Design (Role-Based)

These are the target logical agents for ADK orchestration, each mapped to existing runtime functions.

1. Request Intake Agent
- Responsibility: Parse issue request, normalize prompt, initialize run context
- Input source: GitLab issue payload/CI variables

2. Reference Image Generator Agent
- Responsibility: Create reference image prompt and generate image
- Function mapping: `modal_app.generate_reference_image`

3. 3D Mesh Generator Agent
- Responsibility: Generate base GLB from prompt/image context
- Function mapping: `modal_app.generate_3d_mesh`

4. GLB Output Validator Agent
- Responsibility: Validate generated GLB integrity and readiness
- Function mapping: `modal_app.validate_glb`

5. Mesh Segmentation Agent
- Responsibility: Segment mesh into semantic parts
- Function mapping: `modal_app.segment_mesh`

6. Part Labeling Agent
- Responsibility: Assign semantic labels to segmented parts
- Function mapping: `modal_app.label_parts`

7. Animation Plan Generator Agent
- Responsibility: Create structured animation plan per part/archetype
- Function mapping: `modal_app.generate_animation_plan`

8. Animation Plan Validator Agent
- Responsibility: Verify/fix plan geometry and motion constraints
- Function mapping: `modal_app.validate_animation_plan`

9. Blender Render and Export Agent
- Responsibility: Execute Stage 10 render/export and publish artifacts
- Function mapping: `modal_app.animate_and_render_mesh`

10. GitLab Progress Reporter Agent
- Responsibility: Post stage comments and run status to GitLab
- Function mapping: Existing GitLab API calls in `.gitlab-ci.yml` and helper calls in `modal_app.py`

11. Delivery and Completion Agent
- Responsibility: Final status update, optional issue close, handoff links
- Function mapping: Existing finalization logic in `.gitlab-ci.yml`

### A) Google ADK Agent

- Agent name: GitMesh ADK Agent (single agent instance)
- File: agent.py
- Agent constructor: Agent(...) in agent.py
- Current purpose: Orchestrate tool calls from prompt/issue text, with MCP + pipeline toolbelt

Agent-local tool wrappers (defined in agent.py):

- run_generate_3d_mesh(prompt, style)
  - Calls: modal_app.generate_3d_mesh (or fallback simulation)
  - Purpose: Base mesh generation step

- run_segment_mesh(glb_url, prompt_tags)
  - Calls: modal_app.segment_mesh (or fallback simulation)
  - Purpose: Semantic segmentation step

- run_animate_and_render_mesh(glb_url, animation_plan_json)
  - Calls: modal_app.animate_and_render_mesh (or fallback simulation)
  - Purpose: Stage 10 animation/render step

Dynamic MCP tools (loaded at runtime in agent.py):

- Source: GitLab MCP server via npx @gitlab/mcp-server-gitlab
- Purpose: Branch/MR/comments/files operations depending on discovered tool schema

### B) Remote Modal Webhook App

- App name: gitmesh-webhook
- File: gitlab_webhook.py
- Modal app object: modal.App("gitmesh-webhook")

Function(s):

- gitlab_issue_listener(req)
  - Trigger: GitLab webhook (issue opened)
  - Purpose: Validate issue + prefix, trigger GitLab pipeline with issue variables

### C) Remote Modal Compute App

- App name: gitmesh-compute
- File: modal_app.py
- Modal app object: modal.App(name="gitmesh-compute")

Primary pipeline functions (CI-invoked):

- generate_reference_image
  - Purpose: Stage 2 image generation

- generate_3d_mesh
  - Purpose: Stage 3 mesh generation

- validate_glb
  - Purpose: Stage 3b integrity checks

- segment_mesh
  - Purpose: Stage 4 segmentation

- label_parts
  - Purpose: Stage 7 semantic labeling

- generate_animation_plan
  - Purpose: Stage 8 animation planning

- validate_animation_plan
  - Purpose: Stage 9 plan validation/fix

- animate_and_render_mesh
  - Purpose: Stage 10 animation export

Additional utility/testing functions:

- list_hunyuan_files
- check_trellis_repo

### D) CI Orchestration Layer (Not an agent runtime)

- File: .gitlab-ci.yml
- Role: Calls modal functions in sequence, posts issue comments, handles completion/close logic
- Current mode: Direct stage chain (with optional dry-run branch)

## Scope

In scope:
- Add remote-capable ADK orchestrator mode
- Wire CI to ADK-first execution
- Keep rollback flag and fallback path during cutover
- Add explicit step logs for judging/demo evidence

Out of scope (for initial cutover):
- Rewriting compute functions in modal_app.py
- Full replacement of webhook behavior
- Major UI/server refactors

## Implementation Plan

### Phase 1: ADK Remote Entry Mode
Status: implemented

- Add explicit remote runner mode in agent.py
- Consume ISSUE_TITLE, ISSUE_DESC, ISSUE_IID, GITLAB_API_TOKEN from env/args
- Invoke existing stage functions/tools through ADK toolbelt
- Emit deterministic stage logs

Implementation notes:
- `agent.py --remote-ci` now runs ADK-first remote orchestration.
- The runner names role-based logical agents and dispatches the existing Modal stage functions.
- GitLab issue comments are posted from the ADK orchestration path.

Success criteria:
- ADK path runs end-to-end in a controlled local or CI simulation
- No regressions in existing direct stage calls

### Phase 2: CI ADK-First Wiring
Status: implemented

- Add USE_ADK_ORCHESTRATOR=true/false variable in .gitlab-ci.yml
- If true: run ADK orchestrator path
- If false: run existing direct stage chain
- Add ADK_HARD_FAIL=true/false
  - false: fallback to direct chain if ADK fails
  - true: fail fast on ADK failure

Implementation notes:
- `.gitlab-ci.yml` defaults `USE_ADK_ORCHESTRATOR=true`.
- `.gitlab-ci.yml` defaults `ADK_HARD_FAIL=false` so legacy fallback remains available during early cutover.

Success criteria:
- CI job starts and uses ADK when flag is true
- Fallback behavior is explicit and tested

### Phase 3: Observability and Evidence
Status: partially implemented

- Add clear per-stage logs and summary outputs
- Keep issue comments consistent with stage progression
- Capture ADK planning/decision traces suitable for demo

Success criteria:
- Logs are understandable for judges and maintainers
- Failure points can be diagnosed quickly

Implementation notes:
- `agent.py --remote-ci` prints an ADK supervisor plan when available.
- The remote runner posts issue comments naming ADK logical agents for each stage.
- `.gitlab-ci.yml` posts an explicit ADK fallback warning when ADK fails and fallback is enabled.

### Phase 4: Stabilization and Cutover
Status: pending

- Run repeated remote tests with USE_ADK_ORCHESTRATOR=true
- Keep ADK_HARD_FAIL=false initially
- Flip ADK_HARD_FAIL=true after stable runs
- Remove or deprecate direct chain only when risk is acceptable

Success criteria:
- Stable runs across multiple real prompts
- Team confidence in ADK-only path

## Rollback Plan

- Immediate rollback switch:
  - USE_ADK_ORCHESTRATOR=false
- Soft rollback while preserving ADK attempt:
  - USE_ADK_ORCHESTRATOR=true
  - ADK_HARD_FAIL=false

## Risks

- MCP/network instability in CI environment
- Tool mismatch between ADK planning and runtime function contracts
- Hidden regressions in issue-comment/close flow

Mitigations:
- Keep direct chain as fallback until stable
- Keep stage contracts unchanged in modal_app.py
- Add explicit validation and logging in ADK runner

## Current Decision Log

- Decision: Proceed with ADK-first migration using phased cutover instead of one-shot replacement.
- Reason: Meets hackathon requirement while protecting delivery reliability.
- Decision: Convert repository docs and file layout to remote-first operational scope.
- Change set: Removed local-only helper and Node/web files (`run_modal.ps1`, `run_modal.sh`, `run_modal_download.ps1`, `run_pipeline.py`, `server.ts`, `package.json`, `package-lock.json`, `Dockerfile.pipeline`).
- Reason: Keep hackathon deliverable focused on remote GitLab + Modal pipeline path and reduce maintainer confusion.
- Decision: Restore optional local helper files after reassessing contributor/debug value.
- Reason: Remote remains primary, but optional local smoke/debug paths are useful and documented at the end of README.
- Decision: Add ADK-first CI execution with legacy fallback.
- Reason: Satisfies ADK orchestration requirement while preserving delivery reliability during cutover.
- Decision: Make setup automation sync ADK CI control flags.
- Change set: `setup_remote.ps1` now defaults and upserts `USE_ADK_ORCHESTRATOR`, `ADK_HARD_FAIL`, and `PIPELINE_DRY_RUN`.
- Reason: Keep remote automatic behavior controlled by bootstrap rather than undocumented manual GitLab variable edits.
