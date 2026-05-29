export interface Part {
  name: string;
  label: string;
  color: string;
  shape: 'box' | 'cylinder' | 'sphere' | 'torus';
  size: [number, number, number]; // width, height, depth
  position: [number, number, number]; // x, y, z
}

export interface AnimationStep {
  part: string;
  op: 'ROTATE_HINGE' | 'ROTATE_PIVOT' | 'SLIDE' | 'LATCH_RELEASE' | 'NONE';
  axis: [number, number, number];
  pivot: [number, number, number];
  angle_deg: number;
  duration_s: number;
  order: number;
}

export interface AnimationPlan {
  steps: AnimationStep[];
}

export interface AssetData {
  assetName: string;
  description: string;
  parts: Part[];
  animationPlan: AnimationPlan;
}

export interface PipelineStage {
  id: number;
  name: string;
  description: string;
  status: 'idle' | 'running' | 'completed' | 'failed';
  duration: string;
  icon: string;
}

export interface LogMessage {
  timestamp: string;
  source: 'AGENT' | 'GITLAB_MCP' | 'MODAL_COMPUTE' | 'VALIDATOR';
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
}