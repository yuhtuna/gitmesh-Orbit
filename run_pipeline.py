import subprocess
import sys
import os
import argparse
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import os
import argparse

def run_step(step, *args):
    print(f"Running {step}...")
    cmd = [r".\venv\Scripts\modal.exe", "run", f"modal_app.py::{step}"] + list(args)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf8"
    result = subprocess.run(cmd, capture_output=False, env=env)
    if result.returncode != 0:
        print(f"Failed at {step}")
        exit(1)

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, capture_output=False, shell=True)
    if result.returncode != 0:
        print(f"Failed cmd: {cmd}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()

    prompt = args.prompt
    glb_url = f"trellis_mesh_{prompt.replace(' ', '_')}_lowpoly.glb"

    print("Cleaning modal volume assets...")
    print(f"Starting pipeline for: {prompt}")

    run_step("generate_reference_image", "--prompt", prompt)
    run_step("generate_3d_mesh", "--prompt", prompt)
    run_step("segment_mesh", "--glb-url", glb_url, "--prompt-tags", "{}")
    run_step("label_parts", "--parts-json", "{}", "--asset-name", prompt)
    run_step("generate_animation_plan", "--labels-json", "{}", "--asset-name", prompt)
    run_step("animate_and_render_mesh", "--glb-url", glb_url, "--animation-plan-json", "{}")

    print(f"Pipeline finished for {prompt}!")
