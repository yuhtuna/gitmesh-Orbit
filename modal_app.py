#!/usr/bin/env python3
"""
modal_app.py - Modal Compute Serverless App for GitMesh: Orbit.

Provides:
  1) Reference Image Generation (generate_reference_image)
  2) Trellis 2 3D Reconstruction (generate_3d_mesh) with bounding box scaling.
"""

import os
import sys
import json
import tempfile
import base64
import trimesh
from typing import Dict, Any, Optional

os.environ["ATTN_BACKEND"] = "xformers"

# Check if modal is present
try:
    import modal
    
    pipeline_image = (
        modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
        .env({
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "8.9",
            "MAX_JOBS": "1",
            "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "ATTN_BACKEND": "xformers",
            "CXX": "g++",
            "CC": "gcc"
        })
        .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "build-essential", "ninja-build", "cmake")
        .pip_install("torch==2.4.0", "torchvision", "torchaudio", "xformers", extra_options="--index-url https://download.pytorch.org/whl/cu121")
        .pip_install("setuptools", "wheel", "ninja", "pybind11")
        .pip_install("git+https://github.com/NVlabs/nvdiffrast.git", extra_options="--no-build-isolation")
        .run_commands("rm -rf /usr/lib/python3/dist-packages/blinker*")
        .pip_install("kaolin", extra_options="-f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html")
        .pip_install(
            "pillow", "imageio", "imageio-ffmpeg", "tqdm", "easydict",
            "opencv-python-headless", "scipy", "ninja", "rembg", "onnxruntime",
            "trimesh", "open3d", "xatlas", "pyvista", "pymeshfix", "igraph",
            "transformers==4.44.2",
            "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
            "huggingface_hub", "spconv-cu121",
            "viser", "numba", "gradio", "safetensors",
            "accelerate", "diffusers", "requests",
            "pymcubes", "google-cloud-aiplatform", "google-auth", "plyfile", "google-generativeai", "google-genai",
        )
        .pip_install("torch-scatter", "torch-cluster", extra_options="-f https://data.pyg.org/whl/torch-2.4.0+cu121.html")
        .pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl")
        .run_commands(
            "git clone --recurse-submodules https://github.com/microsoft/TRELLIS /trellis",
            "git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast /diffoctreerast",
            "python -m pip install --no-build-isolation /diffoctreerast",
            "git clone --recurse-submodules https://github.com/autonomousvision/mip-splatting /mip-splatting",
            "python -m pip install --no-build-isolation /mip-splatting/submodules/diff-gaussian-rasterization",
        )
    )

    app = modal.App(name="gitmesh-compute")
    storage_volume = modal.Volume.from_name("gitmesh-storage", create_if_missing=True)
except ImportError:
    class MockApp:
        def function(self, *args, **kwargs):
            return lambda f: f
    modal = None
    app = MockApp()
    pipeline_image = None
    storage_volume = None

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "").strip()
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")
EXPECTED_CUDA_CAPABILITY = (8, 9)

def _require_gcp_project_id() -> str:
    if not GCP_PROJECT_ID:
        raise RuntimeError("GCP_PROJECT_ID is required for Vertex AI calls.")
    return GCP_PROJECT_ID

def _ensure_l4_gpu_runtime(stage_name: str) -> None:
    try:
        import torch
    except Exception as e:
        raise RuntimeError(f"[{stage_name}] Unable to import torch: {e}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"[{stage_name}] CUDA GPU is not available.")
    actual_capability = torch.cuda.get_device_capability(0)
    if actual_capability != EXPECTED_CUDA_CAPABILITY:
        gpu_name = torch.cuda.get_device_name(0)
        raise RuntimeError(f"[{stage_name}] Unsupported GPU '{gpu_name}' with capability {actual_capability}.")

def _get_llm_provider() -> str:
    return os.environ.get("LLM_PROVIDER", "vertex").strip().lower()

def _get_image_model_candidates() -> list[str]:
    image_override = (os.environ.get("IMAGE_MODEL") or "").strip()
    quality_mode = (os.environ.get("QUALITY_MODE") or "med").strip().lower()
    preferred = image_override or ("imagen-4.0-fast-generate-001" if quality_mode == "low" else "gemini-3.1-flash-image")
    return [preferred, "imagen-4.0-fast-generate-001", "gemini-3.1-flash-image", "gemini-3.5-flash"]

def _get_vertex_credentials() -> Optional[Any]:
    service_account_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if service_account_json:
        from google.oauth2 import service_account
        return service_account.Credentials.from_service_account_info(
            json.loads(service_account_json),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    gcp_token = os.environ.get("GCP_ACCESS_TOKEN") or os.environ.get("GCLOUD_TOKEN")
    if gcp_token:
        import google.oauth2.credentials
        return google.oauth2.credentials.Credentials(token=gcp_token)
    return None

def _call_gemini_vertex(prompt: str, model_name: str) -> Optional[str]:
    if _get_llm_provider() not in {"vertex", "auto"}:
        return None
    try:
        from google import genai
        client = genai.Client(
            vertexai=True,
            project=_require_gcp_project_id(),
            location=VERTEX_LOCATION,
            credentials=_get_vertex_credentials()
        )
        response = client.models.generate_content(model=model_name, contents=prompt)
        return response.text
    except Exception as e:
        print(f"⚠️ Vertex AI call failed: {e}")
        return None

def _call_gemini_api(prompt: str, model_name: str, gemini_api_key: Optional[str] = None) -> Optional[str]:
    if _get_llm_provider() not in {"gemini", "auto"}:
        return None
    api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model_name, contents=prompt)
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Gemini API call failed: {e}")
        return None

def _generate_imagen_vertex(prompt: str) -> Optional[bytes]:
    if _get_llm_provider() not in {"vertex", "auto"}:
        return None
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(
            vertexai=True,
            project=_require_gcp_project_id(),
            location=VERTEX_LOCATION,
            credentials=_get_vertex_credentials()
        )
        for model_id in _get_image_model_candidates():
            try:
                print(f"🖼️ Attempting Vertex image model: {model_id}")
                if "gemini" in model_id.lower():
                    response = client.models.generate_content(
                        model=model_id,
                        contents=prompt,
                        config=types.GenerateContentConfig(response_modalities=[types.Modality.IMAGE])
                    )
                    for part in response.candidates[0].content.parts:
                        if part.inline_data:
                            return part.inline_data.data
                else:
                    result = client.models.generate_images(
                        model=model_id,
                        prompt=prompt,
                        config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio="1:1")
                    )
                    if result.generated_images:
                        return result.generated_images[0].image.image_bytes
            except Exception as inner_e:
                print(f"⚠️ Vertex image model failed ({model_id}): {inner_e}")
    except Exception as e:
        print(f"⚠️ _generate_imagen_vertex failed: {e}")
    return None

def _generate_imagen_gemini_api(prompt: str, gemini_api_key: Optional[str] = None) -> Optional[bytes]:
    if _get_llm_provider() not in {"gemini", "auto"}:
        return None
    api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        for model_id in _get_image_model_candidates():
            try:
                print(f"🖼️ Attempting Gemini API image model: {model_id}")
                if "gemini" in model_id.lower():
                    response = client.models.generate_content(
                        model=model_id,
                        contents=prompt,
                        config=types.GenerateContentConfig(response_modalities=[types.Modality.IMAGE])
                    )
                    for part in response.candidates[0].content.parts:
                        if part.inline_data:
                            return part.inline_data.data
                else:
                    result = client.models.generate_images(
                        model=model_id,
                        prompt=prompt,
                        config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio="1:1")
                    )
                    if result.generated_images:
                        return result.generated_images[0].image.image_bytes
            except Exception as inner_e:
                print(f"⚠️ Gemini API image model failed ({model_id}): {inner_e}")
    except Exception as e:
        print(f"⚠️ Gemini API Imagen generation failed: {e}")
    return None

@app.function(
    image=pipeline_image,
    gpu="L4",
    timeout=300,
    secrets=[modal.Secret.from_name("gitmesh-keys")] if modal else [],
    volumes={"/mnt/data": storage_volume} if storage_volume else {}
)
def generate_reference_image(
    prompt: str,
    issue_desc: str = "",
    google_access_token: str = None
) -> Dict[str, Any]:
    if google_access_token:
        os.environ["GOOGLE_ACCESS_TOKEN"] = google_access_token
        os.environ["GCP_ACCESS_TOKEN"] = google_access_token
    from PIL import Image, ImageDraw

    _ensure_l4_gpu_runtime("Stage 2: generate_reference_image")

    storage_dir = "/mnt/data/assets"
    os.makedirs(storage_dir, exist_ok=True)
    ref_path = os.path.join(storage_dir, "reference.png")

    # Enhance prompt and categorize
    enhanced_prompt = prompt
    category = "prop"
    subcategory = "general"
    filename = prompt.lower().replace(" ", "_")
    inferred_dimensions = [800.0, 400.0, 300.0]
    inferred_poly_count = 3000

    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    ai_instruction = (
        f"You are a 3D game asset classification, prompt enhancement, and scaling AI.\n"
        f"Analyze this request for a 3D asset: '{prompt}. {issue_desc}'\n"
        f"Generate a JSON object with the following fields:\n"
        f"1. 'enhanced_prompt': A descriptive physical prompt optimized for 3D generation models (describing details, materials, lighting, isolate on a gray background, etc.).\n"
        f"2. 'category': A single word category for this asset (e.g., castle, house, weapon, furniture, vehicle, character, prop, etc.). Keep it lowercase.\n"
        f"3. 'subcategory': A single word subcategory (e.g., door, chair, table, sword, pistol, tree, etc.). Keep it lowercase.\n"
        f"4. 'filename': A clean slugified filename (words separated by underscores, e.g., laser_pistol, wooden_door). Do not include file extension.\n"
        f"5. 'inferred_dimensions': A 3-element float list [x, y, z] in millimeters representing realistic bounding dimensions for this object (e.g. a sword could be [200.0, 50.0, 1200.0], a door [1000.0, 100.0, 2100.0], a chair [600.0, 600.0, 950.0], a chest [800.0, 500.0, 500.0]).\n"
        f"6. 'inferred_poly_count': An integer budget representing a target polygon limit for this asset type (e.g., 2000 for props, 4000 for weapons, 5000 for animated furniture).\n"
        f"Return ONLY a valid JSON object. Do not add any markdown formatting or comments outside the JSON."
    )
    
    res = _call_gemini_vertex(ai_instruction, "gemini-3.5-flash")
    if not res:
        res = _call_gemini_api(ai_instruction, "gemini-3.5-flash", gemini_api_key=gemini_api_key)
        
    if res:
        try:
            # Clean up markdown code blocks if any
            clean_res = res.strip()
            if clean_res.startswith("```"):
                lines = clean_res.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                clean_res = "\n".join(lines).strip()
            
            data = json.loads(clean_res)
            enhanced_prompt = data.get("enhanced_prompt", prompt)
            category = data.get("category", "prop").strip().lower()
            subcategory = data.get("subcategory", "general").strip().lower()
            filename = data.get("filename", prompt.lower().replace(" ", "_")).strip().lower()
            filename = "".join(c for c in filename if c.isalnum() or c in "_-")
            
            if "inferred_dimensions" in data and isinstance(data["inferred_dimensions"], list) and len(data["inferred_dimensions"]) >= 3:
                inferred_dimensions = [float(d) for d in data["inferred_dimensions"][:3]]
            if "inferred_poly_count" in data:
                inferred_poly_count = int(data["inferred_poly_count"])
        except Exception as parse_err:
            print(f"⚠️ Failed to parse classification JSON: {parse_err}. Response was: {res}")

    # Generate Image
    img_bytes = _generate_imagen_vertex(enhanced_prompt)
    if not img_bytes:
        img_bytes = _generate_imagen_gemini_api(enhanced_prompt, gemini_api_key=gemini_api_key)

    if img_bytes:
        with open(ref_path, "wb") as f:
            f.write(img_bytes)
        print(f"📷 Reference image saved to {ref_path}")
    else:
        # Procedural fallback
        img = Image.new("RGB", (1024, 1024), color=(30, 30, 30))
        draw = ImageDraw.Draw(img)
        draw.ellipse([256, 256, 768, 768], fill=(70, 130, 180), outline=(255, 255, 255), width=8)
        img.save(ref_path)
        print(f"📷 Fallback procedural image saved to {ref_path}")

    # Read base64
    with open(ref_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "status": "success",
        "reference_path": ref_path,
        "enhanced_prompt": enhanced_prompt,
        "image_base64": img_base64,
        "category": category,
        "subcategory": subcategory,
        "filename": filename,
        "inferred_dimensions": inferred_dimensions,
        "inferred_poly_count": inferred_poly_count
    }

@app.function(
    image=pipeline_image,
    gpu="L4",
    timeout=600,
    secrets=[modal.Secret.from_name("gitmesh-keys")] if modal else [],
    volumes={"/mnt/data": storage_volume} if storage_volume else {}
)
def generate_3d_mesh(
    prompt: str,
    style: str = "lowpoly",
    issue_desc: str = "",
    google_access_token: str = None,
    limit_x: Optional[float] = None,
    limit_y: Optional[float] = None,
    limit_z: Optional[float] = None,
    target_dimensions: list[float] = None,
    image_base64: Optional[str] = None
) -> Dict[str, Any]:
    if google_access_token:
        os.environ["GOOGLE_ACCESS_TOKEN"] = google_access_token
        os.environ["GCP_ACCESS_TOKEN"] = google_access_token
    import sys
    import tempfile

    _ensure_l4_gpu_runtime("Stage 3: generate_3d_mesh")

    if "/trellis" not in sys.path:
        sys.path.insert(0, "/trellis")
    
    os.environ["U2NET_HOME"] = "/mnt/data/assets/u2net"
    storage_dir = "/mnt/data/assets"
    os.makedirs(storage_dir, exist_ok=True)
    glb_filename = f"mesh_{prompt.lower().replace(' ', '_')}_{style}.glb"
    glb_path = os.path.join(storage_dir, glb_filename)

    # Load reference image if base64 provided, else fallback to file or solid blue
    from PIL import Image
    import io
    
    img = None
    concept_img_path = os.path.join(storage_dir, "reference.png")
    
    if image_base64:
        try:
            print("📷 Loading reference image from base64 argument...")
            img_data = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            
            # Save it to the volume so subsequent stages can access it
            img.save(concept_img_path)
            print(f"📷 Saved base64 reference image to volume path: {concept_img_path}")
            if storage_volume:
                storage_volume.commit()
                print("Storage volume committed successfully.")
        except Exception as img_err:
            print(f"⚠️ Failed to decode/save base64 reference image: {img_err}")
            
    if img is None:
        if storage_volume:
            try:
                print("Re-syncing storage volume...")
                storage_volume.reload()
            except Exception as vol_err:
                print(f"⚠️ Volume reload failed: {vol_err}")
            
        if os.path.exists(concept_img_path):
            try:
                img = Image.open(concept_img_path).convert("RGB")
                print(f"📷 Loaded reference image from volume path: {concept_img_path}")
            except Exception as read_err:
                print(f"⚠️ Failed to read reference image from volume: {read_err}")
                
    if img is None:
        print("⚠️ No reference image found or loaded. Using solid blue fallback image.")
        img = Image.new("RGB", (1024, 1024), color=(70, 130, 180))

    try:
        from trellis.pipelines import TrellisImageTo3DPipeline
        from trellis.utils import postprocessing_utils

        print("🔮 Loading Trellis pipeline...")
        pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        pipeline.cuda()

        outputs = pipeline.run(
            img,
            seed=42,
            sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
            slat_sampler_params={"steps": 12, "cfg_strength": 3.0}
        )

        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=0.95,
            texture_size=1024
        )
        glb.export(glb_path)
        print(f"✅ TRELLIS 3D Generation successful. Exported to {glb_path}")
    except Exception as e:
        print(f"⚠️ TRELLIS execution failed: {e}. Generating procedural fallback cube...")
        # Fallback minimal valid GLB
        _write_minimal_glb(glb_path)

    # Uniform Scaling to target dimensions (Aspect Ratio Preservation)
    resolved_dimensions = target_dimensions
    if resolved_dimensions is None and (limit_x is not None or limit_y is not None or limit_z is not None):
        resolved_dimensions = [limit_x, limit_y, limit_z]

    if resolved_dimensions is not None:
        try:
            print(f"⚖️ Bounding box scaling to target dimensions: {resolved_dimensions}...")
            mesh = trimesh.load(glb_path)
            extents = mesh.extents
            print(f"Current extents: {extents}")
            
            factors = []
            for i in range(min(len(resolved_dimensions), 3)):
                if resolved_dimensions[i] is not None and resolved_dimensions[i] > 0 and extents[i] > 0:
                    factors.append(float(resolved_dimensions[i]) / extents[i])
                    
            if factors:
                scale_factor = min(factors)
                print(f"Applying uniform scale factor: {scale_factor}")
                mesh.apply_scale(scale_factor)
                mesh.export(glb_path)
                print(f"Scaled mesh exported. New extents: {mesh.extents}")
        except Exception as scale_err:
            print(f"⚠️ Error during bounding box scaling: {scale_err}. Falling back to unscaled asset.")

    # Read base64 GLB content
    glb_base64 = ""
    file_size_kb = 0
    if os.path.exists(glb_path):
        file_size_kb = round(os.path.getsize(glb_path) / 1024, 2)
        try:
            with open(glb_path, "rb") as f:
                glb_base64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as read_err:
            print(f"⚠️ Error reading GLB: {read_err}")

    return {
        "status": "success",
        "glb_path": glb_path,
        "file_size_kb": file_size_kb,
        "glb_base64": glb_base64
    }

def _write_minimal_glb(filepath: str):
    import struct
    import json
    vertices = [
        -0.5,-0.5, 0.5,  0.5,-0.5, 0.5,  0.5, 0.5, 0.5, -0.5, 0.5, 0.5,
        -0.5,-0.5,-0.5,  0.5,-0.5,-0.5,  0.5, 0.5,-0.5, -0.5, 0.5,-0.5,
    ]
    indices = [
        0,1,2, 0,2,3, 4,5,6, 4,6,7, 0,4,7, 0,7,3,
        1,5,6, 1,6,2, 0,1,5, 0,5,4, 2,6,7, 2,7,3,
    ]
    verts_bytes = struct.pack('<' + 'f'*24, *vertices)
    idxs_bytes = struct.pack('<' + 'H'*36, *indices)
    json_data = json.dumps({
        "asset": {"version": "2.0"},
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(verts_bytes)},
            {"buffer": 0, "byteOffset": len(verts_bytes), "byteLength": len(idxs_bytes)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3", "max": [0.5,0.5,0.5], "min": [-0.5,-0.5,-0.5]},
            {"bufferView": 1, "componentType": 5123, "count": 36, "type": "SCALAR"},
        ],
        "buffers": [{"byteLength": len(verts_bytes) + len(idxs_bytes)}],
    })
    json_bytes = json_data.encode('utf-8')
    while len(json_bytes) % 4 != 0:
        json_bytes += b' '
    bin_data = verts_bytes + idxs_bytes
    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    header = struct.pack('<I', 0x46546C67)
    header += struct.pack('<I', 2)
    header += struct.pack('<I', total_len)
    json_chunk = struct.pack('<I', len(json_bytes)) + struct.pack('<I', 0x4E4F534A) + json_bytes
    bin_chunk = struct.pack('<I', len(bin_data)) + struct.pack('<I', 0x004E4942) + bin_data
    with open(filepath, 'wb') as f:
        f.write(header + json_chunk + bin_chunk)
