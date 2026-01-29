# handler.py
import os
import io
import time
import base64
import runpod
import torch
from PIL import Image
from diffusers import FluxPipeline

# -------------- Model init (once per worker) --------------
PIPE = None

def load_pipeline():
    """
    Load the FLUX.1-dev pipeline with Diffusers.
    Authentication:
      - Set HF_TOKEN environment variable in your endpoint settings.
    """
    global PIPE
    if PIPE is not None:
        return PIPE

    model_id = os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-dev")
    dtype = torch.bfloat16  # per HF examples for FLUX
    token = os.environ.get("HF_TOKEN")  # picked up by huggingface_hub if set

    # Download and load the model
    PIPE = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
    # Offload to CPU when idle to reduce VRAM; remove if you have ample GPU
    PIPE.enable_model_cpu_offload()

    return PIPE

# -------------- Utility --------------
def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# -------------- RunPod handler --------------
def handler(event):
    """
    Expected JSON:
    {
      "input": {
        "prompt": "A cinematic photo of ...",
        "negative_prompt": "",
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 40,
        "guidance_scale": 3.5,
        "max_sequence_length": 512,
        "seed": 12345
      }
    }
    """
    t0 = time.time()
    job_input = event.get("input", {}) or {}

    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "Missing required field: input.prompt"}

    negative_prompt   = job_input.get("negative_prompt", None)
    height            = int(job_input.get("height", 1024))
    width             = int(job_input.get("width", 1024))
    steps             = int(job_input.get("num_inference_steps", 40))
    guidance_scale    = float(job_input.get("guidance_scale", 3.5))
    max_seq_len       = int(job_input.get("max_sequence_length", 512))
    seed              = job_input.get("seed", None)

    # Seed handling
    generator = None
    if seed is not None:
        try:
            seed = int(seed)
        except Exception:
            return {"error": "input.seed must be an integer."}
        generator = torch.Generator("cpu").manual_seed(seed)

    try:
        pipe = load_pipeline()

        # Generate
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_seq_len,
            generator=generator,
            output_type="pil",
        ).images[0]

        image_b64 = pil_to_base64_png(image)
        return {
            "image_base64": image_b64,
            "metadata": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "max_sequence_length": max_seq_len,
                "seed": seed,
                "model_id": os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-dev"),
                "latency_sec": round(time.time() - t0, 3),
            }
        }
    except Exception as e:
        # Returning an "error" key signals a failed job to RunPod
        return {"error": f"Generation failed: {e}"}

# This starts the Serverless worker loop
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
