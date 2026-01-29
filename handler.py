import os
import runpod
import torch
import base64
from io import BytesIO
from diffusers import FluxPipeline

# ------------------------------
# 1. Load Hugging Face token from environment
# ------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN environment variable not set! "
        "Please add it in RunPod dashboard."
    )

# ------------------------------
# 2. Load the FLUX model once at startup
# ------------------------------
print("Loading FLUX model...")
try:
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        use_auth_token=HF_TOKEN,
        torch_dtype=torch.float16
    )
    pipe.to("cuda")
    print("FLUX model loaded successfully.")
except Exception as e:
    print(f"Error loading FLUX model: {e}")
    raise e

# ------------------------------
# 3. Helper function to generate Base64 image
# ------------------------------
def generate_image(prompt: str, width: int, height: int, steps: int) -> str:
    try:
        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=3.5
        ).images[0]

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error generating image: {e}")
        raise e

# ------------------------------
# 4. Serverless handler
# ------------------------------
def handler(event):
    """
    Expected input JSON:
    {
        "input": {
            "prompt": "A futuristic city at sunset",
            "width": 512,
            "height": 512,
            "steps": 20
        }
    }
    """
    input_data = event.get("input", {})

    prompt = input_data.get("prompt", "A beautiful landscape")
    width = input_data.get("width", 512)
    height = input_data.get("height", 512)
    steps = input_data.get("steps", 20)

    try:
        image_base64 = generate_image(prompt, width, height, steps)
        return {
            "output": {
                "image": image_base64,
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": steps
            }
        }
    except Exception as e:
        return {"error": str(e)}

# ------------------------------
# 5. Start RunPod serverless
# ------------------------------
runpod.serverless.start({"handler": handler})
