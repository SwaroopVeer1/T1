import runpod
import torch
import base64
from io import BytesIO
from diffusers import FluxPipeline

# Load model once at startup (important for serverless performance)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16
)

pipe.to("cuda")

def generate_image(prompt, width, height, steps):
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

def handler(event):
    """
    Expected input:
    {
        "prompt": "A futuristic city at sunset",
        "width": 1024,
        "height": 1024,
        "steps": 30
    }
    """

    input_data = event.get("input", {})

    prompt = input_data.get("prompt", "A beautiful landscape")
    width = input_data.get("width", 1024)
    height = input_data.get("height", 1024)
    steps = input_data.get("steps", 30)

    image_base64 = generate_image(prompt, width, height, steps)

    return {
        "image": image_base64
    }

# Start RunPod serverless
runpod.serverless.start({"handler": handler})
