from diffusers import StableDiffusionPipeline
import torch
torch.cuda.empty_cache() # Clear the cache to reduce memory usage

# export via python script
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Model identifier for Stable Diffusion v1-5
model_id = "sd-legacy/stable-diffusion-v1-5"

# Load the pipeline with half precision to reduce VRAM usage, and move it to the GPU.
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16
    )
pipe.enable_model_cpu_offload()  # Automatically offloads to CPU/GPU as needed

# Inference
prompt = "A cat holding a sign that says hello world"

# Generate the image based on the prompt.
image = pipe(prompt).images[0]

# Save the generated image.
image.save("CV1_1_stable_diffusion.jpg")
