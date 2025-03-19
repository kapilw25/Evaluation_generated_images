import torch
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
import os

# CLear GPU cache to free memory
torch.cuda.empty_cache()

# Optimize PyTorch memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ensure the 'models' and 'image_generated' directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("image_generated", exist_ok=True)

# Customer cache path for downloaded models
os.environ["HF_HOME"] = os.path.abspath("models")

# shared prompt
prompt = "A cat holding a sign that says hello world"

# --------------------------------------- CV Model1: Stable Diffusion v1.5 ---------------------------------------
model_sd_v1_5_path = "models/stable-diffusion-v1-5"

#Download or load from Local "models" directory
if not os.path.exists(model_sd_v1_5_path):
    pipe_sd_v1_5 = StableDiffusionPipeline.from_pretrained(
        "sd-legacy/stable-diffusion-v1-5",
        torch_dtype= torch.float16,
        cache_dir = model_sd_v1_5_path
    )
else:
    pipe_sd_v1_5 = StableDiffusionPipeline.from_pretrained(model_sd_v1_5_path)
    
pipe_sd_v1_5.enable_model_cpu_offload()   # Automatically offloads to CPU/GPU as needed

# Generate Image using Stable Diffusion v1.5
image_sd_v1_5 = pipe_sd_v1_5(prompt).images[0]
image_sd_v1_5.save("image_generated/CV1_4_stable_diffusion.jpg")

print(f"CV1: stable_diffusion's image generated successfully")

# --------------------------------------- CV Model2: SDXL Turbo ---------------------------------------
model_sdxl_turbo_path = "models/sdxl-turbo"

# Download or Load from Local
if not os.path.exists(model_sdxl_turbo_path):
    pipe_sdxl_turbo = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir = model_sdxl_turbo_path
    )
else:
    pipe_sdxl_turbo = AutoPipelineForText2Image.from_pretrained(model_sdxl_turbo_path)

pipe_sdxl_turbo.enable_model_cpu_offload()

# Generate Image using SDXL Turbo
image_sdxl_turbo = pipe_sdxl_turbo(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=1,
    guidance_scale=0.0
).images[0]

image_sdxl_turbo.save("image_generated/CV3_4_sdxl_turbo.jpg")

print(f"CV2: sdxl_turbo's image generated successfully")