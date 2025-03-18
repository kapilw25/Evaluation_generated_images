import torch
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
import os

# CLear GPU cache to free memory
torch.cuda.empty_cache()

# Optimize PyTorch memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# shared prompt
prompt = "A cat holding a sign that says hello world"

# --------------------------------------- CV Model1: Stable Diffusion v1.5 ---------------------------------------
pipe_sd_v1_5 = StableDiffusionPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5",
    torch_dtype= torch.float16
)
pipe_sd_v1_5.enable_model_cpu_offload()   # Automatically offloads to CPU/GPU as needed

# Generate Image using Stable Diffusion v1.5
image_sd_v1_5 = pipe_sd_v1_5(prompt).images[0]
image_sd_v1_5.save("image_generated/CV1_3_stable_diffusion.jpg")

print(f"CV1: stable_diffusion's image generated successfully")

# --------------------------------------- CV Model2: SDXL Turbo ---------------------------------------
pipe_sdxl_turbo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
)

pipe_sdxl_turbo.enable_model_cpu_offload()

# Generate Image using SDXL Turbo
image_sdxl_turbo = pipe_sdxl_turbo(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=1,
    guidance_scale=0.0
).images[0]

image_sdxl_turbo.save("image_generated/CV3_3_sdxl_turbo.jpg")

print(f"CV2: sdxl_turbo's image generated successfully")