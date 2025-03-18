from diffusers import AutoPipelineForText2Image
import torch
torch.cuda.empty_cache() # Clear the cache to reduce memory usage

# export via python script
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load the pipeline with half precision to reduce VRAM usage, and move it to the GPU.
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", 
    torch_dtype=torch.float16, 
    variant="fp16"
)

pipe.enable_model_cpu_offload()  # Automatically offloads to CPU/GPU as needed

# Inference
prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt=prompt, 
    height=1024, 
    width=1024, 
    num_inference_steps=1, 
    guidance_scale=0.0
    ).images[0]

# save the image
image.save("CV3_1_sdxl_turbo.jpg")