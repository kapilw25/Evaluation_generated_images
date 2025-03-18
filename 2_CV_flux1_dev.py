from diffusers import FluxPipeline
import torch
import os
from accelerate import infer_auto_device_map

torch.cuda.empty_cache() # Clear the cache to reduce memory usage
torch.cuda.ipc_collect()

# Optimize PyTorch memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
    use_auth_token=True,  # Add this for authenticated access
    low_cpu_mem_usage=True  # Key fix to reduce RAM usage
)

# Enable CPU offloading to save VRAM
pipe.enable_model_cpu_offload()

# Enable efficient memory management for GPU
device_map = infer_auto_device_map(pipe, max_memory={0: "6GiB", "cpu": "3GiB"})
pipe = pipe.to(device_map)


prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=256,
    width=256,
    guidance_scale=3.0,
    num_inference_steps=30,
    max_sequence_length=256,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("CV2_1_flux-dev.png")
