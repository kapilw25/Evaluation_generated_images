import torch
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
import os
from evaluation import EvaluationMetrics
from PIL import Image

# CLear GPU cache to free memory
torch.cuda.empty_cache()

# Optimize PyTorch memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ensure the 'models' and 'image_generated' directories exist
os.makedirs("image_generated", exist_ok=True)

# remove manual model caching logic and rely on HF default cache directory
if "HF_HOME" in os.environ:
    del os.environ["HF_HOME"]

# shared prompt
prompts = ["A cat holding a sign that says hello world",
           "A futuristic city with flying cars",
           "An astronaut riding a horse in space"
           ]

# -----CV model1: Stable Diffusion v1.5---------------------------------------
pipe_sd_v1_5 = StableDiffusionPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5",
    torch_dtype= torch.float16
)
    
pipe_sd_v1_5.enable_model_cpu_offload()   # Automatically offloads to CPU/GPU as needed

# ----- CV Model2: SDXL Turbo ---------------------------------------
pipe_sdxl_turbo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
)

pipe_sdxl_turbo.enable_model_cpu_offload()
# ------------------------- Batch Image Generation -------------------------
models = {
    "StableDiffusion" : pipe_sd_v1_5,
    "SDXL_Turbo" : pipe_sdxl_turbo
}

images = {}
for model_name, pipe in models.items():
    for i, prompt in enumerate(prompts):
        image = pipe(prompt).images[0]
        image_path = f"image_generated/{model_name}_image_{i}.jpg"
        image.save(image_path)
        images.setdefault(model_name, []).append(image_path)
        
# ------------------------- Evaluation -------------------------
# for model_name, img_paths in images.items():
#     generated_images = [Image.open(img) for img  in img_paths]
    
#     # Evaluation Metrics
#     clip_scores = EvaluationMetrics.calculate_clip_score(generated_images, prompts[0])
#     bleu_scores = EvaluationMetrics.calculate_bleu_score(prompts[0], prompts)
#     cosine_scores = EvaluationMetrics.calculate_cosine_similarity(generated_images, prompts[0])
#     inference_time = EvaluationMetrics.measure_inference_time(models[model_name], prompts)
#     gpu_memory = EvaluationMetrics.measure_gpu_memory()
#     throughput = EvaluationMetrics.measure_throughput(inference_time * len(prompts), len(prompts))
    
#     print(f"{model_name} Metrics:")
#     print(f"CLIP scores: {clip_scores}")
#     print(f"BLEU scores: {bleu_scores}")
#     print(f" Cosine similarity Scores: {cosine_scores}")
#     print(f"GPU Memory Usage: {gpu_memory:.2f} MB")
#     print(f"Inference Time: {inference_time:.2f} sec/image")
#     print(f" Throughput: {throughput:.2f} img/sec")