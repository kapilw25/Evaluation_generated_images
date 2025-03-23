import torch
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
import os, csv
from evaluation_metrics import EvaluationMetrics
from PIL import Image
from torchvision import transforms

# CLear GPU cache to free memory
torch.cuda.empty_cache()

# Optimize PyTorch memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ensure the 'models' and 'image_generated' directories exist
os.makedirs("image_generated", exist_ok=True)

# shared prompt
prompts = ["A cat holding a sign that says hello world",
           "A futuristic city with flying cars",
           "An astronaut riding a horse in space"
           ]

# -----CV model1: Stable Diffusion v1.5---------------------------------------
pipe_sd_v1_5 = StableDiffusionPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5",
    torch_dtype= torch.float16,
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
        image = pipe(prompt).images[0] # PIL image
        image_path = f"image_generated/{model_name}_image_{i}.jpg"
        image.save(image_path)
        images.setdefault(model_name, []).append(image_path)
   
# ------------------------- Evaluation -------------------------
results = []  # to store evaluation metrics for CSV export

for model_name, img_paths in images.items():
    # Load images as PIL images (CLIPProcessor accepts PIL images directly)
    # generated_images = [Image.open(img).convert("RGB") for img  in img_paths]
    # for each mode, each image corresponds to the prompt with the same index
    for i, img_path in enumerate(img_paths):
        gen_img = Image.open(img_path).convert("RGB")
        gen_prompt = prompts[i]
    
        # Evaluation Metrics
        clip_score = round(EvaluationMetrics.calculate_clip_score([gen_img], gen_prompt)[0], 1)
        bleu_score = round(EvaluationMetrics.calculate_bleu_score(gen_prompt, [gen_prompt])[0], 1) # except 1.0 if identical
        cosine_score = round(EvaluationMetrics.calculate_cosine_similarity([gen_img], gen_prompt)[0], 1)
        inference_time = round(EvaluationMetrics.measure_inference_time(models[model_name], [gen_prompt]), 1)
        gpu_memory = round(EvaluationMetrics.measure_gpu_memory(), 1)
        throughput = round(EvaluationMetrics.measure_throughput(inference_time, 1), 1)

        print(f"Model: {model_name}, Image Index: {i}")
        print(f"Prompt: {gen_prompt}")
        print(f"CLIP score: {clip_score}")
        print(f"BLEU score: {bleu_score}")
        print(f" Cosine similarity Score: {cosine_score}")
        print(f"Inference Time: {inference_time} sec/image")
        print(f"GPU Memory Usage: {gpu_memory} MB")
        print(f"Throughput: {throughput} img/sec")
        
        # append results to a list for CSV export
        results.append({
            "Model": model_name,
            "Image_Index": i,
            "Prompt": gen_prompt,
            "CLIP score": clip_score,
            "BLEU score": bleu_score,
            "Cosine similarity score": cosine_score,
            "Inference Time (sec/image)": inference_time,
            "GPU Memory Usage (MB)": gpu_memory,
            "Throughput (img/sec)": throughput
        })
            
# write results into a CSV file
fieldnames = [
    "Model",
    "Image_Index",
    "Prompt",
    "CLIP score",
    "BLEU score",
    "Cosine similarity score",
    "Inference Time (sec/image)",
    "GPU Memory Usage (MB)",
    "Throughput (img/sec)"
]

with open("evaluation_results.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)
    