from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os, csv, time, re
from evaluation_metrics import EvaluationMetrics
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

# CLear GPU cache to free memory
torch.cuda.empty_cache()

# Optimize PyTorch memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ensure the 'models' and 'image_generated' directories exist
os.makedirs("image_generated", exist_ok=True)

# shared prompt
prompt_map = {
    "MEN-Denim-id_00000089-28_1_front.jpg": 
        "This gentleman is wearing a medium-sleeve shirt with pure color patterns. The shirt is with cotton fabric and its neckline is lapel. This gentleman wears a long trousers. The trousers are with cotton fabric and solid color patterns.",
    "MEN-Jackets_Vests-id_00000084-08_2_side.jpg": 
        "The shirt this gentleman wears has long sleeves, its fabric is mixed, and it has mixed patterns. The shirt has a crew neckline. The outer clothing this person wears is with cotton fabric and color block patterns.",
    "MEN-Jackets_Vests-id_00003336-09_4_full.jpg": 
        "The sweater this man wears has long sleeves and it is with cotton fabric and solid color patterns. The neckline of the sweater is round. This man wears a long trousers, with cotton fabric and solid color patterns. The outer clothing this person wears is with cotton fabric and solid color patterns.",
    "WOMEN-Sweatshirts_Hoodies-id_00000095-01_1_front.jpg": 
        "This lady wears a medium-sleeve shirt with graphic patterns and a three-point shorts. The shirt is with cotton fabric and its neckline is round. The shorts are with leather fabric and pure color patterns. There is an accessory in his her neck. The female is wearing a ring on her finger. The female is wearing a hat.",
    "WOMEN-Tees_Tanks-id_00000112-06_7_additional.jpg": 
        "Her tank shirt has sleeves cut off, cotton fabric and solid color patterns. It has a suspenders neckline. The pants the female wears is of three-point length. The pants are with denim fabric and pure color patterns. The lady has a hat in her head. The female wears a ring. There is an accessory on her wrist. This lady has neckwear.",
   }

# convert prompt_map to list of tuples for consistent ordering
prompt_items = list(prompt_map.items())

# Load API from .env
load_dotenv()
api_key = os.getenv("HF_API_KEY")

# List of models and providers
models = [
    {"provider": "hf-inference", "model": "stable-diffusion-v1-5/stable-diffusion-v1-5"},
    {"provider": "fal-ai", "model": "THUDM/CogView4-6B"},
    {"provider": "fal-ai", "model": "black-forest-labs/FLUX.1-dev"},
]

def sanitize_filename(name):
    # replace all non-alphanumeric characters with underscores
    return re.sub(r'[^A-Za-z0-9]+', '_', name)

images = {}
# Loop through models with tqdm progress bar
for item in tqdm(models, desc="Generating Images", unit="model"):
    for i, (image_key, prompt) in enumerate(prompt_items):
        sanitized_model = sanitize_filename(item["model"])
        sanitized_key = sanitize_filename(image_key)
        image_path = f"image_generated/{sanitized_model}_{sanitized_key}.jpg"
        if os.path.exists(image_path):
            print(f"Already exists : {sanitized_model}_{sanitized_key}.jpg, skipping generation.")
            images.setdefault(sanitized_model, []).append(image_path)
            continue
        for attempt in range(3):
            try:
                client = InferenceClient(provider=item["provider"], api_key=api_key)
                image = client.text_to_image(prompt, model=item["model"])
                image.save(image_path)
                images.setdefault(sanitized_model, []).append(image_path)
                print(f"Generated: {sanitized_model}_{sanitized_key}.jpg successfully")
                break # exit retry loop if successful
            except Exception as e:
                print(f"Attempt {attempt+1} failed for {item['model']} - {e}")
                print(f"sleeping")
                time.sleep(4)
        else:
            print(f"Failed after retries: {item['model']}")
   
# ------------------------- Evaluation -------------------------
results = []  # to store evaluation metrics for CSV export

for model_name, img_paths in images.items():
    # for each model, each image corresponds to the prompt from prompt_items (by order)
    for i, img_path in enumerate(img_paths):
        gen_img = Image.open(img_path).convert("RGB")
        if i < len(prompt_items):
            gen_prompt = prompt_items[i]
        else:
            gen_prompt = "No prompt available"
    
        # Evaluation Metrics
        clip_score = round(EvaluationMetrics.calculate_clip_score([gen_img], gen_prompt)[0], 1)
        cosine_score = round(EvaluationMetrics.calculate_cosine_similarity([gen_img], gen_prompt)[0], 1)

        print(f"Model: {model_name}, Image Index: {i}")
        print(f"Prompt: {gen_prompt}")
        print(f"CLIP score: {clip_score}")
        print(f" Cosine similarity Score: {cosine_score}")
        
        # append results to a list for CSV export
        results.append({
            "Model": model_name,
            "Image_Index": i,
            "Prompt": gen_prompt,
            "CLIP score": clip_score,
            "Cosine similarity score": cosine_score,
            "Filename": os.path.basename(img_path)
        })
            
# Write per-model evaluation results into a CSV file
fieldnames = [
    "Model",
    "Image_Index",
    "Prompt",
    "CLIP score",
    "Cosine similarity score",
    "Filename",
]
with open("evaluation_results.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

# -----------------------------------------------------
# Now, calculate retrieval metrics (Precision@K, Recall@K, MRR)
# for each prompt (across all models)
retrieval_results = []
# Iterate over each prompt in prompt_items (list of tuples: (ground_truth_filename, prompt))
for gt_filename, prompt in prompt_items:
    gt_path = os.path.join("DeepFashion/images", gt_filename)
    if not os.path.exists(gt_path):
        print(f"Ground truth image not found: {gt_filename}")
        continue
    gt_img = Image.open(gt_path).convert("RGB")
    
    candidate_imgs = []
    # For each model, if an image for this prompt exists, add it as candidate
    for model_key, paths in images.items():
        if len(paths) > 0:
            # Assume the order of images corresponds to the order of prompt_items
            index = prompt_items.index((gt_filename, prompt))
            if index < len(paths):
                candidate_imgs.append(Image.open(paths[index]).convert("RGB"))
    if not candidate_imgs:
        continue
    
    metrics = EvaluationMetrics.compute_precision_recall_mrr(gt_img, candidate_imgs, k=3)
    retrieval_results.append({
        "Metric_Type": "Retrieval",
        "Prompt": prompt,
        "Precision@K": round(metrics["precision@k"], 3),
        "Recall@K": round(metrics["recall@k"], 3),
        "MRR": round(metrics["mrr"], 3),
        "FID": ""
    })

# -----------------------------------------------------
# Now, calculate FID score across all real images vs. generated images.
real_images = []
real_dir = "DeepFashion/images"
for file in os.listdir(real_dir):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        real_images.append(Image.open(os.path.join(real_dir, file)).convert("RGB"))

gen_images = []
gen_dir = "image_generated"
for file in os.listdir(gen_dir):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        gen_images.append(Image.open(os.path.join(gen_dir, file)).convert("RGB"))

fid_value = EvaluationMetrics.calculate_fid(real_images, gen_images)
fid_result = {
    "Metric_Type": "FID",
    "Prompt": "",
    "Precision@K": "",
    "Recall@K": "",
    "MRR": "",
    "FID": round(fid_value, 3)
}

# -----------------------------------------------------
# Combine retrieval results and FID result, and write to a new CSV file
combined_fieldnames = [
    "Metric_Type",
    "Prompt",
    "Precision@K",
    "Recall@K",
    "MRR",
    "FID"
]

with open("evaluation_retrieval_fid_results.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=combined_fieldnames)
    writer.writeheader()
    for row in retrieval_results:
        writer.writerow(row)
    writer.writerow(fid_result)

    