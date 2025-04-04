from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os, json, csv, time, re
import pickle
# from evaluation_metrics import EvaluationMetrics
from PIL import Image
from tqdm import tqdm
import pandas as pd
from evaluation_metrics import(
    sanitize_prompt_key
)

# Ensure the 'models' and 'image_generated' directories exist
os.makedirs("image_generated", exist_ok=True)

# load json: DeepFashion/captions_sample.json
with open("DeepFashion/captions_sample.json", "r") as f:
    prompt_map = json.load(f)
    
# convert ".json" to ".csv" and save CSV
with open("DeepFashion/captions_sample.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_key", "prompt"])
    for image_key, prompt in prompt_map.items():
        writer.writerow([image_key, prompt])
        
# Load the CSV into a Dataframe
df_prompt_map = pd.read_csv("DeepFashion/captions_sample.csv")
# print(f"top 5 rows of DeepFashion/captions_sample.csv")
# print(df_prompt_map.head())

# convert df_prompt_map DataFrame to list of tuples for consistent ordering using "image_key" and "prompt" columns
prompt_items = list(
    df_prompt_map[
        ['image_key', 'prompt']
    ].itertuples(index=False, name=None)
)


# Load API from .env
load_dotenv()
api_key = os.getenv("HF_API_KEY")

# List of models and providers
models = [
    {"provider": "hf-inference", "model": "openfree/flux-chatgpt-ghibli-lora"},
    {"provider": "hf-inference", "model": "stable-diffusion-v1-5/stable-diffusion-v1-5"},
    {"provider": "fal-ai", "model": "THUDM/CogView4-6B"},
    {"provider": "fal-ai", "model": "black-forest-labs/FLUX.1-dev"},
]

# "THUDM/CogView4-6B" becomes "CogView4-6B"
# "black-forest-labs/FLUX.1-dev" becomes "FLUX.1-dev"
def sanitize_model_name(name):
    # remove anything before '/' and then replace all non-alphanumeric characters with underscores '_'
    return re.sub(r'[^A-Za-z0-9]+', '_', name.split('/')[-1]) 

images = {}
metadata_list = []
# Loop through models with tqdm progress bar
for item in tqdm(models, desc="Generating Images", unit="model"):
    for i, (image_key, prompt) in enumerate(prompt_items):
        sanitized_model = sanitize_model_name(item["model"])
        model_path = f"image_generated/{sanitized_model}"
        # ensure the directory exists upto the model name
        os.makedirs(model_path, exist_ok=True)
        sanitized_key = sanitize_prompt_key(image_key)
        image_path = f"image_generated/{sanitized_model}/{sanitized_key}.jpg"

        
        if os.path.exists(image_path):
            print(f"Already exists : {image_path}, skipping generation.")
            images.setdefault(sanitized_model, []).append(image_path)
            metadata_list.append({
                "model": sanitized_model,
                "image_key": image_key,
                "prompt": prompt,
                "gen_img_path": image_path
            })
            continue
        for attempt in range(3):
            try:
                client = InferenceClient(provider=item["provider"], api_key=api_key)
                image = client.text_to_image(prompt, model=item["model"])
                image.save(image_path)
                images.setdefault(sanitized_model, []).append(image_path)
                metadata_list.append({
                    "model": sanitized_model,
                    "image_key": image_key,
                    "prompt": prompt,
                    "gen_img_path": image_path
                })
                print(f"Generated: {image_path} successfully")
                break # exit retry loop if successful
            except Exception as e:
                print(f"Attempt {attempt+1} failed for {item['model']} - {e}")
                print(f"sleeping")
                time.sleep(4)
        else:
            print(f"Failed after retries: {item['model']}")

# save MetaData to CSV for evaluation (including model, image_key, prompt, and generated image path)
df_metadata = pd.DataFrame(metadata_list) # metadata_list is generated during image generation loop
df_metadata.to_csv("image_generated/gen_img_metadata.csv", index=False)
print("Images generation complete. Metadata saved as 'image_generated/gen_img_metadata.csv'")

