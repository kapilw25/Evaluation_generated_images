from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os, json, csv, time, re
import pickle
from PIL import Image
from tqdm import tqdm
import pandas as pd

# Ensure the 'image_generated' directories exist
os.makedirs("image_generated", exist_ok=True)
        
# Load the CSV into a Dataframe
df_prompt_map = pd.read_csv("DeepFashion/captions_sample.csv")

# convert df_prompt_map DataFrame to list of tuples for consistent ordering using "image_key" and "prompt" columns
prompt_items = list(
    df_prompt_map[
        ['image_key', 'prompt', 'prompt_MetaData']
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


def sanitize_model_name(name):
    mapping = { # so that model names become [CogView, Flux, Ghibli, StableDiffusion]
        "openfree/flux-chatgpt-ghibli-lora": "Ghibli",
        "stable-diffusion-v1-5/stable-diffusion-v1-5": "StableDiffusion",
        "THUDM/CogView4-6B": "CogView",
        "black-forest-labs/FLUX.1-dev": "Flux"
    }
    if name in mapping:
        return mapping[name]
    # Fallback transformation for unknown models
    return re.sub(r'[^A-Za-z0-9]+', '_', name.split('/')[-1].split('-')[0])

images = {}
metadata_list = []
# Loop through models with tqdm progress bar
for item in tqdm(models, desc="Generating Images", unit="model"):
    for i, (image_key, prompt, prompt_MetaData) in enumerate(prompt_items):
        sanitized_model = sanitize_model_name(item["model"])
        model_path = f"image_generated/{sanitized_model}"
        model_path_meta = f"image_generated/{sanitized_model}_Metadata"
        # ensure the directories exist
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(model_path_meta, exist_ok=True)
        
        # For primary prompt image
        image_path = f"{model_path}/{image_key}"
        image_path_meta = f"{model_path_meta}/{image_key}"

        # for "image_path"
        # Check if the image already exists
        if os.path.exists(image_path):
            print(f"Already exists : {image_path}, skipping generation.")
            images.setdefault(sanitized_model, []).append(image_path)

            metadata_entry = {
                "model": sanitized_model,
                "image_key": image_key,
                "prompt": prompt,
                "gen_img_path": image_path
            }
            metadata_list.append(metadata_entry)
            
        else:
            for attempt in range(3):
                try:
                    client = InferenceClient(provider=item["provider"], api_key=api_key)
                    model=item["model"]
                    # if model contains "ghibli" in the name, then prompt starts with "a ghibli style" + prompt
                    if "ghibli" in model:
                        prompt_updated = f"Ghibli Style {prompt}"
                    else:
                        prompt_updated = prompt
                    image = client.text_to_image(prompt_updated, model=model)
                    image.save(image_path)
                    images.setdefault(sanitized_model, []).append(image_path)
                    
                    metadata_entry = {
                        "model": sanitized_model,
                        "image_key": image_key,
                        "prompt": prompt,
                        "gen_img_path": image_path
                    }
                    metadata_list.append(metadata_entry)

                    print(f"Generated: {image_path} successfully")
                    break # exit retry loop if successful
                except Exception as e:
                    print(f"Attempt {attempt+1} failed for {item['model']} - {e}")
                    print(f"sleeping")
                    time.sleep(4)
            else:
                print(f"Failed after retries: {item['model']}")
                
        # for "image_path_meta"
        if os.path.exists(image_path_meta):
            print(f"Already exists : {image_path_meta}, skipping generation.")
            images.setdefault(sanitized_model, []).append(image_path_meta)

            metadata_entry = {
                "model": f"{sanitized_model}_Metadata",
                "image_key": image_key,
                "prompt": prompt_MetaData,
                "gen_img_path": image_path_meta
            }
            metadata_list.append(metadata_entry)
        else:
            for attempt in range(3):
                try:
                    client = InferenceClient(provider=item["provider"], api_key=api_key)
                    model = item["model"]
                    if "ghibli" in model:
                        prompt_updated = f"Ghibli Style {prompt_MetaData}"
                    else:
                        prompt_updated = prompt_MetaData
                    image = client.text_to_image(prompt_updated, model=model)
                    image.save(image_path_meta)
                    images.setdefault(sanitized_model, []).append(image_path_meta)
                    
                    metadata_entry = {
                        "model": f"{sanitized_model}_Metadata",
                        "image_key": image_key,
                        "prompt": prompt_MetaData,
                        "gen_img_path": image_path_meta
                    }
                    metadata_list.append(metadata_entry)
                    
                    print(f"Generated: {image_path_meta} successfully")
                    break
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

