from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os, json, csv, time, re
import pickle
from PIL import Image
from tqdm import tqdm
import pandas as pd
from local_API import generate_image
import traceback

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
    {"provider": "hf-inference", "model": "openfree/flux-chatgpt-ghibli-lora"}, # Base model: black-forest-labs/FLUX.1-dev
    {"provider": "hf-inference", "model": "stable-diffusion-v1-5/stable-diffusion-v1-5"}, # title={High-Resolution Image Synthesis With Latent Diffusion Models},
    {"provider": "fal-ai", "model": "stabilityai/stable-diffusion-3.5-large-turbo"}, # title={Scaling Rectified Flow Transformers for High-Resolution Image Synthesis}
    {"provider": "fal-ai", "model": "THUDM/CogView4-6B"}, # title={CogView3: Finer and Faster Text-to-Image Generation via Relay Diffusion}
    {"provider": "fal-ai", "model": "black-forest-labs/FLUX.1-dev"}, # title: n/a >> Medium.com={How does Flux work? The new image generation AI that rivals Midjourney}
    # {"provider": "hf-inference", "model": "PixArt-alpha/PixArt-XL-2-1024-MS"}, # title={PixArt-$α$: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis}
    {"provider": "fal-ai", "model": "playgroundai/playground-v2.5-1024px-aesthetic"}, # title={Playground v2.5: Three Insights towards Enhancing Aesthetic Quality in Text-to-Image Generation},
    # {"provider": "hf-inference", "model": "aipicasso/emi"}, # title={SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis}, 
    {"provider": "hf-inference", "model": "ali-vilab/In-Context-LoRA"}, # title={In-Context LoRA for Diffusion Transformers}, title2={Group Diffusion Transformers are Unsupervised Multitask Learners},
    {"provider": "fal-ai", "model": "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"}, # title={SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation}
    {"provider": "fal-ai", "model": "ByteDance/Hyper-SD"}, # title={Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis},
    {"provider": "fal-ai", "model": "Kwai-Kolors/Kolors"}, # title={Kolors: Effective Training of Diffusion Model for Photorealistic Text-to-Image Synthesis}
    {"provider": "fal-ai", "model": "Alpha-VLLM/Lumina-Image-2.0"}, # title={Lumina-Image 2.0: A Unified and Efficient Image Generative Framework},
    # {"provider": "hf-inference", "model": "etri-vilab/koala-700m-llava-cap"}, # title={KOALA: Self-Attention Matters in Knowledge Distillation of Latent Diffusion Models for Memory-Efficient and Fast Image Synthesis}, 
]


def sanitize_model_name(name):
    mapping = { # so that model names become [CogView, Flux, Ghibli, StableDiffusion]
        "openfree/flux-chatgpt-ghibli-lora": "Ghibli",
        "stable-diffusion-v1-5/stable-diffusion-v1-5": "StblDffsn_v15_local",
        "stabilityai/stable-diffusion-3.5-large-turbo": "StblDffsn_lrg",
        "THUDM/CogView4-6B": "CogView",
        "black-forest-labs/FLUX.1-dev": "Flux",
        "PixArt-alpha/PixArt-XL-2-1024-MS": "PixArt",
        "playgroundai/playground-v2.5-1024px-aesthetic":"playground",
        "aipicasso/emi": "emi",
        "ali-vilab/In-Context-LoRA": "Context_LoRA",
        "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers": "Sana_Sprint",
        "ByteDance/Hyper-SD": "Hyper_SD",
        "Kwai-Kolors/Kolors":"Kolors",
        "Alpha-VLLM/Lumina-Image-2.0": "Lumina",
        "etri-vilab/koala-700m-llava-cap":"koala",
    }
    if name in mapping:
        return mapping[name]
    # Fallback transformation for unknown models
    return re.sub(r'[^A-Za-z0-9]+', '_', name.split('/')[-1].split('-')[0])


def run_generation(model_name, prompt, image_path, sanitized_model, metadata_list, images, metadata_key, item, api_key, image_key):
    """Generate (locally or remotely) and save one image, record its metadata."""
    for attempt in range(3):
        # model_name= item["model"]
        try:
            if model_name in [
                "stable-diffusion-v1-5/stable-diffusion-v1-5",
                # "stabilityai/stable-diffusion-3.5-large-turbo"
                ]:
                # local_API 
                image = generate_image(prompt, model_name=model_name)
            else:
                # fallback to remote HF / Fal.ai inference
                client = InferenceClient(provider=item["provider"], api_key=api_key)
                # if model_name contains "ghibli" in the name, then prompt starts with "a ghibli style" + prompt
                prompt_updated = f"Ghibli Style {prompt}" if "ghibli" in model_name else prompt
                image = client.text_to_image(prompt_updated, model=model_name)
                
            # save
            image.save(image_path)
            images.setdefault(sanitized_model + metadata_key, []).append(image_path)
            # metadata
            metadata_list.append({
                "model": sanitized_model + metadata_key,
                "image_key": image_key,
                "prompt": prompt,
                "gen_img_path": image_path
            })
            # metadata_list.append(metadata_entry)

            # print and break
            print(f"Generated: {image_path} successfully")
            break # exit retry loop if successful
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {model_name} - {e}")
            traceback.print_exc()
            print(f"sleeping")
            time.sleep(4)
    else:
        print(f"Failed after retries: {item['model']}")

def record_existing(gen_img_path, model_key, prompt_text, sanitized_model, metadata_list, image, image_key):
    if os.path.exists(gen_img_path):
        print(f"Already exists : {gen_img_path}, skipping generation.")
        images.setdefault(model_key, []).append(gen_img_path)

        metadata_list.append({
            "model": sanitized_model,
            "image_key": image_key,
            "prompt": prompt_text,
            "gen_img_path": image_path
        })
        return True
    return False

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
        
        # model_key
        model_key = sanitized_model
        model_key_meta = f"{sanitized_model}_Metadata"

        # Check both paths first
        exists_main = record_existing(image_path, model_key, prompt, sanitized_model, metadata_list, images, image_key)
        exists_meta = record_existing(image_path_meta, model_key_meta, prompt_MetaData, sanitized_model + "_Metadata", metadata_list, images, image_key)
        
        # only skip if both exist
        if exists_main and exists_meta:
            print(f"Both images exist: {image_path} and {image_path_meta}, skipping generation.")
            continue
        
        if not exists_main:
            # Generate the main image
            run_generation(model_name=item["model"], 
                           prompt=prompt, 
                           image_path=image_path, 
                           sanitized_model=sanitized_model, 
                           metadata_list=metadata_list, 
                           images=images, 
                           metadata_key="",
                           item=item,
                           api_key=api_key,
                           image_key=image_key)
                

        if not exists_meta:
            # Generate the metadata image
            run_generation(model_name=item["model"], 
                prompt=prompt_MetaData, 
                image_path=image_path_meta, 
                sanitized_model=sanitized_model, 
                metadata_list=metadata_list, 
                images=images, 
                metadata_key="_Metadata",
                item=item,
                api_key=api_key,
                image_key=image_key)

# save MetaData to CSV for evaluation (including model, image_key, prompt, and generated image path)
df_metadata = pd.DataFrame(metadata_list) # metadata_list is generated during image generation loop
# df_metadata.to_csv("image_generated/gen_img_metadata.csv", index=False)
# print("Images generation complete. Metadata saved as 'image_generated/gen_img_metadata.csv'")
df_metadata.to_csv("image_generated/gen_img_metadata_2.csv", index=False)
print("Images generation complete. Metadata saved as 'image_generated/gen_img_metadata_2.csv'")

