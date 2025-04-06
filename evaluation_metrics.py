import re
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import lpips
# Load the CLIP (model and processor), LPIPS model locally once
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
lpips_model = lpips.LPIPS(
    net="vgg",
    # weights=VGG16_Weights.IMAGENET1K_V1
    ).to("cuda")

# ----------------- Evaluation Functions -----------------

# "WOMEN-Tees_Tanks-id_00000112-06_7_additional.jpg"  >> becomes >> "WOMEN_Tees_Tanks_additional"
def sanitize_prompt_key(image_key_text):
    # remove the id part 
    # prompt_key = re.sub(r'-id_[^_]+_[^_]+', '', prompt_key_text)
    # remove the extension
    image_key = re.sub(r'\.jpg$', '', image_key_text)
    # replace all non-alphanumeric characters with underscores
    # prompt_key = re.sub(r'[^A-Za-z0-9]+', '_', prompt_key)
    return image_key

# ----------------- Evaluation Functions -----------------
def get_clip_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    return embedding.squeeze(0).cpu().numpy()

def calculate_clip_metrics(clip_embeddings_gt, clip_embeddings_gen, K=3):
    gt_keys = list(clip_embeddings_gt.keys())
    precision_at_k, recall_at_k, reciprocal_ranks = [], [], []
    for key in gt_keys:
        gen_vec = clip_embeddings_gen[key].reshape(1, -1)
        all_gt_vecs = np.array([clip_embeddings_gt[k] for k in gt_keys])
        sim_scores = cosine_similarity(gen_vec, all_gt_vecs).flatten()
        
        # sort all ground truth embeddings by their similarity to the generated image in descending order.
        ranked_indices = np.argsort(sim_scores)[::-1] 
        # find the position of the correct ground truth (using the same key that indexes the generated image embedding).
        correct_index = gt_keys.index(key)
        # select the top K most similar ground truth embeddings.
        top_k = ranked_indices[:K]
        # determine if the correct ground truth image is among those top K matches.
        hit = int(correct_index in top_k)
        
        precision_at_k.append(hit / K)
        recall_at_k.append(hit)
        rank = np.where(ranked_indices == correct_index)[0][0] + 1
        reciprocal_ranks.append(1 / rank)
    return {
        f"Precision@{K}": round(np.mean(precision_at_k), 3),
        f"Recall@{K}": round(np.mean(recall_at_k), 3),
        "MRR": round(np.mean(reciprocal_ranks), 3)
    }

def compute_clip_score(prompt, gen_img_path):
    image = Image.open(gen_img_path).convert("RGB")
    inputs = clip_processor(
        text=[prompt],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,      # Truncate text to avoid exceeding the model's maximum sequence length.
        max_length=77         # Set maximum length to 77 tokens, matching the model's expectation.
    ).to("cuda")
    with torch.no_grad():
        outputs = clip_model(**inputs)
    return outputs.logits_per_image[0][0].item()

def compute_lpips(gen_img_path, gt_img_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                            (0.5, 0.5, 0.5))
    ])
    gen_img = transform(Image.open(gen_img_path).convert("RGB")).unsqueeze(0).to("cuda")
    gt_img = transform(Image.open(gt_img_path).convert("RGB")).unsqueeze(0).to("cuda")
    with torch.no_grad():
        dist = lpips_model(gen_img, gt_img)
    return dist.item()
   
def compute_weighted_score(df):
    """
    Compute a weighted score for each model using min–max normalization.
    
    Expected DataFrame columns:
      - "Avg Clip Score ⬆️ [Prompt vs GenIm]" (higher is better)
      - "Avg Clip Cos Sim ⬆️ [GenImg vs GTimg]" (higher is better)
      - "Avg LPIPS ⬇️ [GenImg vs GTimg]" (lower is better)
      - "FID ⬇️ (Frechet inception distance)" (lower is better)
      - "MRR ⬆️ (Mean Reciprocal Rank)" (higher is better)
      - "Recall@3 ⬆️" (higher is better)
    
    The composite "Normalized Retrieval" is defined as the average of normalized MRR and Recall@3.
    
    Weighted_Score is defined as:
      0.4 × (Normalized CLIP Cosine) +
      0.3 × (Normalized LPIPS) +
      0.15 × (Normalized FID) +
      0.1 × (Normalized Retrieval) +
      0.05 × (Normalized CLIP Score)
    """
    def normalize(series, higher_better=True):
        min_val = series.min()
        max_val = series.max()
        # Avoid division by zero if all values are equal.
        if max_val == min_val:
            return series * 0 + 1  
        if higher_better:
            return (series - min_val) / (max_val - min_val)
        else:
            return (max_val - series) / (max_val - min_val)
    
    norm_clip_cos = normalize(df["Avg Clip Cos Sim ⬆️ [GenImg vs GTimg]"], higher_better=True)
    norm_lpips = normalize(df["Avg LPIPS ⬇️ [GenImg vs GTimg]"], higher_better=False)
    norm_fid = normalize(df["FID ⬇️ (Frechet inception distance)"], higher_better=False)
    norm_clip_score = normalize(df["Avg Clip Score ⬆️ [Prompt vs GenIm]"], higher_better=True)
    norm_mrr = normalize(df["MRR ⬆️ (Mean Reciprocal Rank)"], higher_better=True)
    norm_recall = normalize(df["Recall@3 ⬆️"], higher_better=True)
    
    # Composite retrieval: average of normalized MRR and Recall@3.
    norm_retrieval = (norm_mrr + norm_recall) / 2.0
    
    # Compute the weighted score using the given weights.
    weighted_score = (0.4 * norm_clip_cos +
                      0.3 * norm_lpips +
                      0.15 * norm_fid +
                      0.1 * norm_retrieval +
                      0.05 * norm_clip_score)
    
    return weighted_score
