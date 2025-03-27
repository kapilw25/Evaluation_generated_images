# evaluation_metrics.py (updated version)

import torch
import torch.nn.functional as F
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from PIL import Image
from torchvision.models import inception_v3
from scipy import linalg

# Global: load CLIP model & processor only once
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print(f" CUDA Available: {torch.cuda.is_available()}")
print(f" CUDA Device: {torch.cuda.get_device_name(0)}")

class EvaluationMetrics:

    @staticmethod
    def calculate_clip_score(images, prompt):
        """
        Batch Calculation for CLIP score:
        Returns a list of logits_per_image for each image wrt the given prompt.
        """
        inputs = clip_processor(
            text=[prompt]*len(images),
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            do_rescale=False
        ).to("cuda")

        outputs = clip_model(**inputs.to("cuda"))
        return outputs.logits_per_image[:, 0].tolist()

    @staticmethod
    def calculate_cosine_similarity(images, prompt):
        """
        Batch Calculation for cosine similarity using CLIP embeddings:
        Returns a list of cosine similarities for each image wrt the given prompt.
        """
        # Encode prompt
        prompt_inputs = clip_processor(
            text=prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to("cuda")

        with torch.no_grad():
            prompt_embedding = clip_model.get_text_features(**prompt_inputs)
            prompt_embedding = prompt_embedding / prompt_embedding.norm(dim=-1, keepdim=True)

        cos_scores = []
        for img in images:
            with torch.no_grad():
                image_inputs = clip_processor(images=img, return_tensors="pt").to("cuda")
                image_embedding = clip_model.get_image_features(**image_inputs)
                image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

            score = F.cosine_similarity(prompt_embedding, image_embedding, dim=-1).mean().item()
            cos_scores.append(score)

        return cos_scores
    
    @staticmethod
    def get_clip_image_embedding(image: Image.Image):
        """
        Converts a PIL image into a normalized CLIP embedding (512-d).
        """
        inputs = clip_processor(images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)
        # Normalize to unit vector
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    @staticmethod
    def compute_precision_recall_mrr(gt_image: Image.Image, generated_images: list, k: int = 3):
        """
        Computes Precision@K, Recall@K, and MRR using CLIP-based similarity.
        Assumes exactly ONE relevant image among the candidates.
        Returns a dict with keys: 'precision@k', 'recall@k', 'mrr'.
        """
        # 1) Embeddings
        gt_emb = EvaluationMetrics.get_clip_image_embedding(gt_image)
        cand_embs = [EvaluationMetrics.get_clip_image_embedding(img) for img in generated_images]

        # 2) Similarities
        sims = []
        for emb in cand_embs:
            score = F.cosine_similarity(gt_emb, emb, dim=-1).item()
            sims.append(score)

        # 3) Rank and find correct_index (highest sim)
        correct_index = int(np.argmax(sims))
        ranked_indices = np.argsort(sims)[::-1]  # descending order

        # 4) Precision@K
        relevant_count = sum(1 for idx in ranked_indices[:k] if idx == correct_index)
        precision_at_k = relevant_count / k

        # 5) Recall@K
        # If there's only 1 relevant item, recall@K = 1 if correct item is in top K, else 0
        recall_at_k = 1.0 if correct_index in ranked_indices[:k] else 0.0

        # 6) MRR
        rank_of_correct = np.where(ranked_indices == correct_index)[0][0] + 1  # 1-based
        mrr = 1.0 / rank_of_correct

        return {
            "precision@k": precision_at_k,
            "recall@k": recall_at_k,
            "mrr": mrr
        }

    # --------------------- FID Computation Methods ---------------------
    @staticmethod
    def preprocess_for_inception(image: Image.Image, size=299):
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])
        return transform(image)

    @staticmethod
    def get_inception_activations(images: list):
        """
        Returns activations of the Inception v3 final layer (approx) for a list of PIL images.
        Used to compute FID. For a real FID pipeline, consider a specialized library.
        """
        model = inception_v3(pretrained=True, transform_input=False).eval().to("cuda")
        model.aux_logits = False  # Turn off AuxLogits

        activations = []
        for img in images:
            x = EvaluationMetrics.preprocess_for_inception(img).unsqueeze(0).to("cuda")
            with torch.no_grad():
                features = model(x)
                act = features.cpu().numpy()  # shape [1, 1000]
            activations.append(act)
        activations = np.concatenate(activations, axis=0)  # [num_images, 1000]
        return activations

    @staticmethod
    def calculate_fid(real_images: list, gen_images: list):
        """
        Computes the Frechet Inception Distance (FID) between two sets of images.
        Typically, you need many real vs. generated images for stable FID.
        """
        real_acts = EvaluationMetrics.get_inception_activations(real_images)
        gen_acts = EvaluationMetrics.get_inception_activations(gen_images)

        mu1, sigma1 = real_acts.mean(axis=0), np.cov(real_acts, rowvar=False)
        mu2, sigma2 = gen_acts.mean(axis=0), np.cov(gen_acts, rowvar=False)
        diff = mu1 - mu2

        # sqrtm of product of covariances
        from scipy import linalg
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
