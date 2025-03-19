import torch
import time
import gc, os
import psutil
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu

# Ensure 'models/' directory exists for local caching
os.makedirs("models", exist_ok=True)
# Customer cache path for downloaded models
os.environ["HF_HOME"] = os.path.abspath("models")

# ================================
# Download or Load from Local Directory
# ================================
clip_model_path = "models/clip-vit-base-patch32"
clip_processor_path = "models/clip-vit-base-patch32-processor"
sbert_model_path = "models/all-MiniLM-L6-v2"

# Load CLIP model
if not os.path.exists(clip_model_path):
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        cache_dir=clip_model_path)
else:
    clip_model = CLIPModel.from_pretrained(clip_model_path)
    
# Load CLIP Processor
if not os.path.exists(clip_processor_path):
    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32",
        cache_dir=clip_processor_path)
else:
    clip_model = CLIPModel.from_pretrained(clip_processor_path)
    
# Load SentenceTransformer Model
if not os.path.exists(sbert_model_path):
    sbert_model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2',
        cache_folder=sbert_model_path)
else:
    sbert_model = CLIPModel.from_pretrained(sbert_model_path)


# ================================
# TEXT-TO-IMAGE CONSISTENCY METRICS
# ================================

class EvaluationMetrics:
    
    @staticmethod
    def calculate_clip_score(images, prompt):
        """Batch Calculation for CLIP score."""
        inputs = clip_processor(
            text=[prompt]*len(images),
            images=images,
            return_tensors="pt",
            padding=True
        )
        outputs=  clip_model(**inputs)
        return outputs.logits_per_image[:,0].tolist()
    
    @staticmethod
    def calculate_bleu_score(reference_text, generated_texts):
        """Batch Calculation for BLEU score"""
        reference = [reference_text.lower().split()]
        return [
            sentence_bleu(reference, text.lower().split()) \
                for text in generated_texts]
        
    @staticmethod
    def calculate_cosine_similarity(image, prompt):
        """Batch Calculation for cosine similarity"""
        prompt_embedding = sbert_model.encode(prompt)
        return [
            float(util.pytorch_cos_sim(sbert_model.encode(img.resize((224,224))),
                                       prompt_embedding).item())
            for img in images
        ]
        
    @staticmethod
    def measure_inference_time(model, prompts, height=512, width=512):
        """Measures inference time in batch mode"""
        torch.cuda.synchronize()
        start_time = time.time()
        for prompt in prompts:
            model(prompt, height=height, width=width)
        torch.cuda.synchronize()
        return (time.time - start_time) / len(prompts)
        
    @staticmethod
    def measure_gpu_memory():
        """Tracks peak GPU memory usage (MB)"""
        return torch.cuda.max_memory_allocated() / (1024*1024)
    
    @staticmethod
    def measure_throughput(total_time, total_images):
        """Calculate throughput as images per second"""
        return total_images / total_time if total_time > 0 else 0