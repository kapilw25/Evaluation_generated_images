import torch
import time
import gc, os
import psutil
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load models 
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",)

print(f" CUDA Available: {torch.cuda.is_available()}")
print(f" CUDA Device: {torch.cuda.get_device_name(0)}")


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
            padding=True,
            do_rescale= False # fix to prevent double rescaling
        ).to("cuda")
        # print(f" Tensor Device: {inputs.device}")
        outputs=  clip_model(**inputs.to("cuda"))
        return outputs.logits_per_image[:,0].tolist()
    
    @staticmethod
    def calculate_bleu_score(reference_text, generated_texts):
        """Batch Calculation for BLEU score"""
        reference = [reference_text.lower().split()]
        smooth_fn = SmoothingFunction().method1
        return [
            sentence_bleu(reference,
                        text.lower().split(),
                        smoothing_function = smooth_fn)
                for text in generated_texts]
        
    @staticmethod
    def calculate_cosine_similarity(images, prompt):
        """Batch Calculation for cosine similarity -- Using CLIP only"""

        prompt_inputs = clip_processor(text=prompt, return_tensors="pt").to("cuda")
        prompt_embedding = clip_model.get_text_features(**prompt_inputs)
        print(f" Tensor Device: {prompt_embedding.device}")
        
        cosine_scores = []
        for img in images:
            # encode image using CLIP
            image_inputs = clip_processor(images=img, return_tensors="pt").to("cuda")
            image_embedding = clip_model.get_image_features(**image_inputs)
            
            # Normalize embeddings for cosine similarity
            prompt_embedding = prompt_embedding.to("cuda") / prompt_embedding.norm()
            image_embedding = image_embedding.to("cuda") / image_embedding.norm()
            
            # Calculate Cosine Similarity
            score = torch.nn.functional.cosine_similarity(
                prompt_embedding,
                image_embedding,
                dim =-1 # compute cosine similarity along the last dimension
            ).mean().item()
            
            cosine_scores.append(score)
        
        return cosine_scores
            
        
    @staticmethod
    def measure_inference_time(model, prompts, height=512, width=512):
        """Measures inference time in batch mode"""
        torch.cuda.synchronize()
        start_time = time.time()
        for prompt in prompts:
            model(prompt, height=height, width=width)
        torch.cuda.synchronize()
        return (time.time() - start_time) / len(prompts)
        
    @staticmethod
    def measure_gpu_memory():
        """Tracks peak GPU memory usage (MB)"""
        return torch.cuda.max_memory_allocated() / (1024*1024)
    
    @staticmethod
    def measure_throughput(total_time, total_images):
        """Calculate throughput as images per second"""
        return total_images / total_time if total_time > 0 else 0