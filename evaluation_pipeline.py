import os, csv
# disable parallelism warnings for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd, numpy as np
import json
import mlflow
from sklearn.metrics.pairwise import cosine_similarity
from evaluation_metrics import(
    clip_model,
    clip_processor,
    lpips_model,
    get_clip_embedding,
    sanitize_prompt_key,
    calculate_clip_metrics,
    compute_clip_score,
    compute_lpips
)
from cleanfid import fid # Fréchet inception distance (FID)

# ----------------- Configuration -----------------
# Evaluation model name (scalable to other models later)
MODEL_NAMES = ["stable_diffusion_v1_5", "CogView4_6B", "FLUX_1_dev"]

# Paths for ground truth and generated images
GT_PATH = "DeepFashion/images"

 
# ----------------- Load MetaData -----------------
df_metadata = pd.read_csv("image_generated/gen_img_metadata.csv")
# add ground truth image path column based on the image_key from GT_PATH
GT_PATH = "DeepFashion/images"
df_metadata["gt_img_path"] = df_metadata["image_key"].apply(
    lambda x:os.path.join(GT_PATH, x)
)

# ----------------- Iterate over each Model -----------------

final_metrics_all = {}
with mlflow.start_run(run_name="Evaluation Run"):
    for model in MODEL_NAMES:
        # Filter metadata to evaluate only for current model
        df_eval = df_metadata[df_metadata["model"] == model].copy()
        
        # set generated images path for the current model
        GEN_PATH = f"image_generated/{model}"

        # ----------------- Collect CLIP Embeddings -----------------
        clip_embeddings_gt = {}
        clip_embeddings_gen = {}

        for idx, row in df_eval.iterrows():
            gt_img = row["image_key"]
            gt_img_path = row["gt_img_path"]
            gen_img_path = row["gen_img_path"]
            clip_embeddings_gt[gt_img] = get_clip_embedding(gt_img_path)
            clip_embeddings_gen[gt_img] = get_clip_embedding(gen_img_path)

        # ----------------- Metric Computation -----------------
        # compute average CLIPscore (Prompt vs Gen_Img)
        clip_scores = []
        for idx, row in df_eval.iterrows():
            score = compute_clip_score(
                row["prompt"],
                row["gen_img_path"])
            clip_scores.append(score)
        avg_clip_score = np.mean(clip_scores)

        # Compute average CLIP cosine (Gen_Img vs GT_img)
        clip_cos_sim = []
        for key in clip_embeddings_gt:
            cosine_val = cosine_similarity(
                clip_embeddings_gen[key].reshape(1, -1),
                clip_embeddings_gt[key].reshape(1, -1)
            )[0, 0]
            clip_cos_sim.append(cosine_val)
        avg_clip_cos_sim = np.mean(clip_cos_sim)

        # Compute average LPIPS (Gen_img vs GT_img)
        lpips_values = []
        for idx, row in df_eval.iterrows():
            lpips_values.append(compute_lpips(
                row["gen_img_path"],
                row["gt_img_path"]
            ))
        avg_lpips = np.mean(lpips_values)

        # Compute Fréchet inception distance (FID) using cleanfid
        Frechet_inception_distance_value = fid.compute_fid(GEN_PATH,
                                                            GT_PATH,
                                                            device="cuda")

        # Compute retrieval metrics (MRR, Recall@3) from CLIP embeddings
        retrieval_metrics = calculate_clip_metrics(clip_embeddings_gt, 
                                                    clip_embeddings_gen, 
                                                    K=3)

        # ----------------- Final Metrics Calculation -----------------
        metrics_model = {
            "Model": model,
            "Avg_Clip_Score_Prompt_GenImg": round(avg_clip_score, 2),
            "Avg_Clip_Cos_Sim_GenImg_GTimg": round(avg_clip_cos_sim, 2),
            "Avg_LPIPS_GenImg_GTimg": round(avg_lpips, 2),
            "FID_Frechet_inception_distance": round(Frechet_inception_distance_value, 2),
            "MRR_Mean_Reciprocal_Rank": round(retrieval_metrics["MRR"], 2),
            "Recall@3": round(retrieval_metrics["Recall@3"], 2)
        }
        
        # Log current model metrics to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_param("Model", model)
            mlflow.log_metrics({
                "Clip_Score_Prompt_GenImg": round(avg_clip_score, 2),
                "Clip_Cos_Sim_Gen_GTimg": round(avg_clip_cos_sim, 2),
                "LPIPS_Gen_GTimg": round(avg_lpips, 2),
                "FID": round(Frechet_inception_distance_value, 2),
                "MRR": round(retrieval_metrics["MRR"], 2),
                "Recall_at_3": round(retrieval_metrics["Recall@3"], 2)
            }, step=0)
        
        final_metrics_all[model] = metrics_model
        print(f"Evaluation Metrics for model {model}:")
        print(json.dumps(str(metrics_model)))
    
print("All Evaluation Metrics:")
print(json.dumps(str(final_metrics_all)))

df_results = pd.DataFrame.from_dict(final_metrics_all, orient='index').reset_index(drop=True)
# Ensure "Model" is the first column
cols = df_results.columns.tolist()
if "Model" in cols:
    cols.insert(0, cols.pop(cols.index("Model")))
    df_results = df_results[cols]
df_results.to_csv("results/evaluation_results.csv", index=False)
mlflow.log_artifact("results/evaluation_results.csv")
print(f"Saved Evaluation results at 'results/evaluation_results.csv'")
