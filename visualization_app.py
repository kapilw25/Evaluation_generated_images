import streamlit.components.v1 as components
import pandas as pd, matplotlib.pyplot as plt, numpy as np
import subprocess
import time
import os
import base64
from io import BytesIO
from PIL import Image
import streamlit as st
import seaborn as sns
from pandas.plotting import parallel_coordinates

st.set_page_config(layout="wide")

st.title("Multimodal Benchmarking and Recommendation of Text-to-Image Generation Models")

# Load CSVs
gen_img_metadata = "image_generated/gen_img_metadata.csv"
df_gen_img_metadata = pd.read_csv(gen_img_metadata)
# Generated image metadata column names: ['model', 'image_key', 'prompt', 'gen_img_path']
df_evaluation_results = pd.read_csv("results/evaluation_results.csv")

image_dir = "image_generated"
  
ground_truth_images = "DeepFashion/images"

# check is CSV and images directory exist
if not os.path.exists(gen_img_metadata):
    st.error(f"{gen_img_metadata} not found.")
    st.stop()
    
# Ensure that "image_generated" folder exists
if not os.path.exists(image_dir):
    st.error(f"{image_dir} folder not found")
    st.stop()
    
# 2) Inject CSS for a no-wrap, horizontal scroll strip with per-model vertical stacking
st.markdown(
    """
    <style>
    .scrolling-wrapper {
      display: flex;
      flex-wrap: nowrap;
      overflow-x: auto;
      overflow-y: hidden;
      height: 100vh;             /* fill viewport height */
      align-items: flex-start;   /* align images to top */
      padding: 8px 0;
    }
    .scrolling-wrapper > .model {
      display: flex;
      flex-direction: column;    /* stack Base above Metadata */
      align-items: center;
      flex: none;
      margin-right: 24px;
    }
    .scrolling-wrapper img {
      width: 150px;
      height: auto;
      margin-bottom: 4px;
      display: block;
    }
    .scrolling-wrapper .caption {
      font-size: 0.8em;
      margin-bottom: 8px;
      text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def img_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
  
# -----------------------------
# 1. Load and Prepare Data
# -----------------------------
@st.cache_data
def load_and_prepare_data(url):
    df = pd.read_csv(url)
    metrics = [
        "Weighted Score ⬆️",
        "Avg Clip Cos Sim ⬆️ [GenImg vs GTimg]",
        "Avg LPIPS ⬇️ [GenImg vs GTimg]",
        "FID ⬇️ (Frechet inception distance)",
        "MRR ⬆️ (Mean Reciprocal Rank)",
        "Recall@3 ⬆️",
        "Avg Clip Score ⬆️ [Prompt vs GenIm]"
    ]

    # filter out any “Metadata” models
    all_models = [m for m in df['Model'].unique() if "Metadata" not in m]
    df_filtered = df[df['Model'].isin(all_models)].reset_index(drop=True)

    # normalize metrics to [0,1]
    df_norm = df_filtered.copy()
    for metric in metrics:
        col = df_filtered[metric]
        if '⬆️' in metric:
            df_norm[metric] = (col - col.min()) / (col.max() - col.min())
        else:
            df_norm[metric] = (col.max() - col) / (col.max() - col.min())

    return df_filtered, df_norm, metrics

# -----------------------------
# 2. Plot Functions
# -----------------------------
def plot_radar(df_norm, metrics, models):
    labels = metrics
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6,6))
    for m in models:
        vals = df_norm[df_norm['Model']==m][metrics].iloc[0].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, label=m)
        ax.fill(angles, vals, alpha=0.1)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([0.2,0.5,0.8])
    ax.set_ylim(0,1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.1))
    st.pyplot(fig)

def plot_parallel(df_norm, df_filt, metrics, top_n):
    top = df_filt.sort_values("Weighted Score ⬆️", ascending=False)['Model'].head(top_n)
    data = df_norm[df_norm['Model'].isin(top)]
    fig, ax = plt.subplots(figsize=(8,4))
    parallel_coordinates(data[['Model']+metrics], 'Model', colormap=plt.get_cmap('Set2'), ax=ax)
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(fig)

def plot_heatmap(df_norm, df_filt, metrics, top_n):
    top = df_filt.sort_values("Weighted Score ⬆️", ascending=False)['Model'].head(top_n)
    data = df_norm[df_norm['Model'].isin(top)].set_index('Model')[metrics].T
    fig, ax = plt.subplots(figsize=(8,4))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_scatter(df_norm, df_filt, x_metric, y_metric, top_n):
    top = df_filt.sort_values("Weighted Score ⬆️", ascending=False)['Model'].head(top_n)
    data = df_norm[df_norm['Model'].isin(top)]
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=data, x=x_metric, y=y_metric, hue='Model', s=100, ax=ax)
    for i,row in data.iterrows():
        ax.text(row[x_metric]+0.01, row[y_metric]+0.01, row['Model'], fontsize=8)
    st.pyplot(fig)

# create tabs:
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Compare Images",
    "Evaluation",
    "System Design",
    "Model Architectures",
    "Disclaimer"
])

# --------------------------------------------------- Tab: Compare Images--------------------------------------------------- 
with tab1:        
    # Extract Unique prompts from the CSV
    prompts = df_gen_img_metadata["prompt"].unique()

    # dropdown to select which prompt to display
    selected_prompt = st.selectbox("Select a prompt:", prompts)
    
    # map the selected prompt to its corresponding image key
    selected_image_key = df_gen_img_metadata[df_gen_img_metadata["prompt"] == selected_prompt]["image_key"].iloc[0]
    prompt_text = selected_prompt
    
    # Filter Dataframe by the chosen image key so both model and model_metadata rows are included
    filtered_df = df_gen_img_metadata[df_gen_img_metadata["image_key"] == selected_image_key]
    if filtered_df.empty:
      st.error("No data found for the selected image key")
      st.stop()
    
    # show ground-truth image
    gt_path = os.path.join(ground_truth_images, selected_image_key)
    
    # Filter DataFrame by selected image key to get all model variants
    filtered_df = df_gen_img_metadata[df_gen_img_metadata["image_key"] == selected_image_key]

    # For each model in the filtered DataFrame, vertical display the image + metrics
    models_for_prompt = filtered_df["model"].unique()
    # Sort models by their weighted score
    model_scores = df_evaluation_results[df_evaluation_results["Model"].isin(models_for_prompt)][["Model", "Weighted Score ⬆️"]]
    model_scores = model_scores.sort_values("Weighted Score ⬆️", ascending=False)
    sorted_models = model_scores["Model"].tolist()

    # Resize function to unify image sizes
    def resize_image(img, size=(768, 1024)):
      return img.resize(size)
    
    # Retrieve and resize ground truth image once
    if os.path.exists(gt_path):
      gt_img = Image.open(gt_path).convert("RGB")
      gt_img = resize_image(gt_img)
    else:
      st.warning(f"Ground Truth image not found: {selected_image_key}")
      
    # Ground model images into pairs by base model name (e.g - "Flux", "CogView", etc.)
    ordered_bases = []
    base_model_map = {}
    for model in sorted_models:
      base = model.replace('_Metadata', '')
      if base not in ordered_bases:
        ordered_bases.append(base)
      if base not in base_model_map:
        base_model_map[base] = {}
      if model.endswith("_Metadata"):
        base_model_map[base]['metadata'] = model
      else:
        base_model_map[base]['base'] = model
      
    # create columns: Ground Truth + 1 column per base model
    cols = st.columns(1 + len(ordered_bases))
    
    # First column: Ground Truth
    with cols[0]:
      st.markdown("**Ground Truth**")
      st.image(gt_img, width=150)
      
    # For each model: base + metadata inside one column
    for idx, base in enumerate(ordered_bases):
      with cols[idx + 1]:
        st.markdown(f"**{base}**")
        
        # Display base image
        if 'base' in base_model_map[base]:
          base_row = filtered_df[filtered_df["model"] == base_model_map[base]['base']]
          if not base_row.empty:
            base_img = Image.open(base_row.iloc[0]["gen_img_path"]).convert("RGB")
            st.image(base_img, width=150, caption="Base")
            
        # Display metadata image
        if 'metadata' in base_model_map[base]:
          meta_row = filtered_df[filtered_df["model"] == base_model_map[base]['metadata']]
          if not meta_row.empty:
            meta_img = Image.open(meta_row.iloc[0]["gen_img_path"]).convert("RGB")
            st.image(meta_img, width=150, caption="Metadata")
      
    st.write("---")
    st.markdown("**End of Comparison**")

# --------------------------------------------------- Tab: "Evaluation"  --------------------------------------------------- 
with tab2:

    # create a selectbox to choose the metric for the grouped bar chart (excluding the "Model" column)
    metric_cols = [col for col in df_evaluation_results.columns if col != "Model"]
    selected_metric = st.selectbox("Select a Metric:", metric_cols)
    
    # seperate base models (without "_Metadata") and metadata models (ending with "_Metadata")
    df_base = df_evaluation_results[~df_evaluation_results["Model"].str.endswith("_Metadata")].copy()
    df_meta = df_evaluation_results[df_evaluation_results["Model"].str.endswith("_Metadata")].copy()
    
    # create a common Basename column so e.g - "Flux_Metadata" becomes "Flux"
    df_base["BaseName"] = df_base["Model"]
    df_meta["BaseName"] = df_meta["Model"].str.replace("_Metadata", "")
    
    # Merge metadata rows into base rows on BaseName, so we have corresponding columns side-by-side
    merged = pd.merge(df_base, df_meta, on="BaseName", suffixes=("_base", "_meta"))
    
    # create subplots for each selected metric
    fig, ax = plt.subplots(figsize=(8, 3.5))
      
    x = np.arange(len(merged))
    width = 0.3
    
    # plot bars
    ax.bar(x - width/2, merged[f"{selected_metric}_base"], width, label="Base")
    ax.bar(x + width/2, merged[f"{selected_metric}_meta"], width, label="Metadata")
    
    # configure the x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(merged["BaseName"], rotation=45)
    ax.set_ylabel(selected_metric)
    ax.set_title(f"Base vs. Metadata: {selected_metric}")
    ax.legend()
  
    # dynamically adjust y-axis
    base_vals = merged[f"{selected_metric}_base"]
    meta_vals = merged[f"{selected_metric}_meta"]
    y_min = min(min(base_vals), min(meta_vals))
    y_max = max(max(base_vals), max(meta_vals))
    
    # define delta as 5% of data range
    delta = 0.05 * (y_max - y_min) if (y_max - y_min) != 0 else 0.1
    
    ax.set_ylim([y_min - delta, y_max+ + delta])
  
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("---")
    
    # load once
    CSV_URL = "https://raw.githubusercontent.com/kapilw25/Evaluation_generated_images/main/results/evaluation_results.csv"
    df_filt, df_norm, METRICS = load_and_prepare_data(CSV_URL)

    # sidebar controls
    plot_type = st.sidebar.selectbox("Plot Type", ["Radar", "Parallel", "Heatmap", "Scatter"])
    top_n = st.sidebar.slider("Top N Models", min_value=3, max_value=len(df_filt), value=3, step=1)

    if plot_type == "Scatter":
        x_metric = st.sidebar.selectbox("X-axis Metric", METRICS, index=0)
        y_metric = st.sidebar.selectbox("Y-axis Metric", METRICS, index=1)

    # render
    if plot_type == "Radar":
        models = df_filt.sort_values("Weighted Score ⬆️", ascending=False)['Model'].head(top_n).tolist()
        plot_radar(df_norm, METRICS, models)

    elif plot_type == "Parallel":
        plot_parallel(df_norm, df_filt, METRICS, top_n)

    elif plot_type == "Heatmap":
        plot_heatmap(df_norm, df_filt, METRICS, top_n)

    elif plot_type == "Scatter":
        plot_scatter(df_norm, df_filt, x_metric, y_metric, top_n)
    
    st.markdown("---")
  
    st.subheader("Evaluation Metrics - Per-Model Results")
    # wrap text in the table to make sure it fits the screen
    st.markdown(
        """
        <style>
            .wrapped-text {
                white-space: normal;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    # display each numerical value upto 2 decimal points
    st.table(
        df_evaluation_results.style
          .format(lambda x: "{:.2f}".format(x) if isinstance(x, (int, float)) else x)
          .set_table_attributes('class="wrapped-text"')
    )
    
    st.markdown("---")
    # st.subheader("📊 Evaluation Metric Descriptions")

    st.subheader("Evaluation Metric Descriptions")
    explanation = {
        "Metric": [
            "Weighted Score ⬆️",
            "Avg Clip Score ⬆️ [Prompt vs GenIm]",
            "Avg Clip Cos Sim ⬆️ [GenImg vs GTimg]",
            "Avg LPIPS ⬇️ [GenImg vs GTimg]",
            "FID ⬇️ (Frechet inception distance)",
            "MRR ⬆️ (Mean Reciprocal Rank)",
            "Recall@3 ⬆️"
        ],
        "Motivation": [
            "A composite score that combines multiple metrics to provide an overall evaluation of the model's performance. Higher is better.",
            "Measures how well the generated image aligns with the input prompt. A higher score indicates that the image better captures the semantic content of the prompt",
            "Evaluates the similarity between the generated image and the ground truth using CLIP embeddings. Higher values mean closer similarity",
            "Focuses on perceptual similarity; a lower LPIPS score indicates that the generated image is perceptually closer to the ground truth",
            "Measures the distributional difference between generated and real images. A lower FID score signifies that the generated images are closer in distribution to the ground truth images",
            "Assess how effectively the generated image can “retrieve” its corresponding ground truth image. A higher MRR indicates that the correct match is ranked higher in the list of similar images",
            "Shows how often the correct match is in the top 3 results. A higher recall rate indicates better retrieval performance"
        ],
    "Technical Description": [
        "Computes a composite score as: 0.4 × (Normalized CLIP Cosine) + 0.3 × (Normalized LPIPS) + 0.15 × (Normalized FID) + 0.1 × (Normalized Retrieval) + 0.05 × (Normalized CLIP Score). Normalization is performed via min–max scaling, with inversion for metrics where lower is better.",
        "Uses CLIP ViT-B/32 model. Computes logits_per_image by passing (prompt, generated image) into CLIP and taking the image-text similarity score. Average over all samples.",
        "Generates CLIP embeddings for gen_img and GT_img using ViT-B/32. Computes cosine similarity between each pair and averages the similarity scores over the dataset.",
        "Uses LPIPS metric with a pre-trained VGG network. Extracts intermediate features and computes L2 distance between normalized features of gen and GT images. Lower means perceptually similar.",
        "Uses InceptionV3 activations to extract high-dimensional features. Computes Fréchet Distance between generated and real image distributions using their means (μ) and covariances (Σ).",
        "Ranks GT image among all others based on cosine similarity between gen_img and GT pool embeddings. MRR = mean of 1 / rank_of_correct_GT, measuring average retrieval position.",
        "Similar to MRR but binary. For each gen_img, checks if the correct GT image is among top-3 most similar (cosine) images. Computes average success rate over all samples."
    ]
    }
    st.table(pd.DataFrame(explanation))
  
# # --------------------------------------------------- Tab: "System Design"  --------------------------------------------------- 
with tab3:
    st.subheader("System Architecture")
    
    # read SVG file
    with open("README_files/System_Architecture_hyperlinked.svg", "r", encoding="utf-8") as file:
        svg_content = file.read()
        
    # ensure links open in a new tab
    svg_content = svg_content.replace('<a ', '<a target="_blank" ')
    
    # inject white background
    svg_content = svg_content.replace('<svg', '<svg style="background-color:white;"')
    
    # embed as raw HTML so the <a> tags remain active
    components.html(svg_content, height=600, scrolling=True)
    
# --------------------------------------------------- Tab: "Model Architectures"  --------------------------------------------------- 
with tab4:

    # Title
    st.subheader("T2I models architecture")

    # Load image
    image_path = "README_files/T2I_architecture.png"
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    # HTML + JS using OpenSeadragon (fixing curly braces escape)
    components.html(
        f"""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/2.4.2/openseadragon.min.css" />
        <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/2.4.2/openseadragon.min.js"></script>

        <div id="openseadragon1" style="width: 100vw; height: 90vh;"></div>
        <script type="text/javascript">
            var viewer = OpenSeadragon({{
                id: "openseadragon1",
                prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/2.4.2/images/",
                tileSources: {{
                    type: 'image',
                    format: 'jpg',
                    url: "data:image/png;base64,{encoded_image}"
                }},
                gestureSettingsMouse: {{
                    clickToZoom: true,
                    dblClickToZoom: true,
                    flickEnabled: true,
                    pinchToZoom: true
                }},
                showZoomControl: true,
                showNavigationControl: true
            }});
        </script>
        """,
        height=800,
        scrolling=False,
    )

    
# --------------------------------------------------- Tab: "Disclaimer"  --------------------------------------------------- 
with tab5:
    st.subheader("Disclaimer")
    st.markdown("""
    - This app is for education, research and display purposes only. \n
    - All images are generated via huggingface API. \n
    - MultiModal evaluations are executed on local Nvidia machine [**CUDA Device: NVIDIA GeForce RTX 2080 SUPER**] with 8GB vRAM, provided by San Jose State University, CA.
        """)