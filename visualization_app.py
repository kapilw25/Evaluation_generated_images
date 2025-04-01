import streamlit as st
import pandas as pd
from PIL import Image
import os, ast
import subprocess
import time

st.set_page_config(layout="wide")

st.title("MultiModal Recommendation for Text-to-Image Generation")

if "mlflow_started" not in st.session_state:
  subprocess.Popen(["mlflow", "ui", "--port", "5050", "--backend-store-uri", "mlruns"])
  time.sleep(3) # giving the MLflow UI time to start
  st.session_state.mlflow_started = True

gen_img_metadata = "image_generated/gen_img_metadata.csv"
# Generated image metadata column names: ['model', 'image_key', 'prompt', 'gen_img_path']

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
    
# Load CSV
# df = pd.read_csv(csv_file)
df = pd.read_csv(gen_img_metadata)


# create 2 tabs:
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Compare Images",
    "Evaluation Metrics",
    "MLflow UI",
    "Project Structure",
    "Disclaimer"
])

# --------------------------------------------------- Tab1: Compare Image  --------------------------------------------------- 
with tab1:        
    # Extract Unique prompts from the CSV
    prompts = sorted(df["prompt"].unique())

    # dropdown to select which prompt to display
    selected_prompt = st.selectbox("Select a prompt:", prompts)
    
    filtered_df = df[df["prompt"] == selected_prompt]
    if not filtered_df.empty:
      gt_filename = filtered_df.iloc[0]["image_key"]
      prompt_text = selected_prompt
    else:
      st.error("No data found for the selected prompt")
      st.stop()
    
    
    # show ground-truth image
    gt_path = os.path.join(ground_truth_images, gt_filename)
    
    # Filter DataFrame by the chosen prompt
    filtered_df = df[df["prompt"] == selected_prompt]

    # For each model in the filtered DataFrame, vertical display the image + metrics
    models_for_prompt = filtered_df["model"].unique()
    
    # Resize function to unify image sizes
    def resize_image(img, size=(768, 1024)):
      return img.resize(size)
    
    # gather all images (ground truth + generated) into a list
    images_to_display = []
    
    # ground truth first
    if os.path.exists(gt_path):
      gt_img = Image.open(gt_path).convert("RGB")
      gt_img = resize_image(gt_img)
      images_to_display.append(("Ground Truth from DeepFashion Dataset", gt_img))
    else:
      st.warning(f"Ground Truth image not found: {gt_filename}")
      
    # then each model's generated image
    for model_name in models_for_prompt:
      row = filtered_df[filtered_df["model"]==model_name]
      if row.empty:
        continue
      row_data = row.iloc[0]
      # image_filename = row_data["Filename"]
      # image_path = os.path.join(image_dir, image_filename)
      image_path = row_data["gen_img_path"]
      if os.path.exists(image_path):
        gen_img = Image.open(image_path).convert("RGB")
        gen_img = resize_image(gen_img)
        images_to_display.append((model_name, gen_img))
        
      
    # display images in a grid
    cols_per_row = 3
    rows_needed = (len(images_to_display) + cols_per_row -1 )//cols_per_row
    
    index = 0
    for _ in range(rows_needed):
      cols = st.columns(cols_per_row)
      for col_i in range(cols_per_row):
        if index < len(images_to_display):
          model_title, img_obj = images_to_display[index]
          with cols[col_i]:
                st.markdown(
                    f"<h3 style='font-size:24px; color:#2e5f9c;'>{model_title}</h3>",
                    unsafe_allow_html=True
                )
                st.image(img_obj, use_container_width=True)
          index += 1
           
    st.markdown(f"<p style='font-size:14px;'>{prompt_text}</p>", unsafe_allow_html=True)
    st.write("---")
    st.markdown("**End of Comparison**")

# --------------------------------------------------- Tab2: "Evaluation Metrics"  --------------------------------------------------- 
with tab2:
    st.subheader("Evaluation Metrics - Per-Model Results")
    df1 = pd.read_csv("results/evaluation_results.csv")
    st.dataframe(df1)
    
# --------------------------------------------------- Tab3: "MLflow UI"  --------------------------------------------------- 
with tab3:
  st.subheader("Models' Performance analysis via Mlflow")
  st.markdown("Live experiment tracking and comaprison across model/prompt combinations. The MLflow UI is embedded below.")
  st.components.v1.iframe("http://localhost:5050", height=1200, scrolling=True)

# --------------------------------------------------- Tab4: "Project Structure"  --------------------------------------------------- 
with tab4:

    st.subheader("Project Structure")
    st.markdown("""
- **[evaluation_metrics.py](https://github.com/kapilw25/Evaluation_generated_images/blob/main/evaluation_metrics.py)**  
  Contains functions to calculate evaluation metrics for text-to-image outputs.
- **[evaluation_pipeline.py](https://github.com/kapilw25/Evaluation_generated_images/blob/main/evaluation_pipeline.py)**  
  Generates images using text-to-image models and computes evaluation metrics, saving results to CSV.
- **[evaluation_results.csv](https://github.com/kapilw25/Evaluation_generated_images/blob/main/results/evaluation_results.csv)**  
  CSV file that stores all computed evaluation metrics.
- **[visualization_app.py](https://github.com/kapilw25/Evaluation_generated_images/blob/main/visualization_app.py)**  
  Streamlit app that visualizes the generated images and evaluation metrics.
- **[image_generated/](https://github.com/kapilw25/Evaluation_generated_images/blob/main/image_generated)**  
  Directory containing the generated images.
    """, unsafe_allow_html=True)
    
    st.subheader("System Architecture")
    # st.image("README_files/architechture.png", use_container_width=True)

# --------------------------------------------------- Tab5: "Disclaimer"  --------------------------------------------------- 
with tab5:
    st.subheader("Disclaimer")
    st.markdown("""
**Disclaimer:** This app is for education, research and display purposes only. All images are generated via huggingface API and CLIP based evaluation with  local Nvidia machine [**CUDA Device: NVIDIA GeForce RTX 2080 SUPER**] with 8GB vRAM, provided by San Jose State University, CA.
    """)