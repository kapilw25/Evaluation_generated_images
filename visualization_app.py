import streamlit as st
import pandas as pd
from PIL import Image
import os

st.title("MultiModal Recommendation for Text-to-Image Generation")

csv_file = "evaluation_results.csv"
image_dir = "image_generated"

# check is CSV and images directory exist
if not os.path.exists(csv_file):
    st.error(f"{csv_file} not found.")
    st.stop()
    
# Ensure that "image_generated" folder exists

if not os.path.exists(image_dir):
    st.error(f"{image_dir} folder not found")
    st.stop()
    
# Load CSV
df = pd.read_csv(csv_file)

# create 2 tabs:
tab1, tab2, tab3 = st.tabs([
    "Compare Images",
    "Evaluation Metrics",
    "Disclaimer"
    
])

# --------------------------------------------------- Tab1: Compare Image  --------------------------------------------------- 
with tab1:
    # Make sure these names match exactly with pipeline's CSV headers
    col_clip = "CLIP score"
    col_cos = "Cosine similarity score"
        
    # Extract Unique prompts from the CSV
    prompts = sorted(df["Prompt"].unique())

    # dropdown to select which prompt to display
    selected_prompt = st.selectbox("Select a prompt:", prompts)

    # Filter DataFrame by the chosen prompt
    filtered_df = df[df["Prompt"] == selected_prompt]

    # show the selected Prompt as a heading
    st.markdown(f"## Prompt: **{selected_prompt}**")

    # For each model in the filtered DataFrame, vertical display the image + metrics
    models_for_prompt = filtered_df["Model"].unique()

    for model_name in models_for_prompt:
        row = filtered_df[filtered_df["Model"] == model_name]
        
        if row.empty:
            st.warning(f"No data for {model_name} & {selected_prompt}")
            continue
        
        # we assume each (model, prompt) is a single row >> use.iloc[0]
        row_data = row.iloc[0]
        
        # Extract image index and other metrics
        image_index = int(row_data["Image_Index"])
        clip_score = row_data[col_clip]
        cos_score = row_data[col_cos]
        
        # Retrieve the generated filename from the CSV
        image_filename = row_data["Filename"]
        image_path = os.path.join(image_dir, image_filename)

        
        # check if file exists
        if not os.path.exists(image_path):
            st.error(f"Image not found: {image_filename}")
            continue
        
        # Enlarge model name text via HTML styling
        st.markdown(
            f"<h3 style='font-size:24px; color:#2e5f9c;'>{model_name}</h3>",
            unsafe_allow_html=True
        )
        
        st.image(Image.open(image_path), use_container_width=True)
        
        st.write(f"**CLIP Score**: {clip_score}")
        st.write(f"**Cosine Similarity**: {cos_score}")
                
    st.write("---")
    st.markdown("**End of Comparison**")

# --------------------------------------------------- Tab2: "Evaluation Metrics"  --------------------------------------------------- 
with tab2:
    st.subheader("Evaluation Metrics - Per-Model Results")
    df1 = pd.read_csv("evaluation_results.csv")
    st.dataframe(df1)

    st.subheader("Evaluation Metrics - Retrieval & FID Results")
    df2 = pd.read_csv("evaluation_retrieval_fid_results.csv")
    st.dataframe(df2)

    
# --------------------------------------------------- Tab3: "Disclaimer"  --------------------------------------------------- 
with tab3:
    st.subheader("Disclaimer and Project Structure")
    st.markdown("""
**Disclaimer:** This app is for display purposes only. All images are generated on a local Nvidia machine [**CUDA Device: NVIDIA GeForce RTX 2080 SUPER**] with 8GB vRAM.

**Project Structure:**
- **[evaluation_metrics.py](https://github.com/kapilw25/Evaluation_generated_images/blob/main/evaluation_metrics.py)**  
  Contains functions to calculate evaluation metrics for text-to-image outputs.
- **[evaluation_pipeline.py](https://github.com/kapilw25/Evaluation_generated_images/blob/main/evaluation_pipeline.py)**  
  Generates images using text-to-image models and computes evaluation metrics, saving results to CSV.
- **[evaluation_results.csv](https://github.com/kapilw25/Evaluation_generated_images/blob/main/evaluation_results.csv)**  
  CSV file that stores all computed evaluation metrics.
- **[visualization_app.py](https://github.com/kapilw25/Evaluation_generated_images/blob/main/visualization_app.py)**  
  Streamlit app that visualizes the generated images and evaluation metrics.
- **[image_generated/](https://github.com/kapilw25/Evaluation_generated_images/blob/main/image_generated)**  
  Directory containing the generated images.
    """, unsafe_allow_html=True)
