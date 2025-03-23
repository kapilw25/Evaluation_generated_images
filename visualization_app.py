import streamlit as st
import pandas as pd
from PIL import Image
import os

# use full screen width for the app
st.set_page_config(layout="wide")

st.title("Text-to-Image Models Evaluation Dashboard")

# Load Evaluation results from CSV
csv_file = "evaluation_results.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    st.subheader("Evaluation Metrics")
    st.dataframe(df)
else:
    st.error(f"{csv_file} not found.")
    st.stop()
    
# Ensure that "image_generated" folder exists
image_dir = "image_generated"
if not os.path.exists(image_dir):
    st.error(f"{image_dir} folder not found")
    st.stop()
    
#    Make sure these names match exactly with your pipeline's CSV headers
column_clip_score = "CLIP score"
column_bleu_score = "BLEU score"
column_cosine_score = "Cosine similarity score"
    
# Extract Unique model and prompts from the CSV
models = df["Model"].unique()
prompts = df["Prompt"].unique()

st.subheader("Image Comparison for Computer Vision Models")
    
# For each prompt, show a row of image (one per model)
for prompt in prompts:
    st.markdown(f"### Prompt: *{prompt}*")
    columns = st.columns(len(models)) # one column per model
    
    for col_index, model_name in enumerate(models):
        # filter the dataframe row for (model_name, prompt)
        row = df[
            (df["Model"] == model_name) &\
            (df["Prompt"]== prompt)
        ]
        
        # if no row found for this combination, skip
        if row.empty:
            columns[col_index].warning(f"No Data for {model_name} & {prompt}")
            continue
        
        # Extract image index and other metrics
        image_index = int(row["Image_Index"].values[0])
        clip_score = row[column_clip_score].values[0]
        bleu_score = row[column_bleu_score].values[0]
        cos_score = row[column_cosine_score].values[0]
        
        # Construct Expected filename
        image_filename = f"{model_name}_image_{image_index}.jpg"
        image_path = os.path.join(image_dir, image_filename)
        
        # check if file exists
        if not os.path.exists(image_path):
            columns[col_index].error(f"Image not found: {image_filename}")
            continue
        
        # Display the image and metrics
        with columns[col_index]:
            st.image(Image.open(image_path), caption=f"**{model_name}**", use_container_width=True)
            st.write(f"**CLIP Score**: {clip_score}")
            st.write(f"**BLEU Score**: {bleu_score}")
            st.write(f"**Cosine Similarity**: {cos_score}")
            
st.write("---")
st.markdown("**End of Comparison**")