import streamlit as st
import pandas as pd
from PIL import Image
import os

st.title("Text-to-Image Models Evaluation dashboard")

# Load Evaluation results from CSV
csv_file = "evaluation_results.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    st.subheader("Evaluation Metrics")
    st.dataframe(df)
else:
    st.error(f"{csv_file} not found.")
    
# List generated images
image_dir = "image_generated"
if os.path.exists(image_dir):
    st.subheader("Generated Images by Model and Prompt")
    models = os.listdir(image_dir)
    models.sort()
    for image_file in models:
        image_path = os.path.join(image_dir, image_file)
        # Extract model and image index from filename
        st.markdown(f"**{image_file}**")
        img = Image.open(image_path)
        st.image(img, use_column_width=True)
else:
    st.error(f"{image_dir} folder not found")