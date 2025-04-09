# from dotenv import load_dotenv
# from huggingface_hub import InferenceClient
import os, json, csv, time, re
import pandas as pd

# Load annotation file paths
shape_annotation_path = "DeepFashion/labels/shape/shape_anno_all.txt"
fabric_annotation_path = "DeepFashion/labels/texture/fabric_ann.txt"
pattern_annotation_path = "DeepFashion/labels/texture/pattern_ann.txt"

# Feature names for shape, fabric, and pattern
shape_feature_names = [
    "sleeve_length", "lower_clothing_length", "socks", "hat", "glasses",
    "neckwear", "wrist_wearing", "ring", "waist_accessories",
    "neckline", "outer_clothing_cardigan", "upper_clothing_covers_navel"
]
fabric_feature_names = ["upper_fabric", "lower_fabric", "outer_fabric"]
pattern_feature_names = ["upper_color", "lower_color", "outer_color"]

# Feature maps for shape, fabric, and pattern
shape_feature_maps = {
    0: ["sleeveless", "short-sleeve", "medium-sleeve", "long-sleeve", "not long-sleeve", "NA"],
    1: ["three-point", "medium short", "three-quarter", "long", "NA"],
    2: ["no", "socks", "leggings", "NA"],
    3: ["no", "yes", "NA"],
    4: ["no", "eyeglasses", "sunglasses", "glasses in hand/clothes", "NA"],
    5: ["no", "yes", "NA"],
    6: ["no", "yes", "NA"],
    7: ["no", "yes", "NA"],
    8: ["no", "belt", "have clothing", "hidden", "NA"],
    9: ["v-shape", "square", "round", "standing", "lapel", "suspenders", "NA"],
    10: ["yes", "no", "NA"],
    11: ["no", "yes", "NA"]
}
fabric_feature_map = ["denim", "cotton", "leather", "furry", "knitted", "chiffon", "other", "NA"]
pattern_feature_map = ["floral", "graphic", "striped", "pure color", "lattice", "other", "color block", "NA"]

# Load shape annotations
shape_annotations = {}
with open(shape_annotation_path, "r") as f:
    for line in f:
        tokens = line.strip().split()
        img_name, shape_vals = tokens[0], list(map(int, tokens[1:]))
        shape_annotations[img_name] = {
            shape_feature_names[i]: shape_feature_maps[i][val]
            for i, val in enumerate(shape_vals)
        }

# Load fabric annotations
fabric_annotations = {}
with open(fabric_annotation_path, "r") as f:
    for line in f:
        tokens = line.strip().split()
        img_name, values = tokens[0], list(map(int, tokens[1:]))
        fabric_annotations[img_name] = {
            fabric_feature_names[i]: fabric_feature_map[val]
            for i, val in enumerate(values)
        }

# Load pattern annotations
pattern_annotations = {}
with open(pattern_annotation_path, "r") as f:
    for line in f:
        tokens = line.strip().split()
        img_name, values = tokens[0], list(map(int, tokens[1:]))
        pattern_annotations[img_name] = {
            pattern_feature_names[i]: pattern_feature_map[val]
            for i, val in enumerate(values)
        }

# Load JSON prompt map
with open("DeepFashion/captions_sample.json", "r") as f:
    prompt_map = json.load(f)

# Function to extract basic metadata from image_key
def extract_image_metadata(image_key):
    base_name = os.path.basename(image_key)
    parts = base_name.split('-')
    gender = parts[0]
    category = parts[1]
    view = parts[-1].replace('.jpg', '').split('_')[-1]
    return gender, category, view

# Build enriched metadata list
metadata_list = []
for image_key, prompt in prompt_map.items():
    gender, category, view = extract_image_metadata(image_key)
    shape_data = shape_annotations.get(image_key, {k: "unknown" for k in shape_feature_names})
    fabric_data = fabric_annotations.get(image_key, {k: "unknown" for k in fabric_feature_names})
    pattern_data = pattern_annotations.get(image_key, {k: "unknown" for k in pattern_feature_names})
    
    metadata_entry = {
        "image_key": image_key,
        "prompt": prompt,
        "gender": gender,
        "category": category,
        "view": view
    }
    metadata_entry.update(shape_data)
    metadata_entry.update(fabric_data)
    metadata_entry.update(pattern_data)
    
    # Combine prompt with all metadata features and their values
    all_features = [
        ("gender", gender),
        ("category", category),
        ("view", view),
        ("sleeve_length", shape_data.get("sleeve_length", "unknown")),
        ("lower_clothing_length", shape_data.get("lower_clothing_length", "unknown")),
        ("socks", shape_data.get("socks", "unknown")),
        ("hat", shape_data.get("hat", "unknown")),
        ("glasses", shape_data.get("glasses", "unknown")),
        ("neckwear", shape_data.get("neckwear", "unknown")),
        ("wrist_wearing", shape_data.get("wrist_wearing", "unknown")),
        ("ring", shape_data.get("ring", "unknown")),
        ("waist_accessories", shape_data.get("waist_accessories", "unknown")),
        ("neckline", shape_data.get("neckline", "unknown")),
        ("outer_clothing_cardigan", shape_data.get("outer_clothing_cardigan", "unknown")),
        ("upper_clothing_covers_navel", shape_data.get("upper_clothing_covers_navel", "unknown")),
        ("upper_fabric", fabric_data.get("upper_fabric", "unknown")),
        ("lower_fabric", fabric_data.get("lower_fabric", "unknown")),
        ("outer_fabric", fabric_data.get("outer_fabric", "unknown")),
        ("upper_color", pattern_data.get("upper_color", "unknown")),
        ("lower_color", pattern_data.get("lower_color", "unknown")),
        ("outer_color", pattern_data.get("outer_color", "unknown"))
    ]
    meta_parts = [f"{k}: {v}" for k, v in all_features]
    metadata_entry["prompt_MetaData"] = f"{prompt} | " + ", ".join(meta_parts)
    
    metadata_list.append(metadata_entry)


# Save the enriched metadata to DeepFashion/captions_sample.csv
df_metadata = pd.DataFrame(metadata_list)
df_metadata.to_csv("DeepFashion/captions_sample.csv", index=False)
print("Metadata stored in 'DeepFashion/captions_sample.csv'")
