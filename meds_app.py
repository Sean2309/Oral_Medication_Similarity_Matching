# Imports and Setup
import streamlit as st
import torch
import faiss
import pickle
import numpy as np
from PIL import Image
import os
import re
from torchvision import models, transforms
import pandas as pd
from collections import defaultdict

# Configuration
main_folder = r"augmented_images(Stef)" # File format
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model and the pre-trained model's weights
# Load model EXACTLY like notebook
resnet = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(resnet.children())[:-1])
model.eval().to(device)

# Image transform (same as notebook)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Name extraction
def extract_medication_name(filename):
    name = os.path.splitext(filename)[0]
    name = re.sub(r'_aug_\d+', '', name)
    name = re.sub(r'_\d+$', '', name)
    name = re.sub(r'\[[^\]]*\]', '', name)
    name = re.sub(r'[-–](Box|Front box|Blister|Back|Front)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\(MA.*?\)', '', name)
    name = re.sub(r'\(M&mf.*?\)', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def extract_medication_key(med_name):
    return re.split(r'\(|-', med_name)[0].strip().lower()

def extract_embedding(image):
    if (isinstance(image, np.ndarray)):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor).squeeze().cpu().numpy()
    return embedding


# original_image_base = "./augmented_images(Stef)/All Images"

# def get_original_image_path(med_name):
#     """
#     Given a medication name, search the 'All Images' folder 
#     to find the original image by matching cleaned filenames (without augmentation tags).
#     """
#     # Strip augmentation suffixes and clean up the name
#     med_key = extract_medication_key(med_name)

#     # Walk through all subfolders in the 'All Images' directory
#     for root, dirs, files in os.walk(original_image_base):
#         for fname in files:
#             clean_name = extract_medication_key(extract_medication_name(fname))
            
#             # Match the original image key with the cleaned medication key
#             if clean_name == med_key:
#                 return os.path.join(root, fname)
    
#     return None  # Return None if the original image is not found

# Load saved data
index = faiss.read_index("med_faiss_index.index")
with open("med_image_paths.pkl", "rb") as f:
    image_paths = pickle.load(f)
with open("med_names.pkl", "rb") as f:
    med_names = pickle.load(f)

# UI
st.title("💊 Medication Lookalike Finder")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    query_image = Image.open(uploaded_file)
    st.image(query_image, caption=f"Query: {uploaded_file.name}", width=250)

    query_name = extract_medication_name(uploaded_file.name)
    query_key = extract_medication_key(query_name)
    query_embedding = extract_embedding(query_image).reshape(1, -1)

    # FAISS search
    D, I = index.search(query_embedding, k=50)
    medkey_to_best = {}

    for dist, idx in zip(D[0], I[0]):
        name = med_names[idx]
        key = extract_medication_key(name)
        if key != query_key and key not in medkey_to_best:
            medkey_to_best[key] = (name, image_paths[idx], dist)
        if len(medkey_to_best) == 5:
            break

    results = sorted(medkey_to_best.values(), key=lambda x: x[2])

    st.subheader("🔍 Top 5 Visually Similar Medications")
    cols = st.columns(5)
    for i, (name, path, dist) in enumerate(results):
        with cols[i]:
            if os.path.exists(path):
                st.image(path, caption=f"{name}\nScore: {round(dist, 2)}")
            else:
                st.error(f"Image not found:\n{path}")

# =============================================================================================================================
# # UI
# st.title("💊 Medication Lookalike Finder")

# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
# if uploaded_file:
#     query_image = Image.open(uploaded_file)
#     st.image(query_image, caption=f"Query: {uploaded_file.name}", width=250)

#     query_name = extract_medication_name(uploaded_file.name)
#     query_key = extract_medication_key(query_name)
#     query_embedding = extract_embedding(query_image).reshape(1, -1)

#     # FAISS search
#     D, I = index.search(query_embedding, k=50)
#     medkey_to_best = {}

#     for dist, idx in zip(D[0], I[0]):
#         name = med_names[idx]
#         key = extract_medication_key(name)
#         if key != query_key and key not in medkey_to_best:
#             medkey_to_best[key] = (name, image_paths[idx], dist)
#         if len(medkey_to_best) == 5:
#             break

#     results = sorted(medkey_to_best.values(), key=lambda x: x[2])

#     st.subheader("🔍 Top 5 Visually Similar Medications")
#     cols = st.columns(5)
#     for i, (name, path, dist) in enumerate(results):
#         with cols[i]:
#             # Get the original image path using the cleaned medication name
#             original_path = get_original_image_path(name)
            
#             if original_path and os.path.exists(original_path):
#                 # Display the original image
#                 st.image(original_path, caption=f"{name}\nScore: {round(dist, 2)}")
#             else:
#                 # If the original image is not found, show a warning
#                 st.warning(f"Original image not found for {name}")

