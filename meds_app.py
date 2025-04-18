import streamlit as st
import torch
import faiss
import pickle
import numpy as np
from PIL import Image
import os
import re
from torchvision import models, transforms
import torch.nn as nn

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_image_base = "./Dataset/All Images"
pkh_pkl_base_path = "./pth_pkl_files"
index_base_path = "./index_files"

# Sidebar: Model selection
st.sidebar.title("Model Settings")
model_choice = st.sidebar.selectbox(
    "Select backbone model:",
    ("ResNet50", "DenseNet121")
)

@st.cache_resource
def load_backbone(choice: str):
    """
    Load and cache a feature-extractor backbone based on the user's choice.
    """
    if choice == "ResNet50":
        # Pretrained ResNet50
        resnet_model = models.resnet50(pretrained=True)
        # Remove final classification layer → 2048-dim features
        backbone = nn.Sequential(*list(resnet_model.children())[:-1])
        dim = 2048

    elif choice == "DenseNet121":
        # DenseNet121 architecture
        densenet_model = models.densenet121(pretrained=False)
        # Load checkpoint dict
        ckpt = torch.load(os.path.join(
            pkh_pkl_base_path,"densenet121_full_checkpoint_iter1_new_dataset.pth"),
            map_location=device
        )
        # Unwrap if saved as {'model_state_dict': ...}
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else (
            ckpt if isinstance(ckpt, dict) else ckpt.state_dict()
        )
        # Remove DataParallel prefixes
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # Filter only feature extractor weights
        feat_dict = {k: v for k, v in state_dict.items() if k.startswith('features.')}
        # Load with strict=False to ignore missing classifier keys
        densenet_model.load_state_dict(feat_dict, strict=False)
        # Build backbone: features + ReLU + global pool → 1024-dim features
        backbone = nn.Sequential(
            densenet_model.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        dim = 1024

    else:
        st.error(f"Unknown model choice: {choice}")
        return None

    backbone.to(device)
    backbone.eval()
    return backbone, dim

# Initialize selected backbone and dimension
model, feat_dim = load_backbone(model_choice)

@st.cache_resource
def load_faiss_index(choice: str, dimension: int):
    """
    Load the FAISS index corresponding to the backbone's feature dimension.
    """
    if choice == "ResNet50" and dimension == 2048:
        return faiss.read_index(os.path.join(
            index_base_path,
            "med_faiss_index.index"
            ))
    elif choice == "DenseNet121" and dimension == 1024:
        return faiss.read_index(os.path.join(
            index_base_path,
            "med_faiss_index_densenet.index"
            ))
    else:
        st.error("No FAISS index matching model choice and dimension.")
        return None

# Image transform pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Helper: clean filename to medication name
def extract_medication_name(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    base = re.sub(r'_aug_\d+', '', base)
    base = re.sub(r'_\d+$', '', base)
    base = re.sub(r'\[[^\]]*\]', '', base)
    base = re.sub(r'[-–](Box|Front box|Blister|Back|Front)', '', base, flags=re.IGNORECASE)
    base = re.sub(r'\(MA.*?\)', '', base)
    base = re.sub(r'\(M&mf.*?\)', '', base)
    return re.sub(r'\s+', ' ', base).strip()

# Helper: medication key for uniqueness
def extract_medication_key(med_name: str) -> str:
    return re.split(r'\(|-', med_name)[0].strip().lower()

# Helper: get embedding from selected backbone
def extract_embedding(image: Image.Image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor).squeeze().cpu().numpy()
    return emb

# Helper: find original image path
def get_original_image_path(augmented_name: str) -> str:
    clean = extract_medication_name(augmented_name)
    for root, _, files in os.walk(original_image_base):
        for fname in files:
            if extract_medication_name(fname) == clean:
                return os.path.join(root, fname)
    return None

# Load FAISS index and metadata
index = load_faiss_index(model_choice, feat_dim)
with open(os.path.join(
    pkh_pkl_base_path, 
    "med_image_paths.pkl"), 
    "rb") as f:
    image_paths = pickle.load(f)
with open(os.path.join(
    pkh_pkl_base_path, 
    "med_names.pkl"
    ), 
    "rb") as f:
    med_names = pickle.load(f)

# Streamlit UI
st.title("💊 Medication Lookalike Finder")
st.subheader(f"Selected model: **{model_choice}**")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    query_img = Image.open(uploaded_file)
    st.image(query_img, caption=f"Query: {uploaded_file.name}", width=250)

    q_name = extract_medication_name(uploaded_file.name)
    q_key = extract_medication_key(q_name)
    q_emb = extract_embedding(query_img).reshape(1, -1)

    # FAISS search
    distances, indices = index.search(q_emb, k=50)
    
    # 1) gather top unique candidates (up to 20 to allow for gaps)
    candidates = []
    seen_keys = set()
    for dist, idx in zip(distances[0], indices[0]):
        name = med_names[idx]
        key  = extract_medication_key(name)
        if key != q_key and key not in seen_keys:
            candidates.append((name, image_paths[idx], dist))
            seen_keys.add(key)
        if len(candidates) >= 20:
            break

    # 2) filter down to the first 5 whose originals actually exist
    results = []
    for name, path, score in candidates:
        orig = get_original_image_path(name)
        if orig and os.path.exists(orig):
            results.append((name, orig, score))
        if len(results) == 5:
            break

    # 3) warn if we couldn’t find 5 originals
    if len(results) < 5:
        st.warning(f"Only found {len(results)} similar medications with available originals.")

    # 4) display however many we have
    st.subheader("🔍 Top 5 Visually Similar Medications")
    cols = st.columns(len(results))
    for col, (name, orig, score) in zip(cols, results):
        with col:
            st.image(orig, caption=f"{name}\nScore: {round(score, 2)}")