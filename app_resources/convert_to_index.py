import os
import pickle
import faiss
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

# 1) Setup device & DenseNet backbone
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dnet = models.densenet121(pretrained=False).to(device)

ckpt = torch.load("./model_checkpoints/densenet121_full_checkpoint_iter1.pth", map_location=device)
sd = ckpt.get("model_state_dict", ckpt if isinstance(ckpt, dict) else ckpt.state_dict())
sd = {k.replace("module.", ""): v for k, v in sd.items()}
feat_sd = {k: v for k, v in sd.items() if k.startswith("features.")}
dnet.load_state_dict(feat_sd, strict=False)

backbone = nn.Sequential(
    dnet.features,
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1, 1))
).to(device).eval()

# 2) Transform pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 3) Prepare your dataset embeddings
image_dir = "../Dataset/All Images"
paths = []
embeddings = []

# only consider real image extensions
VALID_EXT = {'.jpg', '.jpeg', '.png'}

for root, _, files in os.walk(image_dir):
    for fname in files:
        ext = os.path.splitext(fname)[1].lower()
        if ext not in VALID_EXT:
            # skip zone.identifier or any other junk
            continue

        img_path = os.path.join(root, fname)
        try:
            img = Image.open(img_path).convert("RGB")
        except (IOError, OSError):
            print(f"⚠️  Skipping unreadable file {img_path}")
            continue

        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = backbone(tensor).squeeze().cpu().numpy()

        embeddings.append(emb.astype("float32"))
        paths.append(img_path)

print(f"Found {len(paths)} valid images for indexing.")
if not paths:
    raise RuntimeError(f"No images indexed—check that '{image_dir}' contains only .jpg/.png files.")

# Stack, build FAISS index, write it out
emb_array = np.stack(embeddings)
dim = emb_array.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(emb_array)
faiss.write_index(index, "med_faiss_index_densenet.index")
print("✅ Wrote med_faiss_index_densenet.index")

# Save the paths list
with open("med_image_paths_densenet.pkl", "wb") as f:
    pickle.dump(paths, f)
print("✅ Wrote med_image_paths_densenet.pkl")
