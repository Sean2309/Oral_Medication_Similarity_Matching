# 💊 Oral Medication Similarity Matching

An end-to-end workflow for finding visually similar oral-medication packages using deep-learning embeddings and FAISS nearest--neighbour search.  
The project ships with:

* **Streamlit app** for interactive querying (`meds_app.py`)  
* **Training notebooks** for ResNet-50 and DenseNet-121 backbones (`training/`)  
* **Utilities** to (re)build FAISS indexes from raw images (`app_resources/convert_to_index.py`)  
* **EDA notebooks & plots** for quick dataset exploration (`eda/`)

---

## 🗂️ Folder Structure

```text
oral_medication_similarity_matching/
├── app_resources/
│   ├── faiss_indexes/            # FAISS *.index files used by the Streamlit app
│   ├── model_checkpoints/        # Trained *.pth files & accompanying *.pkl metadata
│   ├── convert_to_index.py       # Script to regenerate FAISS index + metadata
│   └── preprocessing.ipynb       # Optional: feature-extraction sanity checks
│
├── Dataset/
│   ├── All Images/               # Raw, canonical medication images
│   ├── augmented_images(Stef)/   # Augmented variants (for training / demos)
│   ├── demo images/              # Quick-demo images for end-users
│   └── Organised Images_STEF/    # Alternative structure used in earlier experiments
│
├── eda/
│   ├── eda.ipynb                 # Exploratory Data Analysis notebook
│   ├── label_summary.csv         # Class breakdown table
│   └── plots/                    # Generated charts / figures
│
├── training/                     # Model-training & experimentation notebooks
│   ├── AlexNet.ipynb
│   ├── DenseNet.ipynb
│   ├── EfficientNet.ipynb
│   └── Resnet-50_ChosenModel.ipynb
│
├── meds_app.py                   # 💻 Streamlit entry-point
├── requirements.txt              # Python dependencies
└── README.md                     # You’re here!
```

## 🚀 Quick Start
1.  Create & activate an environment
```bash
conda create --name <name of env> python=3.10
conda activate <name of env>
python -m pip install -r requirements.txt
```

2. Getting `All Images` Dataset
- Download the "Oral Dose Forms" dataset and add the images contents into the "`Dataset/All Images`" folder

3. Launch the Streamlit app
```bash
streamlit run meds_app.py
```

Upload a medication image and the app returns the top-5 visually similar products (excluding identical matches).
<br>
We've included demo images for basic testing in the `Dataset` folder. 
## 🔧 (Re)building the FAISS Index
If you add new images or retrain a backbone:

```bash
python app_resources/convert_to_index.py
```
The script will:

1. Load the chosen CNN backbone
2. Extract 1 × N embeddings from every image in Dataset/All Images/
3. Write: 
- app_resources/faiss_indexes/med_faiss_index_*.index
- app_resources/model_checkpoints/med_image_paths*.pkl
- app_resources/model_checkpoints/med_names.pkl

5. The Streamlit app picks these up automatically on the next run.

## 🧠 Models

| Backbone      | Feature Dim | Training Notes                          |
|---------------|-------------|------------------------------------------|
| ResNet-50     | 2048        | ImageNet pretrained (frozen)             |
| DenseNet-121  | 1024        | Custom-trained on medication set         |
| AlexNet       | 4096        | ImageNet pretrained                      |
| EfficientNet  | 1280        | ImageNet pretrained                      |

All models produce global-average-pooled embeddings suitable for L2 similarity search with FAISS.


## 📈 Exploratory Data Analysis
See eda/eda.ipynb for:

- Label distribution & class imbalance
- Sample augmentations
- t-SNE & PCA visualisations of embedding clusters
- Charts are auto-exported to eda/plots/.