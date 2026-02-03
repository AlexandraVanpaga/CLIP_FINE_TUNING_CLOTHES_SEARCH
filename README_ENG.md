# CLIP Fine-Tuning for Clothing Search

A project for fine-tuning the CLIP model on a clothing product dataset to create an image search system based on text descriptions.

## Description

A system for searching clothing products by text queries using a fine-tuned CLIP model has been implemented. The model learned to understand the relationship between clothing images and their text descriptions, achieving ~93% accuracy.

### Key Features:
- Product search by text description
- CLIP Score > 30 (achieved 33.6)
- Accuracy ~93% (Loss 0.0719)
- Data augmentation for improved generalization
- Vector embedding database for fast search

## Project Structure
```
CLIP_FT/
├── config/
│   └── paths.py              # Paths configuration
├── data/
│   ├── raw/                  # Raw data
│   ├── extracted/            # Extracted data
│   └── processed/            # Processed data
│       ├── split/            # Train/test split
│       └── image_embeddings.npz  # Vector representations
├── src/
│   ├── get_raw_data.py       # Dataset download
│   ├── Preprocess.py         # Data preprocessing
│   ├── Model_training.py     # Model training
│   └── search.py             # Product search
├── checkpoints/
│   ├── best_model.pt         # Best model
│   └── training_history.png  # Training graphs
└── notebooks/
    └── solution.ipynb        # Analysis and experiments
```

## Installation

### Requirements
- Python 3.8+
- CUDA (optional, for GPU)

### Installing Dependencies
```bash
# Creating a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Installing packages
pip install -r requirements.txt
```

## Usage

### 1. Downloading the Dataset
```bash
python src/get_raw_data.py
```

### 2. Data Preprocessing
```bash
python src/preprocess.py
```

Performs:
- Duplicate removal
- Cleaning empty descriptions
- Train/test split (90/10)
- Concatenating columns with available text descriptions into `full_text`

### 3. Model Training
```bash
python src/Model_training.py
```

**Training Parameters:**
- Model: `openai/clip-vit-base-patch32`
- Epochs: up to 5 (with early stopping)
- Learning rate: 5e-6 with Cosine Annealing
- Batch size: 32
- Augmentation: RandomCrop, Flip, ColorJitter, Rotation

### 4. Product Search
```bash
python src/search.py
```

Interactive search system:
```
Query: blue jeans
→ Top-5 relevant products with CLIP Scores
```

## 📈 Results

| Metric | Value |
|--------|-------|
| **Train CLIP Score** | 33.98 |
| **Test CLIP Score** | 33.61 |
| **Train Loss** | 0.0335 |
| **Test Loss** | 0.0719 |
| **Accuracy** | ~93% |

### Training Graphs

![Training History](checkpoints/training_history.png)

**Conclusions:**
- Target (CLIP Score > 30) achieved
- No overfitting
- Augmentation is effective
- Model is ready for use

## 🔧 API Functions

### Product Search
```python
from src.search import search_products, load_embeddings

# Loading data
data = load_embeddings('data/processed/image_embeddings.npz')

# Search
results = search_products(
    query="red summer dress",
    model=model,
    processor=processor,
    image_embeddings=data['embeddings'],
    image_names=data['image_names'],
    dataframe=data['dataframe'],
    top_k=5
)
```

### Model Loading
```python
from transformers import CLIPModel, CLIPProcessor
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Dataset

**Fashion Product Images Dataset**
- Source: Kaggle
- Size: ~44k images
- Categories: clothing, footwear, accessories
- Format: JPG images + CSV with metadata

**CSV Structure:**
- `image`: file name
- `description`: product description
- `display name`: product name
- `category`: category

## Improvements

Implemented optimizations:
- Image augmentation (crop, flip, color jitter, rotation)
- Learning Rate Scheduling (Cosine Annealing)
- Concatenation of title and description (`full_text`)

Possible improvements:
- Hard Negative Mining for cases where the model confuses dresses and nightgowns or short skirts and shorts
- Freezing early layers
- Expanding the dataset
- Using a more powerful model

## Technologies

- **PyTorch** — training framework
- **Transformers** — CLIP model from HuggingFace
- **NumPy** — working with vectors
- **Pandas** — data processing
- **Pillow** — image processing
- **Matplotlib** — visualization
EOF
Salida

# CLIP Fine-Tuning for Clothing Search

A project for fine-tuning the CLIP model on a clothing product dataset to create an image search system based on text descriptions.

## Description

A system for searching clothing products by text queries using a fine-tuned CLIP model has been implemented. The model learned to understand the relationship between clothing images and their text descriptions, achieving ~93% accuracy.

### Key Features:
- Product search by text description
- CLIP Score > 30 (achieved 33.6)
- Accuracy ~93% (Loss 0.0719)
- Data augmentation for improved generalization
- Vector embedding database for fast search

## Project Structure
```
CLIP_FT/
├── config/
│   └── paths.py              # Paths configuration
├── data/
│   ├── raw/                  # Raw data
│   ├── extracted/            # Extracted data
│   └── processed/            # Processed data
│       ├── split/            # Train/test split
│       └── image_embeddings.npz  # Vector representations
├── src/
│   ├── get_raw_data.py       # Dataset download
│   ├── Preprocess.py         # Data preprocessing
│   ├── Model_training.py     # Model training
│   └── search.py             # Product search
├── checkpoints/
│   ├── best_model.pt         # Best model
│   └── training_history.png  # Training graphs
└── notebooks/
    └── solution.ipynb        # Analysis and experiments
```

## Installation

### Requirements
- Python 3.8+
- CUDA (optional, for GPU)

### Installing Dependencies
```bash
# Creating a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Installing packages
pip install -r requirements.txt
```

## Usage

### 1. Downloading the Dataset
```bash
python src/get_raw_data.py
```

### 2. Data Preprocessing
```bash
python src/preprocess.py
```

Performs:
- Duplicate removal
- Cleaning empty descriptions
- Train/test split (90/10)
- Concatenating columns with available text descriptions into `full_text`

### 3. Model Training
```bash
python src/Model_training.py
```

**Training Parameters:**
- Model: `openai/clip-vit-base-patch32`
- Epochs: up to 5 (with early stopping)
- Learning rate: 5e-6 with Cosine Annealing
- Batch size: 32
- Augmentation: RandomCrop, Flip, ColorJitter, Rotation

### 4. Product Search
```bash
python src/search.py
```

Interactive search system:
```
Query: blue jeans
→ Top-5 relevant products with CLIP Scores
```

## 📈 Results

| Metric | Value |
|--------|-------|
| **Train CLIP Score** | 33.98 |
| **Test CLIP Score** | 33.61 |
| **Train Loss** | 0.0335 |
| **Test Loss** | 0.0719 |
| **Accuracy** | ~93% |

### Training Graphs

![Training History](checkpoints/training_history.png)

**Conclusions:**
- Target (CLIP Score > 30) achieved
- No overfitting
- Augmentation is effective
- Model is ready for use

## 🔧 API Functions

### Product Search
```python
from src.search import search_products, load_embeddings

# Loading data
data = load_embeddings('data/processed/image_embeddings.npz')

# Search
results = search_products(
    query="red summer dress",
    model=model,
    processor=processor,
    image_embeddings=data['embeddings'],
    image_names=data['image_names'],
    dataframe=data['dataframe'],
    top_k=5
)
```

### Model Loading
```python
from transformers import CLIPModel, CLIPProcessor
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Dataset

**Fashion Product Images Dataset**
- Source: Kaggle
- Size: ~44k images
- Categories: clothing, footwear, accessories
- Format: JPG images + CSV with metadata

**CSV Structure:**
- `image`: file name
- `description`: product description
- `display name`: product name
- `category`: category

## Improvements

Implemented optimizations:
- Image augmentation (crop, flip, color jitter, rotation)
- Learning Rate Scheduling (Cosine Annealing)
- Concatenation of title and description (`full_text`)

Possible improvements:
- Hard Negative Mining for cases where the model confuses dresses and nightgowns or short skirts and shorts
- Freezing early layers
- Expanding the dataset
- Using a more powerful model

## Technologies

- **PyTorch** — training framework
- **Transformers** — CLIP model from HuggingFace
- **NumPy** — working with vectors
- **Pandas** — data processing
- **Pillow** — image processing
- **Matplotlib** — visualization
