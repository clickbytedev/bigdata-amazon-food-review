# Customer Review Analysis and MongoDB Practice
**Dataset:** Amazon Fine Food Reviews (`Reviews_withURL.csv`)

---

## Project Overview

This project analyzes ~568,000 Amazon fine food product reviews sourced from [Kaggle – Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews). The dataset covers reviews submitted between 1999 and 2012 across hundreds of food products.

All data is stored and queried from **MongoDB**. The analysis is split into four modules:
1. **Helpfulness Score Analysis** – Compute `HelpfulnessRatio` and identify what makes a review useful.
2. **Sentiment Trend Over Time** – Track brand sentiment using VADER and DeBERTa v3 across 13 years.
3. **Food Safety Detection** – Flag explicit and implicit safety/health hazard reports using regex + semantic search (MiniLM + DeBERTa NLI).
4. **Business Recommendations** – Derive actionable insights for marketing, product, and regulatory teams.

---

## Last Run Specification

| Component | Details |
|-----------|---------|
| **GPU** | NVIDIA Quadro RTX 4000 |
| **GPU Architecture** | Turing (2018) |
| **CUDA Cores** | 2,304 |
| **GPU Memory** | 8 GB GDDR6 (8,192 MiB) |
| **Memory Bandwidth** | 416 GB/s |
| **TDP / Power Cap** | 125W |
| **Driver Version** | 595.97 (WDDM – Windows) |
| **CUDA Version** | 13.2 |
| **Display Mode** | Off (headless / compute mode) |

> The Quadro RTX 4000 is a professional-grade workstation GPU with dedicated Tensor Cores (2nd gen) and RT Cores. Tensor Cores accelerate transformer inference (DeBERTa, MiniLM), reducing DeBERTa inference on 10,000 samples from ~30 min (CPU) to ~3–5 min. Driver 595.97 with CUDA 13.2 confirms the GPU is fully operational.

---

## Dataset Columns

| Column | Description |
|---|---|
| `Id` | Row ID |
| `ProductId` | Unique product identifier |
| `UserId` | Unique user identifier |
| `ProfileName` | User display name |
| `HelpfulnessNumerator` | # users who found review helpful |
| `HelpfulnessDenominator` | # users who voted on helpfulness |
| `Score` | Star rating (1–5) |
| `Time` | Unix timestamp of review |
| `Summary` | Short review title |
| `Text` | Full review text |
| `ProductURL` | Amazon product URL |

---

## Setup & Environment

### 1. Install Miniconda

Download and install **Miniconda** from: https://docs.conda.io/en/latest/miniconda.html

Miniconda is a minimal version of Anaconda — it includes just Python, conda, and their dependencies. It lets you create isolated environments so project libraries never conflict with each other.

---

### 2. Create a New Conda Environment

Open **Anaconda Prompt** (Windows) or a terminal and run:

```bash
conda create -n bigdata python=3.10 -y
```

This creates a new isolated environment named `bigdata` running Python 3.10.

---

### 3. Activate the Environment

```bash
conda activate bigdata
```

You should see `(bigdata)` appear at the start of your terminal prompt. All installs below go into this environment only.

---

### 4. Install PyTorch with CUDA Support

This machine runs **CUDA 13.2** (Driver 595.97). Install the latest stable PyTorch with CUDA 12.8 (the highest CUDA toolkit currently supported by PyTorch stable builds — it is forward-compatible with driver CUDA 13.2):

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia -y
```

> If you do not have an NVIDIA GPU, install the CPU-only version instead:
> ```bash
> conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
> ```

---

### 5. Install All Other Dependencies

```bash
conda install -c conda-forge pandas numpy pymongo matplotlib seaborn wordcloud tqdm openpyxl -y
```

```bash
pip install nltk vaderSentiment transformers accelerate scikit-learn sentence-transformers
```

---

### 6. Download Required NLP Data

```bash
python -m nltk.downloader stopwords wordnet punkt
```

---

### Full Library List

| Library | Install Source | Purpose |
|---------|---------------|---------|
| `pandas` | conda-forge | Data manipulation and analysis |
| `numpy` | conda-forge | Numerical computing |
| `pymongo` | conda-forge | MongoDB Python driver |
| `matplotlib` | conda-forge | Plotting and charts |
| `seaborn` | conda-forge | Statistical data visualization |
| `wordcloud` | conda-forge | Word cloud generation |
| `tqdm` | conda-forge | Progress bars for long loops |
| `openpyxl` | conda-forge | Excel file export engine |
| `torch` (PyTorch) | pytorch / nvidia | GPU-accelerated deep learning |
| `torchvision` | pytorch / nvidia | Required by PyTorch |
| `torchaudio` | pytorch / nvidia | Required by PyTorch |
| `nltk` | pip | NLP tokenization and stopwords |
| `vaderSentiment` | pip | Rule-based sentiment analysis |
| `transformers` | pip | Hugging Face transformer models (DeBERTa) |
| `accelerate` | pip | Hugging Face GPU acceleration |
| `scikit-learn` | pip | Accuracy scores and confusion matrices |
| `sentence-transformers` | pip | Sentence embeddings (MiniLM semantic search) |

---

## MongoDB Setup

**Ensure MongoDB is running locally on `mongodb://localhost:27017`**

Download MongoDB Community Server from: https://www.mongodb.com/try/download/community

The notebook auto-imports the CSV into MongoDB on the first run. Optionally, you can import manually:

```bash
mongoimport --db amazon_reviews --collection reviews --type csv --headerline --file Reviews_withURL.csv
```

---

## How to Run

1. **Clone or download** this project folder.
2. **Place `Reviews_withURL.csv`** in the project root (same folder as `analysis.ipynb`).
3. **Start MongoDB** (ensure it is running on `localhost:27017`).
4. **Activate the conda environment:**
   ```bash
   conda activate bigdata
   ```
5. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```
   Or open `analysis.ipynb` directly in VS Code with the Jupyter extension.
6. **Run all cells top to bottom.** Each section builds on the previous one — do not skip sections.

> **Note:** Sections using DeBERTa or MiniLM will automatically use the GPU if CUDA is available. First run will download model weights (~80 MB for MiniLM, ~300 MB for DeBERTa).

---

## File Structure

```
project1_nlp_amazon_food_review/
├── Reviews_withURL.csv              # Raw dataset (~568K reviews)
├── README.md                        # This file
├── analysis.ipynb                   # Main Jupyter notebook (all modules)
└── outputs/
    ├── section3_sentiment.xlsx      # VADER + DeBERTa sentiment results
    └── section4_food_safety.xlsx    # Food safety detection results
```

---

## Business Recommendations (Summary)

### [A] Review Helpfulness Ranking

- **42,887 reviews (7.5%)** have helpfulness ratio ≥ 0.80 with ≥ 5 votes
- 5-star reviews consistently receive higher helpfulness scores
- **Action:** Promote high-helpfulness reviews to *Top Reviews*; apply a "Verified + Helpful" badge to reviews with ratio ≥ 0.8 and ≥ 10 votes

---

### [B] Negative Sentiment Monitoring & Product Intervention

- **55,363 reviews (9.7%)** classified as VADER-Negative; **501,357 (88.2%)** as VADER-Positive
- VADER accuracy: **69.1%** | DeBERTa v3 accuracy: **81.7%**
- **Action:** Deploy VADER for real-time dashboards; use DeBERTa v3 for monthly deep-dives; flag product categories exceeding **30% negative sentiment** for immediate review

---

### [C] Food Safety Alerting & Supplier Compliance

**81,626 reviews (14.36%)** flagged for food safety concerns:

| Severity | Category | Count |
|----------|----------|-------|
| 🔴 High | Contamination (mold, insects, foreign objects) | 65,156 |
| 🔴 High | Illness (food poisoning, sickness reports) | 7,491 |
| 🔴 High | Allergen (undisclosed/mislabeled allergens) | 4,414 |
| 🟡 Medium | Spoilage (expired, rancid, bad smell) | 12,726 |
| 🟢 Low | Quality Defect (mislabeled, broken packaging) | 615 |

- **Action:** Trigger 24-hour supplier alerts for all High-severity flags; initiate supplier audits for ≥ 10 Spoilage/Contamination cases in a 30-day window; route Allergen and Illness reports to Regulatory Affairs for FDA compliance review

---

### [D] Recommended Production Pipeline

| Stage | Tool | Scope | Cadence |
|-------|------|-------|---------|
| Stage 1 – Regex | Keyword scanner (Section 4a–c) | Full 568K+ reviews | Daily |
| Stage 2 – MiniLM | Semantic search (Section 4e) | Full 568K+ reviews | Daily |
| Stage 3 – DeBERTa | NLI precision filter (Section 4h) | MiniLM candidates only | Weekly |

Export results weekly to Excel (Section 4i) for stakeholder reporting.
