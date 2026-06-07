# Spam Email Detection

Comparative study of three classifiers — Logistic Regression, XGBoost, and DistilBERT — for binary spam/ham email classification.

## Results

| Model               | Accuracy | Precision | Recall | F1     | FPR    | FNR    | ROC-AUC | PR-AUC |
|---------------------|----------|-----------|--------|--------|--------|--------|---------|--------|
| Logistic Regression | 0.9696   | 0.9599    | 0.9873 | 0.9734 | 0.0534 | 0.0127 | 0.9961  | 0.9968 |
| XGBoost             | 0.9678   | 0.9612    | 0.9825 | 0.9717 | 0.0513 | 0.0175 | 0.9948  | 0.9958 |
| DistilBERT          | **0.9794** | **0.9795** | 0.9841 | **0.9818** | **0.0267** | 0.0159 | **0.9975** | **0.9980** |

DistilBERT achieves the best overall performance. Logistic Regression has the lowest false-negative rate (0.0127), meaning it misses the fewest spam emails.

## Dataset

The dataset combines four sources:

| Directory       | Class | Source                              |
|-----------------|-------|-------------------------------------|
| `spam/`         | spam  | SpamAssassin public corpus          |
| `spam_2/`       | spam  | SpamAssassin public corpus          |
| `hard_ham/`     | ham   | SpamAssassin public corpus          |
| `easy_ham_2/`   | ham   | SpamAssassin public corpus          |
| `phishing-2025` | spam  | Phishing email mbox (2025 dataset)  |

Emails are deduplicated before training. The final split uses 70% train / 30% test with stratification.

## Project Structure

```
.
├── data_processing.py      # Parse SpamAssassin .eml files into data.pickle
├── phishing_processing.py  # Parse phishing mbox and merge into data.pickle
├── kostur.py               # Train all three models and evaluate them
├── praksa.py               # Quick dataset inspection utility
├── data.pickle             # Serialised dataset (generated, not tracked in git)
├── results_table.tex       # LaTeX results table (generated)
├── output_koda.txt         # Sample run output
├── hard_ham/               # Ham email corpus
├── easy_ham_2/             # Ham email corpus
├── spam/                   # Spam email corpus
├── spam_2/                 # Spam email corpus
└── phishing-2025           # Phishing mbox file
```

## Setup

**Prerequisites:** Python 3.10+

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux / macOS
pip install scikit-learn xgboost transformers torch beautifulsoup4
```

## Usage

Run the scripts in order:

```bash
# 1. Parse SpamAssassin emails
python data_processing.py

# 2. Parse and merge phishing emails (requires data.pickle from step 1)
python phishing_processing.py

# 3. Train and evaluate all models
python kostur.py
```

`kostur.py` prints per-model classification reports, a comparison table, and writes `results_table.tex`.

## Models

**Logistic Regression** and **XGBoost** both use a TF-IDF pipeline (unigrams + bigrams, top 20,000 features, English stop words removed).

**DistilBERT** fine-tunes `distilbert-base-uncased` for sequence classification with a linear learning-rate schedule over 3 epochs (batch size 16, max token length 256). GPU is used automatically if available.
