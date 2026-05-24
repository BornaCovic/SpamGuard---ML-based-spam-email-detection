import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             average_precision_score, classification_report)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


with open("data.pickle", "rb") as f:
    data_dict = pickle.load(f)

spam_email_contents = data_dict["spam"]
ham_email_contents = data_dict["ham"]

spam_y = np.full(len(spam_email_contents), 1)
ham_y = np.full(len(ham_email_contents), 0)
y = np.concatenate([spam_y, ham_y])

X_train, X_test, y_train, y_test = train_test_split(spam_email_contents + ham_email_contents, y, test_size=0.3, random_state=69, stratify=y) 

results = []

METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1", "FPR", "FNR", "ROC_AUC", "PR_AUC"]
LOWER_BETTER = {"FPR", "FNR"}


def evaluate(name, y_true, y_pred, y_score=None, verbose=True):
    """Compute spam-detection metrics, print a per-model report, return a dict row.

    Positive class = spam (1). A false positive is a legitimate email wrongly
    flagged as spam, so FPR is the operationally critical metric here.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "FPR": fp / (fp + tn) if (fp + tn) else 0.0,
        "FNR": fn / (fn + tp) if (fn + tp) else 0.0,
        "ROC_AUC": roc_auc_score(y_true, y_score) if y_score is not None else float("nan"),
        "PR_AUC": average_precision_score(y_true, y_score) if y_score is not None else float("nan"),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    if verbose:
        print(f"\n===== {name} =====")
        print(f"Confusion matrix:  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
        print(classification_report(y_true, y_pred, target_names=["ham", "spam"], digits=4))
    return row


def print_comparison(results):
    header = f"{'Model':<22}" + "".join(f"{c:>10}" for c in METRIC_COLS)
    bar = "=" * len(header)
    print(f"\n{bar}\nMODEL COMPARISON (held-out test set)\n{bar}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['Model']:<22}" + "".join(f"{r[c]:>10.4f}" for c in METRIC_COLS))
    print(bar)


def to_latex(results, path="results_table.tex",
             caption=("Classifier performance on the held-out test set. "
                      "Best value per column in bold; FPR and FNR are lower-is-better."),
             label="tab:results"):
    headers = {"ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC"}
    best = {c: (min if c in LOWER_BETTER else max)(r[c] for r in results) for c in METRIC_COLS}
    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{l" + "c" * len(METRIC_COLS) + "}", r"\toprule",
        "Model & " + " & ".join(headers.get(c, c) for c in METRIC_COLS) + r" \\", r"\midrule",
    ]
    for r in results:
        cells = []
        for c in METRIC_COLS:
            txt = f"{r[c]:.4f}"
            if abs(r[c] - best[c]) < 1e-9:
                txt = rf"\textbf{{{txt}}}"
            cells.append(txt)
        lines.append(r["Model"] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    latex = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(latex + "\n")
    print(f"\n% --- LaTeX table written to {path} (requires \\usepackage{{booktabs}}) ---")
    print(latex)
    return latex


model_log_reg = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2)
    )),
    ("clf", LogisticRegression(max_iter=2000))
])
model_log_reg.fit(X_train, y_train)
y_pred = model_log_reg.predict(X_test)
y_score = model_log_reg.predict_proba(X_test)[:, 1]
accuracy_log_reg = accuracy_score(y_test, y_pred)
print(f"Logistic regression accuracy: {accuracy_log_reg}")
print(model_log_reg.predict(["Click this link for free Nigerian prince money: I need your credit card info!"]))
results.append(evaluate("Logistic Regression", y_test, y_pred, y_score))



xgb_model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))),
    ("clf", XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        tree_method="hist", eval_metric="logloss",
        n_jobs=-1, random_state=69,
    )),
])
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
y_score = xgb_model.predict_proba(X_test)[:, 1]
accuracy_xgb = accuracy_score(y_test, y_pred)
print(f"XGBoost accuracy: {accuracy_xgb}")
print(xgb_model.predict(["Click this link for free Nigerian prince money: I need your credit card info!"]))
results.append(evaluate("XGBoost", y_test, y_pred, y_score))


from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, get_scheduler
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

train_dataset = EmailDataset(X_train, y_train, tokenizer)
test_dataset = EmailDataset(X_test, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training DistilBERT on: {device}")
bert_model.to(device)

optimizer = AdamW(bert_model.parameters(), lr=2e-5)
num_epochs = 3
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_epochs * len(train_loader),
)

bert_model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = bert_model(**batch)
        outputs.loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch + 1}/{num_epochs} complete")

bert_model.eval()
all_preds = []
all_scores = []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = bert_model(**batch).logits
        all_scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

accuracy_bert = accuracy_score(y_test, all_preds)
print(f"DistilBERT accuracy: {accuracy_bert}")
results.append(evaluate("DistilBERT", y_test, all_preds, all_scores))

print_comparison(results)
to_latex(results)