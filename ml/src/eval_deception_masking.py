import json
import os
import re
from datetime import datetime
from pathlib import Path

# Matplotlib cache setup if needed
mpl_cache = Path("ml/reports/.matplotlib")
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_cache.resolve())

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ml.src.data.cleaning import clean_text
from ml.src.features.stylometric import StylometricExtractor, TextLengthExtractor

DATA_PATH = Path("data/processed/ott_deception_5fold.csv")
REPORT_PATH = Path("ml/reports/deception_masking_ablation_v2.json")
SEED = 42
TOKEN_PATTERN = r"(?u)\b\w+\b"

# Entity terms: 20 Chicago hotels + artifacts + landmarks + platforms
ENTITY_TERMS = [
    # Multi-word terms first for proper greedy matching
    "hard rock",
    "magnificent mile",
    "michigan ave",
    "michigan avenue",
    "palmer house",
    "homewood suites",
    "chicago",
    "hilton",
    "hyatt",
    "omni",
    "priceline",
    "expedia",
    "tripadvisor",
    "sofitel",
    "intercontinental",
    "affinia",
    "allegro",
    "amalfi",
    "ambassador",
    "conrad",
    "fairmont",
    "hardrock",
    "homewood",
    "james",
    "knickerbocker",
    "monaco",
    "palmer",
    "sheraton",
    "swissotel",
    "talbott",
]

# Regex for case-insensitive replacement with boundary checking
sorted_terms = sorted(ENTITY_TERMS, key=len, reverse=True)
pattern_str = r"\b(?:" + "|".join(re.escape(t) for t in sorted_terms) + r")\b"
ENTITY_REGEX = re.compile(pattern_str, re.IGNORECASE)

def mask_entities(text: str) -> str:
    return ENTITY_REGEX.sub("[ENTITY]", text)

def run_5fold_cv(texts, labels, folds, models_dict):
    results = {}
    for m_name, model_fn in models_dict.items():
        fold_prec = []
        fold_rec = []
        fold_f1 = []
        fold_macro_f1 = []
        fold_auc = []

        oof_preds = np.zeros(len(texts))
        oof_probs = np.zeros(len(texts))

        for fold in range(5):
            tr_idx = np.where(folds != fold)[0]
            te_idx = np.where(folds == fold)[0]

            X_tr, y_tr = texts.iloc[tr_idx], labels[tr_idx]
            X_te, y_te = texts.iloc[te_idx], labels[te_idx]

            pipe = model_fn()
            pipe.fit(X_tr, y_tr)

            preds = pipe.predict(X_te)
            probs = pipe.predict_proba(X_te)[:, 1]

            oof_preds[te_idx] = preds
            oof_probs[te_idx] = probs

            p = precision_score(y_te, preds, pos_label=1, zero_division=0)
            r = recall_score(y_te, preds, pos_label=1, zero_division=0)
            f1 = f1_score(y_te, preds, pos_label=1, zero_division=0)
            macro = f1_score(y_te, preds, average="macro", zero_division=0)
            try:
                auc = roc_auc_score(y_te, probs)
            except Exception:
                auc = None

            fold_prec.append(p)
            fold_rec.append(r)
            fold_f1.append(f1)
            fold_macro_f1.append(macro)
            if auc is not None:
                fold_auc.append(auc)

        results[m_name] = {
            "macro_f1_mean": float(np.mean(fold_macro_f1)),
            "macro_f1_std": float(np.std(fold_macro_f1)),
            "precision_mean": float(np.mean(fold_prec)),
            "precision_std": float(np.std(fold_prec)),
            "recall_mean": float(np.mean(fold_rec)),
            "recall_std": float(np.std(fold_rec)),
            "f1_mean": float(np.mean(fold_f1)),
            "f1_std": float(np.std(fold_f1)),
            "roc_auc_mean": float(np.mean(fold_auc)) if fold_auc else None,
        }
    return results

def get_models():
    return {
        "tfidf_lr": lambda: Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True, min_df=2, token_pattern=TOKEN_PATTERN)),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
        ]),
        "tfidf_svc": lambda: Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True, min_df=2, token_pattern=TOKEN_PATTERN)),
            ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, dual=False, random_state=SEED), method="sigmoid", cv=3)),
        ]),
        "stylometric_lr": lambda: Pipeline([
            ("style", StylometricExtractor()),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
        ]),
        "length_only": lambda: Pipeline([
            ("len", TextLengthExtractor()),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
        ]),
    }

def main():
    print("=" * 80)
    print("ITEM 5: DECEPTION ENTITY-MASKING ABLATION (Ott et al. 5-Fold CV)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    labels = (df["deception_label"] == "deceptive").astype(int).values
    folds = df["deception_cv_fold"].values

    # Clean text
    print("Applying clean_text()...")
    df["clean_text"] = df["text"].apply(clean_text)

    # Mask entities
    print(f"Masking {len(ENTITY_TERMS)} entity terms with '[ENTITY]'...")
    df["masked_text"] = df["clean_text"].apply(mask_entities)

    # Sample comparison
    sample_idx = 0
    print("\n--- Masking Sample (First row) ---")
    print("Original Cleaned:", df.loc[sample_idx, "clean_text"][:200])
    print("Entity Masked:   ", df.loc[sample_idx, "masked_text"][:200])

    # 1. Run Unmasked v2
    print("\nRunning 5-fold CV on Unmasked Cleaned Text...")
    unmasked_results = run_5fold_cv(df["clean_text"], labels, folds, get_models())

    # 2. Run Masked v2
    print("\nRunning 5-fold CV on Entity-Masked Text...")
    masked_results = run_5fold_cv(df["masked_text"], labels, folds, get_models())

    # Build comparison report
    comparison = {}
    for m in unmasked_results.keys():
        diff_f1 = masked_results[m]["macro_f1_mean"] - unmasked_results[m]["macro_f1_mean"]
        diff_auc = (masked_results[m]["roc_auc_mean"] - unmasked_results[m]["roc_auc_mean"]) if unmasked_results[m]["roc_auc_mean"] else None
        comparison[m] = {
            "unmasked": unmasked_results[m],
            "masked": masked_results[m],
            "delta_macro_f1": float(diff_f1),
            "delta_roc_auc": float(diff_auc) if diff_auc is not None else None,
        }

    full_report = {
        "timestamp": datetime.now().isoformat(),
        "entity_terms": ENTITY_TERMS,
        "token_pattern": TOKEN_PATTERN,
        "results": comparison,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nSaved ablation report to: {REPORT_PATH.resolve()}")

    # Print Comparison Table
    print("\n" + "=" * 80)
    print("DECEPTION ENTITY-MASKING ABLATION: 5-FOLD CV COMPARISON")
    print("=" * 80)
    rows = []
    for m, c in comparison.items():
        un = c["unmasked"]
        ma = c["masked"]
        rows.append({
            "Model": m,
            "Unmasked Macro F1": f"{un['macro_f1_mean']:.4f} ± {un['macro_f1_std']:.4f}",
            "Masked Macro F1": f"{ma['macro_f1_mean']:.4f} ± {ma['macro_f1_std']:.4f}",
            "Delta F1": f"{c['delta_macro_f1']:+.4f}",
            "Unmasked AUC": f"{un['roc_auc_mean']:.4f}" if un['roc_auc_mean'] else "N/A",
            "Masked AUC": f"{ma['roc_auc_mean']:.4f}" if ma['roc_auc_mean'] else "N/A",
            "Delta AUC": f"{c['delta_roc_auc']:+.4f}" if c['delta_roc_auc'] else "N/A",
        })

    table_df = pd.DataFrame(rows)
    print(table_df.to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    main()
