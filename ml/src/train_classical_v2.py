import json
import os
from datetime import datetime
from pathlib import Path

# Matplotlib cache configuration
mpl_cache = Path("ml/reports/.matplotlib")
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_cache.resolve())

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ml.src.data.cleaning import clean_text
from ml.src.features.stylometric import StylometricExtractor, TextLengthExtractor

DATA_DIR = Path("data/processed")
REPORTS_DIR = Path("ml/reports")
MODELS_DIR = Path("api/models")
V1_METRICS_PATH = REPORTS_DIR / "classical_metrics.json"
V2_METRICS_PATH = REPORTS_DIR / "classical_metrics_v2.json"
SEED = 42

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TOKEN_PATTERN = r"(?u)\b\w+\b"

# -----------------------------------------------------------------------------
# Metric helper
# -----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    prec_0 = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec_0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_0 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    prec_1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()

    auc = None
    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = None

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "roc_auc": auc,
        "class_0": {"precision": float(prec_0), "recall": float(rec_0), "f1": float(f1_0)},
        "class_1": {"precision": float(prec_1), "recall": float(rec_1), "f1": float(f1_1)},
        "confusion_matrix": cm,
    }


def get_feature_importances(pipeline, model_name: str, top_n: int = 20) -> dict:
    """Extract top positive and negative features for linear models."""
    try:
        step_names = list(pipeline.named_steps.keys())
        clf = pipeline.named_steps[step_names[-1]]

        if isinstance(clf, CalibratedClassifierCV):
            coef_list = []
            for cal_clf in clf.calibrated_classifiers_:
                base = getattr(cal_clf, "estimator", cal_clf)
                if hasattr(base, "coef_"):
                    coef_list.append(base.coef_)
            if coef_list:
                coefs = np.mean(coef_list, axis=0).ravel()
            else:
                return {"top_negative": [], "top_positive": []}
        elif hasattr(clf, "coef_"):
            coefs = clf.coef_.ravel()
        else:
            return {"top_negative": [], "top_positive": []}

        feat_extractor = pipeline.named_steps[step_names[0]]
        if hasattr(feat_extractor, "get_feature_names_out"):
            feat_names = np.array(feat_extractor.get_feature_names_out())
        else:
            feat_names = np.array([f"f_{i}" for i in range(len(coefs))])

        sorted_indices = np.argsort(coefs)
        top_neg_idx = sorted_indices[:top_n]
        top_pos_idx = sorted_indices[-top_n:][::-1]

        return {
            "top_negative": [{"feature": str(feat_names[i]), "coef": float(coefs[i])} for i in top_neg_idx],
            "top_positive": [{"feature": str(feat_names[i]), "coef": float(coefs[i])} for i in top_pos_idx],
        }
    except Exception as e:
        return {"error": str(e), "top_negative": [], "top_positive": []}


# -----------------------------------------------------------------------------
# Authorship Training & Evaluation (v2)
# -----------------------------------------------------------------------------
def run_authorship():
    print("=" * 80)
    print("1. RUNNING AUTHORSHIP EXPERIMENTS (v2: Cleaned Text, 1-char token_pattern, Val-only Selection)")
    print("=" * 80)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    ott_df = pd.read_csv(DATA_DIR / "ott_deception_5fold.csv")

    # Apply shared text cleaning
    print("Applying clean_text() to train, val, test, and ott splits...")
    train_df["text_clean"] = train_df["text"].apply(clean_text)
    val_df["text_clean"] = val_df["text"].apply(clean_text)
    test_df["text_clean"] = test_df["text"].apply(clean_text)
    ott_df["text_clean"] = ott_df["text"].apply(clean_text)

    # Exclude Ott from main authorship training/testing
    train_main = train_df[train_df["source_dataset"].isin(["salminen", "raid"])].copy()
    val_main = val_df[val_df["source_dataset"].isin(["salminen", "raid"])].copy()
    test_main = test_df[test_df["source_dataset"].isin(["salminen", "raid"])].copy()

    y_train = (train_main["authorship_label"] == "ai").astype(int).values
    y_val = (val_main["authorship_label"] == "ai").astype(int).values
    y_test_main = (test_main["authorship_label"] == "ai").astype(int).values

    test_sal = test_main[test_main["source_dataset"] == "salminen"].copy()
    y_test_sal = (test_sal["authorship_label"] == "ai").astype(int).values

    test_raid = test_main[test_main["source_dataset"] == "raid"].copy()
    y_test_raid = (test_raid["authorship_label"] == "ai").astype(int).values

    print(f"Authorship Train size: {len(train_main):,} | Val size: {len(val_main):,}")
    print(f"Test Main: {len(test_main):,} (Salminen: {len(test_sal):,}, RAID: {len(test_raid):,})")

    # Define model pipelines with token_pattern=r"(?u)\b\w+\b"
    models = {
        "tfidf_lr": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True, min_df=2, token_pattern=TOKEN_PATTERN)),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
        ]),
        "tfidf_svc": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True, min_df=2, token_pattern=TOKEN_PATTERN)),
            ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, dual=False, random_state=SEED), method="sigmoid", cv=3)),
        ]),
        "stylometric_lr": Pipeline([
            ("style", StylometricExtractor()),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
        ]),
        "length_only": Pipeline([
            ("len", TextLengthExtractor()),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
        ]),
    }

    results = {}
    best_model_name = None
    best_val_macro_f1 = -1.0

    plt.figure(figsize=(9, 7))

    for m_name, pipe in models.items():
        print(f"\nFitting Authorship Model: {m_name}...")
        pipe.fit(train_main["text_clean"], y_train)

        # Val score
        val_pred = pipe.predict(val_main["text_clean"])
        val_prob = pipe.predict_proba(val_main["text_clean"])[:, 1]
        val_metrics = compute_metrics(y_val, val_pred, val_prob)

        # Test Main
        test_pred = pipe.predict(test_main["text_clean"])
        test_prob = pipe.predict_proba(test_main["text_clean"])[:, 1]
        test_metrics = compute_metrics(y_test_main, test_pred, test_prob)

        # Test Salminen
        sal_pred = pipe.predict(test_sal["text_clean"])
        sal_prob = pipe.predict_proba(test_sal["text_clean"])[:, 1]
        sal_metrics = compute_metrics(y_test_sal, sal_pred, sal_prob)

        # Test RAID
        raid_pred = pipe.predict(test_raid["text_clean"])
        raid_prob = pipe.predict_proba(test_raid["text_clean"])[:, 1]
        raid_metrics = compute_metrics(y_test_raid, raid_pred, raid_prob)

        # Feature importances
        top_feats = get_feature_importances(pipe, m_name)

        results[m_name] = {
            "validation": val_metrics,
            "test_main": test_metrics,
            "test_salminen": sal_metrics,
            "test_raid": raid_metrics,
            "top_features": top_feats,
        }

        print(f"  Val Macro F1: {val_metrics['macro_f1']:.4f} | Test Main Macro F1: {test_metrics['macro_f1']:.4f}")
        print(f"  Salminen Test F1: {sal_metrics['macro_f1']:.4f} | RAID Test F1: {raid_metrics['macro_f1']:.4f}")

        # Strictly val-only selection (Fix from line 217 in v1)
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_model_name = m_name

        # Plot ROC curve for test_main
        if test_prob is not None:
            fpr, tpr, _ = roc_curve(y_test_main, test_prob)
            plt.plot(fpr, tpr, label=f"{m_name} (AUC={test_metrics['roc_auc']:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate (AI predicted as Human)")
    plt.ylabel("True Positive Rate (AI predicted as AI)")
    plt.title("Authorship ROC Curves v2 (Test Set: Salminen + RAID)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "authorship_roc_curves_v2.png", dpi=300)
    plt.close()

    print(f"\n[Val-Only Model Selection Result]")
    print(f"Selected best model: '{best_model_name}' with Val Macro F1 = {best_val_macro_f1:.4f}")

    # -------------------------------------------------------------------------
    # Cross-dataset Generalization Experiments
    # -------------------------------------------------------------------------
    print("\n--- Running Cross-Dataset Generalization Experiments (v2) ---")
    cross_results = {}

    # Train on Salminen only -> Test on RAID test
    sal_only_train = train_main[train_main["source_dataset"] == "salminen"].copy()
    y_sal_train = (sal_only_train["authorship_label"] == "ai").astype(int).values
    cross_sal_model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True, min_df=2, token_pattern=TOKEN_PATTERN)),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
    ])
    cross_sal_model.fit(sal_only_train["text_clean"], y_sal_train)
    sal2raid_pred = cross_sal_model.predict(test_raid["text_clean"])
    sal2raid_prob = cross_sal_model.predict_proba(test_raid["text_clean"])[:, 1]
    cross_results["train_salminen_test_raid"] = compute_metrics(y_test_raid, sal2raid_pred, sal2raid_prob)

    # Train on RAID only -> Test on Salminen test
    raid_only_train = train_main[train_main["source_dataset"] == "raid"].copy()
    y_raid_train = (raid_only_train["authorship_label"] == "ai").astype(int).values
    cross_raid_model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True, min_df=2, token_pattern=TOKEN_PATTERN)),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
    ])
    cross_raid_model.fit(raid_only_train["text_clean"], y_raid_train)
    raid2sal_pred = cross_raid_model.predict(test_sal["text_clean"])
    raid2sal_prob = cross_raid_model.predict_proba(test_sal["text_clean"])[:, 1]
    cross_results["train_raid_test_salminen"] = compute_metrics(y_test_sal, raid2sal_pred, raid2sal_prob)

    print(f"Salminen-only -> RAID Test Macro F1: {cross_results['train_salminen_test_raid']['macro_f1']:.4f}")
    print(f"RAID-only -> Salminen Test Macro F1: {cross_results['train_raid_test_salminen']['macro_f1']:.4f}")

    # -------------------------------------------------------------------------
    # Balanced-domain Variant (Cap Salminen at 4k/class)
    # -------------------------------------------------------------------------
    print("\n--- Training Balanced-Domain Variant (Salminen capped at 4k/class, v2) ---")
    sal_h = sal_only_train[sal_only_train["authorship_label"] == "human"].sample(n=4000, random_state=SEED)
    sal_a = sal_only_train[sal_only_train["authorship_label"] == "ai"].sample(n=4000, random_state=SEED)
    balanced_train_df = pd.concat([sal_h, sal_a, raid_only_train], ignore_index=True)
    y_balanced_train = (balanced_train_df["authorship_label"] == "ai").astype(int).values

    balanced_model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True, min_df=2, token_pattern=TOKEN_PATTERN)),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
    ])
    balanced_model.fit(balanced_train_df["text_clean"], y_balanced_train)

    bal_main_pred = balanced_model.predict(test_main["text_clean"])
    bal_main_prob = balanced_model.predict_proba(test_main["text_clean"])[:, 1]
    bal_main_metrics = compute_metrics(y_test_main, bal_main_pred, bal_main_prob)

    bal_sal_pred = balanced_model.predict(test_sal["text_clean"])
    bal_sal_metrics = compute_metrics(y_test_sal, bal_sal_pred)

    bal_raid_pred = balanced_model.predict(test_raid["text_clean"])
    bal_raid_metrics = compute_metrics(y_test_raid, bal_raid_pred)

    results["tfidf_lr_balanced_domain"] = {
        "test_main": bal_main_metrics,
        "test_salminen": bal_sal_metrics,
        "test_raid": bal_raid_metrics,
    }
    print(f"Balanced Variant Main Test F1: {bal_main_metrics['macro_f1']:.4f} (Salminen: {bal_sal_metrics['macro_f1']:.4f}, RAID: {bal_raid_metrics['macro_f1']:.4f})")

    # -------------------------------------------------------------------------
    # Best Model Evaluation on Ott Rows
    # -------------------------------------------------------------------------
    print("\n--- Running Best Authorship Model on Ott Rows (v2) ---")
    best_pipe = models[best_model_name]
    ott_preds = best_pipe.predict(ott_df["text_clean"])
    ott_probs = best_pipe.predict_proba(ott_df["text_clean"])[:, 1]

    ott_df_eval = ott_df.copy()
    ott_df_eval["predicted_ai"] = (ott_preds == 1).astype(int)

    total_ai_share = ott_df_eval["predicted_ai"].mean()
    truthful_ai_share = ott_df_eval[ott_df_eval["deception_label"] == "genuine"]["predicted_ai"].mean()
    deceptive_ai_share = ott_df_eval[ott_df_eval["deception_label"] == "deceptive"]["predicted_ai"].mean()

    ott_independence_report = {
        "best_authorship_model": best_model_name,
        "total_ott_rows": len(ott_df),
        "total_predicted_ai_share": float(total_ai_share),
        "genuine_rows_predicted_ai_share": float(truthful_ai_share),
        "deceptive_rows_predicted_ai_share": float(deceptive_ai_share),
        "note": "Ott is 100% human-authored. As demonstrated by the positive control audit, low AI-predicted share in this unseen domain reflects low sensitivity / domain shift."
    }
    print(f"Overall Ott share predicted as AI: {total_ai_share*100:.2f}%")
    print(f"  Truthful rows predicted as AI: {truthful_ai_share*100:.2f}%")
    print(f"  Deceptive rows predicted as AI: {deceptive_ai_share*100:.2f}%")

    # Save best authorship model (both v2 and default)
    joblib.dump(best_pipe, MODELS_DIR / "authorship_classical_v2.joblib")
    joblib.dump(best_pipe, MODELS_DIR / "authorship_classical.joblib")
    print(f"Saved best authorship model to {MODELS_DIR / 'authorship_classical_v2.joblib'} and {MODELS_DIR / 'authorship_classical.joblib'}")

    # Plot Confusion Matrices v2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_sal_data = np.array(results[best_model_name]["test_salminen"]["confusion_matrix"])
    cm_raid_data = np.array(results[best_model_name]["test_raid"]["confusion_matrix"])
    sns.heatmap(cm_sal_data, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
    axes[0].set_title(f"Authorship v2: Salminen Test ({best_model_name})")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    sns.heatmap(cm_raid_data, annot=True, fmt="d", cmap="Greens", ax=axes[1],
                xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
    axes[1].set_title(f"Authorship v2: RAID Test ({best_model_name})")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "authorship_confusion_matrices_v2.png", dpi=300)
    plt.close()

    return results, cross_results, ott_independence_report, best_model_name, best_pipe


# -----------------------------------------------------------------------------
# Deception Training & 5-Fold Stratified CV (v2)
# -----------------------------------------------------------------------------
def run_deception():
    print("\n" + "=" * 80)
    print("2. RUNNING DECEPTION EXPERIMENTS (v2: 5-Fold Stratified CV on Cleaned Ott et al.)")
    print("=" * 80)

    ott_df = pd.read_csv(DATA_DIR / "ott_deception_5fold.csv")
    ott_df["text_clean"] = ott_df["text"].apply(clean_text)
    y_all = (ott_df["deception_label"] == "deceptive").astype(int).values

    models = {
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

    deception_results = {}
    best_deception_f1 = -1.0
    best_deception_model = None

    for m_name, model_fn in models.items():
        print(f"\nRunning 5-fold CV for Deception Model: {m_name}...")
        fold_metrics = []
        oof_preds = np.zeros(len(ott_df))
        oof_probs = np.zeros(len(ott_df))

        for fold in range(5):
            train_idx = ott_df[ott_df["deception_cv_fold"] != fold].index
            test_idx = ott_df[ott_df["deception_cv_fold"] == fold].index

            X_tr, y_tr = ott_df.loc[train_idx, "text_clean"], y_all[train_idx]
            X_te, y_te = ott_df.loc[test_idx, "text_clean"], y_all[test_idx]

            pipe = model_fn()
            pipe.fit(X_tr, y_tr)

            preds = pipe.predict(X_te)
            probs = pipe.predict_proba(X_te)[:, 1]

            oof_preds[test_idx] = preds
            oof_probs[test_idx] = probs

            m = compute_metrics(y_te, preds, probs)
            fold_metrics.append(m)

        precisions = [m["class_1"]["precision"] for m in fold_metrics]
        recalls = [m["class_1"]["recall"] for m in fold_metrics]
        f1s = [m["class_1"]["f1"] for m in fold_metrics]
        macro_f1s = [m["macro_f1"] for m in fold_metrics]
        aucs = [m["roc_auc"] for m in fold_metrics if m["roc_auc"] is not None]

        mean_macro_f1 = float(np.mean(macro_f1s))
        std_macro_f1 = float(np.std(macro_f1s))

        deception_results[m_name] = {
            "mean_macro_f1": mean_macro_f1,
            "std_macro_f1": std_macro_f1,
            "precision_mean": float(np.mean(precisions)),
            "precision_std": float(np.std(precisions)),
            "recall_mean": float(np.mean(recalls)),
            "recall_std": float(np.std(recalls)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "roc_auc_mean": float(np.mean(aucs)) if aucs else None,
            "overall_oof_metrics": compute_metrics(y_all, oof_preds, oof_probs),
            "fold_details": fold_metrics,
        }

        print(f"  CV Macro F1: {mean_macro_f1:.4f} ± {std_macro_f1:.4f} | AUC: {np.mean(aucs):.4f}")
        print(f"  Deceptive Class F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f} (Prec: {np.mean(precisions):.4f}, Rec: {np.mean(recalls):.4f})")

        if mean_macro_f1 > best_deception_f1:
            best_deception_f1 = mean_macro_f1
            best_deception_model = m_name

    # Fit final production deception model on all 5 folds
    print(f"\nFitting final best deception model ({best_deception_model}) on all cleaned Ott rows...")
    final_deception_pipe = models[best_deception_model]()
    final_deception_pipe.fit(ott_df["text_clean"], y_all)
    joblib.dump(final_deception_pipe, MODELS_DIR / "deception_classical_v2.joblib")
    joblib.dump(final_deception_pipe, MODELS_DIR / "deception_classical.joblib")
    print(f"Saved {MODELS_DIR / 'deception_classical_v2.joblib'} and {MODELS_DIR / 'deception_classical.joblib'}")

    top_deception_feats = get_feature_importances(final_deception_pipe, best_deception_model)
    deception_results["best_model_top_features"] = top_deception_feats

    # Plot Deception Confusion Matrix & ROC Curve v2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    oof_cm = deception_results[best_deception_model]["overall_oof_metrics"]["confusion_matrix"]
    sns.heatmap(oof_cm, annot=True, fmt="d", cmap="Purples", ax=axes[0],
                xticklabels=["Genuine", "Deceptive"], yticklabels=["Genuine", "Deceptive"])
    axes[0].set_title(f"Deception 5-Fold OOF Confusion Matrix v2 ({best_deception_model})")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    best_oof_probs = np.zeros(len(ott_df))
    for fold in range(5):
        tr_idx = ott_df[ott_df["deception_cv_fold"] != fold].index
        te_idx = ott_df[ott_df["deception_cv_fold"] == fold].index
        p = models[best_deception_model]()
        p.fit(ott_df.loc[tr_idx, "text_clean"], y_all[tr_idx])
        best_oof_probs[te_idx] = p.predict_proba(ott_df.loc[te_idx, "text_clean"])[:, 1]

    fpr, tpr, _ = roc_curve(y_all, best_oof_probs)
    axes[1].plot(fpr, tpr, label=f"OOF ROC (AUC = {roc_auc_score(y_all, best_oof_probs):.3f})", color="purple")
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("Deception OOF ROC Curve v2 (Ott et al.)")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "deception_evaluation_v2.png", dpi=300)
    plt.close()

    return deception_results, best_deception_model, final_deception_pipe


# -----------------------------------------------------------------------------
# Main & Reporting
# -----------------------------------------------------------------------------
def main():
    auth_results, cross_results, ott_independence, best_auth_name, best_auth_pipe = run_authorship()
    dec_results, best_dec_name, best_dec_pipe = run_deception()

    # Save comprehensive metrics JSON v2
    all_metrics_v2 = {
        "timestamp": datetime.now().isoformat(),
        "version": "v2",
        "cleaning_applied": {
            "html_entity_unescaping": True,
            "html_tags_stripped": True,
            "unicode_quotes_dashes_normalized": True,
            "chat_preambles_stripped": True,
            "whitespace_collapsed": True,
            "token_pattern": TOKEN_PATTERN,
        },
        "authorship": {
            "best_model": best_auth_name,
            "selection_criterion": "val.csv macro_f1",
            "models": auth_results,
            "cross_dataset": cross_results,
            "ott_held_out_independence_test": ott_independence,
        },
        "deception": {
            "best_model": best_dec_name,
            "models": dec_results,
        },
        "datasets": {
            "train": "data/processed/train.csv",
            "val": "data/processed/val.csv",
            "test": "data/processed/test.csv",
            "ott_cv": "data/processed/ott_deception_5fold.csv",
            "unified": "data/processed/unified_dataset.csv",
        }
    }

    with open(V2_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics_v2, f, indent=2)
    print(f"\nSaved v2 evaluation metrics to: {V2_METRICS_PATH.resolve()}")

    # Save production models metadata JSON v2
    prod_metadata_v2 = {
        "timestamp": datetime.now().isoformat(),
        "version": "v2",
        "authorship_model": {
            "filename": "authorship_classical_v2.joblib",
            "architecture": best_auth_name,
            "val_macro_f1": auth_results[best_auth_name]["validation"]["macro_f1"],
            "test_macro_f1": auth_results[best_auth_name]["test_main"]["macro_f1"],
            "test_roc_auc": auth_results[best_auth_name]["test_main"]["roc_auc"],
            "salminen_test_f1": auth_results[best_auth_name]["test_salminen"]["macro_f1"],
            "raid_test_f1": auth_results[best_auth_name]["test_raid"]["macro_f1"],
            "ott_predicted_ai_share": ott_independence["total_predicted_ai_share"],
        },
        "deception_model": {
            "filename": "deception_classical_v2.joblib",
            "architecture": best_dec_name,
            "cv_macro_f1_mean": dec_results[best_dec_name]["mean_macro_f1"],
            "cv_macro_f1_std": dec_results[best_dec_name]["std_macro_f1"],
            "cv_roc_auc_mean": dec_results[best_dec_name]["roc_auc_mean"],
        },
        "feature_settings": {
            "cleaning": "ml.src.data.cleaning.clean_text",
            "token_pattern": TOKEN_PATTERN,
            "tfidf_ngram_range": [1, 2],
            "tfidf_sublinear_tf": True,
            "min_df": 2,
            "stylometric_features": ["burstiness", "type_token_ratio", "avg_word_length", "punctuation_density", "text_length"],
        },
        "dataset_files": [
            "data/processed/train.csv",
            "data/processed/val.csv",
            "data/processed/test.csv",
            "data/processed/ott_deception_5fold.csv"
        ]
    }

    with open(MODELS_DIR / "classical_metadata_v2.json", "w", encoding="utf-8") as f:
        json.dump(prod_metadata_v2, f, indent=2)
    with open(MODELS_DIR / "classical_metadata.json", "w", encoding="utf-8") as f:
        json.dump(prod_metadata_v2, f, indent=2)
    print(f"Saved production metadata to: {MODELS_DIR / 'classical_metadata_v2.json'}")

    # -------------------------------------------------------------------------
    # Top 20 Features v2
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"TOP 20 FEATURES FOR BEST AUTHORSHIP MODEL v2 ({best_auth_name})")
    print("=" * 80)
    auth_top = auth_results[best_auth_name]["top_features"]
    print("Top 20 Features Pushing Toward HUMAN (Negative Coefs):")
    for item in auth_top.get("top_negative", []):
        print(f"  {item['feature']:<25} : {item['coef']:+.4f}")
    print("\nTop 20 Features Pushing Toward AI (Positive Coefs):")
    for item in auth_top.get("top_positive", []):
        print(f"  {item['feature']:<25} : {item['coef']:+.4f}")

    print("\n" + "=" * 80)
    print(f"TOP 20 FEATURES FOR BEST DECEPTION MODEL v2 ({best_dec_name})")
    print("=" * 80)
    dec_top = dec_results["best_model_top_features"]
    print("Top 20 Features Pushing Toward GENUINE (Negative Coefs):")
    for item in dec_top.get("top_negative", []):
        print(f"  {item['feature']:<25} : {item['coef']:+.4f}")
    print("\nTop 20 Features Pushing Toward DECEPTIVE (Positive Coefs):")
    for item in dec_top.get("top_positive", []):
        print(f"  {item['feature']:<25} : {item['coef']:+.4f}")

    # -------------------------------------------------------------------------
    # PRINT PER-CLASS PRECISION AND RECALL FOR BEST V2 AUTHORSHIP MODEL
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"PER-CLASS PRECISION AND RECALL: BEST V2 AUTHORSHIP MODEL ({best_auth_name})")
    print("=" * 80)
    best_sal = auth_results[best_auth_name]["test_salminen"]
    best_raid = auth_results[best_auth_name]["test_raid"]
    best_main = auth_results[best_auth_name]["test_main"]

    print(f"Salminen Test (N=4,041):")
    print(f"  Human (Class 0): Precision = {best_sal['class_0']['precision']:.4f} | Recall = {best_sal['class_0']['recall']:.4f} | F1 = {best_sal['class_0']['f1']:.4f}")
    print(f"  AI    (Class 1): Precision = {best_sal['class_1']['precision']:.4f} | Recall = {best_sal['class_1']['recall']:.4f} | F1 = {best_sal['class_1']['f1']:.4f}")
    print(f"  Macro F1 = {best_sal['macro_f1']:.4f} | ROC-AUC = {best_sal['roc_auc']:.4f}")

    print(f"\nRAID Test (N=297):")
    print(f"  Human (Class 0): Precision = {best_raid['class_0']['precision']:.4f} | Recall = {best_raid['class_0']['recall']:.4f} | F1 = {best_raid['class_0']['f1']:.4f}")
    print(f"  AI    (Class 1): Precision = {best_raid['class_1']['precision']:.4f} | Recall = {best_raid['class_1']['recall']:.4f} | F1 = {best_raid['class_1']['f1']:.4f}")
    print(f"  Macro F1 = {best_raid['macro_f1']:.4f} | ROC-AUC = {best_raid['roc_auc']:.4f}")

    print(f"\nOverall Main Test (N=4,338):")
    print(f"  Human (Class 0): Precision = {best_main['class_0']['precision']:.4f} | Recall = {best_main['class_0']['recall']:.4f} | F1 = {best_main['class_0']['f1']:.4f}")
    print(f"  AI    (Class 1): Precision = {best_main['class_1']['precision']:.4f} | Recall = {best_main['class_1']['recall']:.4f} | F1 = {best_main['class_1']['f1']:.4f}")
    print(f"  Macro F1 = {best_main['macro_f1']:.4f} | ROC-AUC = {best_main['roc_auc']:.4f}")

    # -------------------------------------------------------------------------
    # PRINT CLEAN V1 VS V2 COMPARISON TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("V1 VS V2 COMPARISON TABLE (MODEL x DATASET x MACRO F1)")
    print("=" * 80)

    # Load v1 metrics
    v1_metrics = {}
    if V1_METRICS_PATH.exists():
        with open(V1_METRICS_PATH, "r", encoding="utf-8") as f:
            v1_metrics = json.load(f)

    comparison_rows = []

    # Authorship models
    auth_model_list = ["tfidf_lr", "tfidf_svc", "stylometric_lr", "length_only"]
    for m in auth_model_list:
        # Salminen
        v1_sal = v1_metrics.get("authorship", {}).get("models", {}).get(m, {}).get("test_salminen", {}).get("macro_f1", None)
        v2_sal = auth_results[m]["test_salminen"]["macro_f1"]
        diff_sal = (v2_sal - v1_sal) if v1_sal is not None else None
        comparison_rows.append({
            "Task": "Authorship",
            "Model": m,
            "Dataset Slice": "Salminen Test",
            "v1 Macro F1": f"{v1_sal:.4f}" if v1_sal is not None else "N/A",
            "v2 Macro F1": f"{v2_sal:.4f}",
            "Diff (v2 - v1)": f"{diff_sal:+.4f}" if diff_sal is not None else "N/A",
        })

        # RAID
        v1_raid = v1_metrics.get("authorship", {}).get("models", {}).get(m, {}).get("test_raid", {}).get("macro_f1", None)
        v2_raid = auth_results[m]["test_raid"]["macro_f1"]
        diff_raid = (v2_raid - v1_raid) if v1_raid is not None else None
        comparison_rows.append({
            "Task": "Authorship",
            "Model": m,
            "Dataset Slice": "RAID Test",
            "v1 Macro F1": f"{v1_raid:.4f}" if v1_raid is not None else "N/A",
            "v2 Macro F1": f"{v2_raid:.4f}",
            "Diff (v2 - v1)": f"{diff_raid:+.4f}" if diff_raid is not None else "N/A",
        })

        # Overall
        v1_main = v1_metrics.get("authorship", {}).get("models", {}).get(m, {}).get("test_main", {}).get("macro_f1", None)
        v2_main = auth_results[m]["test_main"]["macro_f1"]
        diff_main = (v2_main - v1_main) if v1_main is not None else None
        comparison_rows.append({
            "Task": "Authorship",
            "Model": m,
            "Dataset Slice": "Overall Test",
            "v1 Macro F1": f"{v1_main:.4f}" if v1_main is not None else "N/A",
            "v2 Macro F1": f"{v2_main:.4f}",
            "Diff (v2 - v1)": f"{diff_main:+.4f}" if diff_main is not None else "N/A",
        })

    # Balanced domain
    v1_bal_sal = v1_metrics.get("authorship", {}).get("models", {}).get("tfidf_lr_balanced_domain", {}).get("test_salminen", {}).get("macro_f1", None)
    v2_bal_sal = auth_results["tfidf_lr_balanced_domain"]["test_salminen"]["macro_f1"]
    comparison_rows.append({
        "Task": "Authorship",
        "Model": "tfidf_lr (Balanced 4k)",
        "Dataset Slice": "Salminen Test",
        "v1 Macro F1": f"{v1_bal_sal:.4f}" if v1_bal_sal is not None else "N/A",
        "v2 Macro F1": f"{v2_bal_sal:.4f}",
        "Diff (v2 - v1)": f"{(v2_bal_sal - v1_bal_sal):+.4f}" if v1_bal_sal is not None else "N/A",
    })

    v1_bal_raid = v1_metrics.get("authorship", {}).get("models", {}).get("tfidf_lr_balanced_domain", {}).get("test_raid", {}).get("macro_f1", None)
    v2_bal_raid = auth_results["tfidf_lr_balanced_domain"]["test_raid"]["macro_f1"]
    comparison_rows.append({
        "Task": "Authorship",
        "Model": "tfidf_lr (Balanced 4k)",
        "Dataset Slice": "RAID Test",
        "v1 Macro F1": f"{v1_bal_raid:.4f}" if v1_bal_raid is not None else "N/A",
        "v2 Macro F1": f"{v2_bal_raid:.4f}",
        "Diff (v2 - v1)": f"{(v2_bal_raid - v1_bal_raid):+.4f}" if v1_bal_raid is not None else "N/A",
    })

    # Deception models (Ott 5-Fold CV Mean)
    dec_model_list = ["tfidf_lr", "tfidf_svc", "stylometric_lr", "length_only"]
    for m in dec_model_list:
        v1_dec = v1_metrics.get("deception", {}).get("models", {}).get(m, {}).get("mean_macro_f1", None)
        v2_dec = dec_results[m]["mean_macro_f1"]
        diff_dec = (v2_dec - v1_dec) if v1_dec is not None else None
        comparison_rows.append({
            "Task": "Deception",
            "Model": m,
            "Dataset Slice": "Ott (5-Fold CV Mean)",
            "v1 Macro F1": f"{v1_dec:.4f}" if v1_dec is not None else "N/A",
            "v2 Macro F1": f"{v2_dec:.4f}",
            "Diff (v2 - v1)": f"{diff_dec:+.4f}" if diff_dec is not None else "N/A",
        })

    comp_df = pd.DataFrame(comparison_rows)
    print(comp_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()

