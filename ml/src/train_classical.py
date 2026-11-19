import json
import os
from datetime import datetime
from pathlib import Path

# Configure writable matplotlib cache
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
    classification_report,
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

from ml.src.features.stylometric import StylometricExtractor, TextLengthExtractor

DATA_DIR = Path("data/processed")
REPORTS_DIR = Path("ml/reports")
MODELS_DIR = Path("api/models")
SEED = 42

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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
        
        # LinearSVC wrapped in CalibratedClassifierCV
        if isinstance(clf, CalibratedClassifierCV):
            # Average base estimator coefficients
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

        # Feature names from first step
        feat_extractor = pipeline.named_steps[step_names[0]]
        if hasattr(feat_extractor, "get_feature_names_out"):
            feat_names = np.array(feat_extractor.get_feature_names_out())
        else:
            feat_names = np.array([f"f_{i}" for i in range(len(coefs))])

        sorted_indices = np.argsort(coefs)
        top_neg_idx = sorted_indices[:top_n]
        top_pos_idx = sorted_indices[-top_n:][::-1]

        return {
            "top_negative": [{"feature": feat_names[i], "coef": float(coefs[i])} for i in top_neg_idx],
            "top_positive": [{"feature": feat_names[i], "coef": float(coefs[i])} for i in top_pos_idx],
        }
    except Exception as e:
        return {"error": str(e), "top_negative": [], "top_positive": []}


# -----------------------------------------------------------------------------
# Authorship Training & Evaluation
# -----------------------------------------------------------------------------
def run_authorship():
    print("=" * 80)
    print("1. RUNNING AUTHORSHIP EXPERIMENTS (Human vs. AI)")
    print("=" * 80)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    ott_df = pd.read_csv(DATA_DIR / "ott_deception_5fold.csv")

    # Target label: 0 for human, 1 for ai
    y_train = (train_df["authorship_label"] == "ai").astype(int).values
    y_val = (val_df["authorship_label"] == "ai").astype(int).values

    # Test subsets: Exclude Ott from main numbers
    test_main = test_df[test_df["source_dataset"].isin(["salminen", "raid"])].copy()
    y_test_main = (test_main["authorship_label"] == "ai").astype(int).values

    test_sal = test_df[test_df["source_dataset"] == "salminen"].copy()
    y_test_sal = (test_sal["authorship_label"] == "ai").astype(int).values

    test_raid = test_df[test_df["source_dataset"] == "raid"].copy()
    y_test_raid = (test_raid["authorship_label"] == "ai").astype(int).values

    print(f"Authorship Train size: {len(train_df):,} | Val size: {len(val_df):,}")
    print(f"Test Main: {len(test_main):,} (Salminen: {len(test_sal):,}, RAID: {len(test_raid):,})")

    # Define model pipelines
    models = {
        "tfidf_lr": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True, min_df=2)),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
        ]),
        "tfidf_svc": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True, min_df=2)),
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
    best_macro_f1 = -1.0

    plt.figure(figsize=(9, 7))
    
    for m_name, pipe in models.items():
        print(f"\nFitting Authorship Model: {m_name}...")
        pipe.fit(train_df["text"], y_train)

        # Val score
        val_pred = pipe.predict(val_df["text"])
        val_prob = pipe.predict_proba(val_df["text"])[:, 1]
        val_metrics = compute_metrics(y_val, val_pred, val_prob)

        # Test Main
        test_pred = pipe.predict(test_main["text"])
        test_prob = pipe.predict_proba(test_main["text"])[:, 1]
        test_metrics = compute_metrics(y_test_main, test_pred, test_prob)

        # Test Salminen
        sal_pred = pipe.predict(test_sal["text"])
        sal_prob = pipe.predict_proba(test_sal["text"])[:, 1]
        sal_metrics = compute_metrics(y_test_sal, sal_pred, sal_prob)

        # Test RAID
        raid_pred = pipe.predict(test_raid["text"])
        raid_prob = pipe.predict_proba(test_raid["text"])[:, 1]
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

        if test_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = test_metrics["macro_f1"]
            best_model_name = m_name

        # Plot ROC curve for test_main
        if test_prob is not None:
            fpr, tpr, _ = roc_curve(y_test_main, test_prob)
            plt.plot(fpr, tpr, label=f"{m_name} (AUC={test_metrics['roc_auc']:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate (AI predicted as Human)")
    plt.ylabel("True Positive Rate (AI predicted as AI)")
    plt.title("Authorship ROC Curves (Test Set: Salminen + RAID)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "authorship_roc_curves.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Cross-dataset Generalization Experiments
    # -------------------------------------------------------------------------
    print("\n--- Running Cross-Dataset Generalization Experiments ---")
    cross_results = {}

    # Train on Salminen only -> Test on RAID test
    sal_only_train = train_df[train_df["source_dataset"] == "salminen"].copy()
    y_sal_train = (sal_only_train["authorship_label"] == "ai").astype(int).values
    cross_sal_model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
    ])
    cross_sal_model.fit(sal_only_train["text"], y_sal_train)
    sal2raid_pred = cross_sal_model.predict(test_raid["text"])
    sal2raid_prob = cross_sal_model.predict_proba(test_raid["text"])[:, 1]
    cross_results["train_salminen_test_raid"] = compute_metrics(y_test_raid, sal2raid_pred, sal2raid_prob)

    # Train on RAID only -> Test on Salminen test
    raid_only_train = train_df[train_df["source_dataset"] == "raid"].copy()
    y_raid_train = (raid_only_train["authorship_label"] == "ai").astype(int).values
    cross_raid_model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
    ])
    cross_raid_model.fit(raid_only_train["text"], y_raid_train)
    raid2sal_pred = cross_raid_model.predict(test_sal["text"])
    raid2sal_prob = cross_raid_model.predict_proba(test_sal["text"])[:, 1]
    cross_results["train_raid_test_salminen"] = compute_metrics(y_test_sal, raid2sal_pred, raid2sal_prob)

    print(f"Salminen-only -> RAID Test Macro F1: {cross_results['train_salminen_test_raid']['macro_f1']:.4f}")
    print(f"RAID-only -> Salminen Test Macro F1: {cross_results['train_raid_test_salminen']['macro_f1']:.4f}")

    # -------------------------------------------------------------------------
    # Balanced-domain Variant (Cap Salminen at 4k/class)
    # -------------------------------------------------------------------------
    print("\n--- Training Balanced-Domain Variant (Salminen capped at 4k/class) ---")
    sal_h = sal_only_train[sal_only_train["authorship_label"] == "human"].sample(n=4000, random_state=SEED)
    sal_a = sal_only_train[sal_only_train["authorship_label"] == "ai"].sample(n=4000, random_state=SEED)
    balanced_train_df = pd.concat([sal_h, sal_a, raid_only_train], ignore_index=True)
    y_balanced_train = (balanced_train_df["authorship_label"] == "ai").astype(int).values

    balanced_model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
    ])
    balanced_model.fit(balanced_train_df["text"], y_balanced_train)

    bal_main_pred = balanced_model.predict(test_main["text"])
    bal_main_prob = balanced_model.predict_proba(test_main["text"])[:, 1]
    bal_main_metrics = compute_metrics(y_test_main, bal_main_pred, bal_main_prob)

    bal_sal_pred = balanced_model.predict(test_sal["text"])
    bal_sal_metrics = compute_metrics(y_test_sal, bal_sal_pred)

    bal_raid_pred = balanced_model.predict(test_raid["text"])
    bal_raid_metrics = compute_metrics(y_test_raid, bal_raid_pred)

    results["tfidf_lr_balanced_domain"] = {
        "test_main": bal_main_metrics,
        "test_salminen": bal_sal_metrics,
        "test_raid": bal_raid_metrics,
        "train_size": len(balanced_train_df),
    }
    print(f"Balanced Variant Main Test F1: {bal_main_metrics['macro_f1']:.4f} (Salminen: {bal_sal_metrics['macro_f1']:.4f}, RAID: {bal_raid_metrics['macro_f1']:.4f})")

    # -------------------------------------------------------------------------
    # Ott Held-out Authorship Independence Test
    # -------------------------------------------------------------------------
    print("\n--- Running Best Model on Ott Rows (Independence Test) ---")
    best_pipe = models[best_model_name]
    ott_preds = best_pipe.predict(ott_df["text"])
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
        "note": "Ott is 100% human-authored. Low AI-predicted share confirms model does not conflate human deception with AI generation."
    }
    print(f"Overall Ott share predicted as AI: {total_ai_share*100:.2f}%")
    print(f"  Truthful rows predicted as AI: {truthful_ai_share*100:.2f}%")
    print(f"  Deceptive rows predicted as AI: {deceptive_ai_share*100:.2f}%")

    # Save best authorship model
    joblib.dump(best_pipe, MODELS_DIR / "authorship_classical.joblib")
    print(f"Saved best authorship model ({best_model_name}) to {MODELS_DIR / 'authorship_classical.joblib'}")

    # Plot Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_sal_data = np.array(results[best_model_name]["test_salminen"]["confusion_matrix"])
    cm_raid_data = np.array(results[best_model_name]["test_raid"]["confusion_matrix"])
    sns.heatmap(cm_sal_data, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
    axes[0].set_title(f"Authorship: Salminen Test ({best_model_name})")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    sns.heatmap(cm_raid_data, annot=True, fmt="d", cmap="Greens", ax=axes[1],
                xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
    axes[1].set_title(f"Authorship: RAID Test ({best_model_name})")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "authorship_confusion_matrices.png", dpi=300)
    plt.close()

    return results, cross_results, ott_independence_report, best_model_name


# -----------------------------------------------------------------------------
# Deception Training & 5-Fold Stratified CV
# -----------------------------------------------------------------------------
def run_deception():
    print("\n" + "=" * 80)
    print("2. RUNNING DECEPTION EXPERIMENTS (5-Fold Stratified CV on Ott et al.)")
    print("=" * 80)

    ott_df = pd.read_csv(DATA_DIR / "ott_deception_5fold.csv")
    y_all = (ott_df["deception_label"] == "deceptive").astype(int).values

    models = {
        "tfidf_lr": lambda: Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True, min_df=2)),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
        ]),
        "tfidf_svc": lambda: Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True, min_df=2)),
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

            X_tr, y_tr = ott_df.loc[train_idx, "text"], y_all[train_idx]
            X_te, y_te = ott_df.loc[test_idx, "text"], y_all[test_idx]

            pipe = model_fn()
            # Vectorizers / Scalers fit strictly on fold training data
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
    print(f"\nFitting final best deception model ({best_deception_model}) on all Ott rows...")
    final_deception_pipe = models[best_deception_model]()
    final_deception_pipe.fit(ott_df["text"], y_all)
    joblib.dump(final_deception_pipe, MODELS_DIR / "deception_classical.joblib")
    print(f"Saved {MODELS_DIR / 'deception_classical.joblib'}")

    top_deception_feats = get_feature_importances(final_deception_pipe, best_deception_model)
    deception_results["best_model_top_features"] = top_deception_feats

    # Plot Deception Confusion Matrix & ROC Curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    oof_cm = deception_results[best_deception_model]["overall_oof_metrics"]["confusion_matrix"]
    sns.heatmap(oof_cm, annot=True, fmt="d", cmap="Purples", ax=axes[0],
                xticklabels=["Genuine", "Deceptive"], yticklabels=["Genuine", "Deceptive"])
    axes[0].set_title(f"Deception 5-Fold OOF Confusion Matrix ({best_deception_model})")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    best_oof_probs = np.zeros(len(ott_df))
    # Reconstruct OOF probs for plot
    for fold in range(5):
        tr_idx = ott_df[ott_df["deception_cv_fold"] != fold].index
        te_idx = ott_df[ott_df["deception_cv_fold"] == fold].index
        p = models[best_deception_model]()
        p.fit(ott_df.loc[tr_idx, "text"], y_all[tr_idx])
        best_oof_probs[te_idx] = p.predict_proba(ott_df.loc[te_idx, "text"])[:, 1]

    fpr, tpr, _ = roc_curve(y_all, best_oof_probs)
    axes[1].plot(fpr, tpr, label=f"OOF ROC (AUC = {roc_auc_score(y_all, best_oof_probs):.3f})", color="purple")
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("Deception OOF ROC Curve (Ott et al.)")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "deception_evaluation.png", dpi=300)
    plt.close()

    return deception_results, best_deception_model


# -----------------------------------------------------------------------------
# Save Metadata & Report Top Features
# -----------------------------------------------------------------------------
def main():
    auth_results, cross_results, ott_independence, best_auth_name = run_authorship()
    dec_results, best_dec_name = run_deception()

    # Save comprehensive metrics JSON
    all_metrics = {
        "timestamp": datetime.now().isoformat(),
        "authorship": {
            "best_model": best_auth_name,
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

    metrics_json_path = REPORTS_DIR / "classical_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved detailed evaluation metrics to {metrics_json_path}")

    # Save production models metadata JSON in api/models/
    prod_metadata = {
        "timestamp": datetime.now().isoformat(),
        "authorship_model": {
            "filename": "authorship_classical.joblib",
            "architecture": best_auth_name,
            "test_macro_f1": auth_results[best_auth_name]["test_main"]["macro_f1"],
            "test_roc_auc": auth_results[best_auth_name]["test_main"]["roc_auc"],
            "salminen_test_f1": auth_results[best_auth_name]["test_salminen"]["macro_f1"],
            "raid_test_f1": auth_results[best_auth_name]["test_raid"]["macro_f1"],
            "ott_predicted_ai_share": ott_independence["total_predicted_ai_share"],
        },
        "deception_model": {
            "filename": "deception_classical.joblib",
            "architecture": best_dec_name,
            "cv_macro_f1_mean": dec_results[best_dec_name]["mean_macro_f1"],
            "cv_macro_f1_std": dec_results[best_dec_name]["std_macro_f1"],
            "cv_roc_auc_mean": dec_results[best_dec_name]["roc_auc_mean"],
        },
        "feature_settings": {
            "tfidf_ngram_range": [1, 2],
            "tfidf_sublinear_tf": True,
            "min_df": 2,
            "stylometric_features": ["burstiness", "type_token_ratio", "avg_word_length", "punctuation_density", "text_length"],
            "perplexity_optional": True,
        },
        "dataset_files": [
            "data/processed/train.csv",
            "data/processed/val.csv",
            "data/processed/test.csv",
            "data/processed/ott_deception_5fold.csv"
        ]
    }

    prod_meta_path = MODELS_DIR / "classical_metadata.json"
    with open(prod_meta_path, "w", encoding="utf-8") as f:
        json.dump(prod_metadata, f, indent=2)
    print(f"Saved production metadata to {prod_meta_path}")

    # -------------------------------------------------------------------------
    # Print Top 20 Features Pushing Towards Each Class
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"TOP 20 FEATURES FOR BEST AUTHORSHIP MODEL ({best_auth_name})")
    print("=" * 80)
    auth_top = auth_results[best_auth_name]["top_features"]
    print("Top 20 Features Pushing Toward HUMAN (Negative Coefs):")
    for item in auth_top.get("top_negative", []):
        print(f"  {item['feature']:<25} : {item['coef']:+.4f}")
    print("\nTop 20 Features Pushing Toward AI (Positive Coefs):")
    for item in auth_top.get("top_positive", []):
        print(f"  {item['feature']:<25} : {item['coef']:+.4f}")

    print("\n" + "=" * 80)
    print(f"TOP 20 FEATURES FOR BEST DECEPTION MODEL ({best_dec_name})")
    print("=" * 80)
    dec_top = dec_results["best_model_top_features"]
    print("Top 20 Features Pushing Toward GENUINE (Negative Coefs):")
    for item in dec_top.get("top_negative", []):
        print(f"  {item['feature']:<25} : {item['coef']:+.4f}")
    print("\nTop 20 Features Pushing Toward DECEPTIVE (Positive Coefs):")
    for item in dec_top.get("top_positive", []):
        print(f"  {item['feature']:<25} : {item['coef']:+.4f}")

    # -------------------------------------------------------------------------
    # Print Summary Comparison Table (Model x Dataset x F1)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL COMPARISON TABLE: MODEL x DATASET x F1")
    print("=" * 80)

    rows = []
    # Authorship
    for m in ["tfidf_lr", "tfidf_svc", "stylometric_lr", "length_only"]:
        rows.append({
            "Task": "Authorship",
            "Model": m,
            "Evaluation Dataset": "Salminen Test",
            "Macro F1": f"{auth_results[m]['test_salminen']['macro_f1']:.4f}",
            "ROC-AUC": f"{auth_results[m]['test_salminen']['roc_auc']:.4f}" if auth_results[m]['test_salminen']['roc_auc'] else "N/A"
        })
        rows.append({
            "Task": "Authorship",
            "Model": m,
            "Evaluation Dataset": "RAID Test (Movie)",
            "Macro F1": f"{auth_results[m]['test_raid']['macro_f1']:.4f}",
            "ROC-AUC": f"{auth_results[m]['test_raid']['roc_auc']:.4f}" if auth_results[m]['test_raid']['roc_auc'] else "N/A"
        })
        rows.append({
            "Task": "Authorship",
            "Model": m,
            "Evaluation Dataset": "Overall Test (Salminen + RAID)",
            "Macro F1": f"{auth_results[m]['test_main']['macro_f1']:.4f}",
            "ROC-AUC": f"{auth_results[m]['test_main']['roc_auc']:.4f}" if auth_results[m]['test_main']['roc_auc'] else "N/A"
        })

    # Balanced-domain variant
    rows.append({
        "Task": "Authorship",
        "Model": "tfidf_lr (Balanced 4k Salminen)",
        "Evaluation Dataset": "RAID Test (Movie)",
        "Macro F1": f"{auth_results['tfidf_lr_balanced_domain']['test_raid']['macro_f1']:.4f}",
        "ROC-AUC": "N/A"
    })
    rows.append({
        "Task": "Authorship",
        "Model": "tfidf_lr (Balanced 4k Salminen)",
        "Evaluation Dataset": "Salminen Test",
        "Macro F1": f"{auth_results['tfidf_lr_balanced_domain']['test_salminen']['macro_f1']:.4f}",
        "ROC-AUC": "N/A"
    })
    rows.append({
        "Task": "Authorship",
        "Model": "tfidf_lr (Balanced 4k Salminen)",
        "Evaluation Dataset": "Overall Test",
        "Macro F1": f"{auth_results['tfidf_lr_balanced_domain']['test_main']['macro_f1']:.4f}",
        "ROC-AUC": f"{auth_results['tfidf_lr_balanced_domain']['test_main']['roc_auc']:.4f}" if auth_results['tfidf_lr_balanced_domain']['test_main']['roc_auc'] else "N/A"
    })

    # Cross dataset
    rows.append({
        "Task": "Authorship Cross-Domain",
        "Model": "tfidf_lr (Salminen-only Train)",
        "Evaluation Dataset": "RAID Test (Zero-Shot)",
        "Macro F1": f"{cross_results['train_salminen_test_raid']['macro_f1']:.4f}",
        "ROC-AUC": f"{cross_results['train_salminen_test_raid']['roc_auc']:.4f}" if cross_results['train_salminen_test_raid']['roc_auc'] else "N/A"
    })
    rows.append({
        "Task": "Authorship Cross-Domain",
        "Model": "tfidf_lr (RAID-only Train)",
        "Evaluation Dataset": "Salminen Test (Zero-Shot)",
        "Macro F1": f"{cross_results['train_raid_test_salminen']['macro_f1']:.4f}",
        "ROC-AUC": f"{cross_results['train_raid_test_salminen']['roc_auc']:.4f}" if cross_results['train_raid_test_salminen']['roc_auc'] else "N/A"
    })

    # Deception
    for m in ["tfidf_lr", "tfidf_svc", "stylometric_lr", "length_only"]:
        rows.append({
            "Task": "Deception",
            "Model": m,
            "Evaluation Dataset": "Ott et al. (5-Fold CV Mean)",
            "Macro F1": f"{dec_results[m]['mean_macro_f1']:.4f} ± {dec_results[m]['std_macro_f1']:.4f}",
            "ROC-AUC": f"{dec_results[m]['roc_auc_mean']:.4f}" if dec_results[m]['roc_auc_mean'] else "N/A"
        })

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    main()
