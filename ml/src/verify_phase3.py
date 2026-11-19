import json
import os
from datetime import datetime
from pathlib import Path

# Matplotlib cache setup if needed
mpl_cache = Path("ml/reports/.matplotlib")
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_cache.resolve())

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("api/models")
REPORTS_DIR = Path("ml/reports")
REPORT_OUTPUT_PATH = REPORTS_DIR / "phase3_verification_v1.json"
SEED = 42

def extract_top_features(pipeline, pos_label_name="positive", neg_label_name="negative", top_n=20) -> dict:
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    if isinstance(clf, CalibratedClassifierCV):
        coef_list = []
        for cal_clf in clf.calibrated_classifiers_:
            base = getattr(cal_clf, "estimator", cal_clf)
            if hasattr(base, "coef_"):
                coef_list.append(base.coef_)
        if coef_list:
            coefs = np.mean(coef_list, axis=0).ravel()
        else:
            raise ValueError("No calibrated estimator with coef_ found.")
    elif hasattr(clf, "coef_"):
        coefs = clf.coef_.ravel()
    else:
        raise ValueError(f"Classifier {type(clf)} does not expose coef_.")

    feat_names = np.array(tfidf.get_feature_names_out())
    sorted_idx = np.argsort(coefs)

    top_neg = [
        {"feature": str(feat_names[i]), "weight": float(coefs[i]), "target_class": neg_label_name}
        for i in sorted_idx[:top_n]
    ]
    top_pos = [
        {"feature": str(feat_names[i]), "weight": float(coefs[i]), "target_class": pos_label_name}
        for i in sorted_idx[-top_n:][::-1]
    ]

    return {
        "negative_weights": top_neg,
        "positive_weights": top_pos,
    }


def bootstrap_macro_f1(y_true, y_pred, n_bootstraps=1000, seed=42) -> dict:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_f1s = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        f1 = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
        boot_f1s.append(f1)
    boot_f1s = np.array(boot_f1s)
    return {
        "n_bootstraps": n_bootstraps,
        "observed_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "bootstrap_mean": float(np.mean(boot_f1s)),
        "bootstrap_std": float(np.std(boot_f1s)),
        "ci_95_lower": float(np.percentile(boot_f1s, 2.5)),
        "ci_95_upper": float(np.percentile(boot_f1s, 97.5)),
    }


def compute_near_duplicates(train_df, test_df, dataset_name: str, threshold: float = 0.9):
    tr = train_df[train_df["source_dataset"] == dataset_name].reset_index(drop=True)
    te = test_df[test_df["source_dataset"] == dataset_name].reset_index(drop=True)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_tr = vec.fit_transform(tr["text"])
    X_te = vec.transform(te["text"])

    # Batch cosine similarity computation
    batch_size = 500
    max_sims = np.zeros(len(te))
    for start_idx in range(0, len(te), batch_size):
        end_idx = min(start_idx + batch_size, len(te))
        sim_batch = linear_kernel(X_te[start_idx:end_idx], X_tr)
        max_sims[start_idx:end_idx] = sim_batch.max(axis=1)

    near_dup_mask = max_sims > threshold
    num_near_duplicates = int(near_dup_mask.sum())
    near_dup_indices = np.where(near_dup_mask)[0].tolist()

    return {
        "dataset": dataset_name,
        "total_test_rows": len(te),
        "threshold": threshold,
        "near_duplicate_count": num_near_duplicates,
        "near_duplicate_ratio": float(num_near_duplicates / len(te)) if len(te) > 0 else 0.0,
        "near_dup_mask": near_dup_mask,
        "te_subset": te,
    }


def main():
    print("=" * 80)
    print("PHASE 3 VERIFICATION AUDIT")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    # 1. Clean Model Loading Verification
    auth_path = MODELS_DIR / "authorship_classical.joblib"
    dec_path = MODELS_DIR / "deception_classical.joblib"

    print(f"\n[MODEL LOADING CHECK]")
    print(f"Loading Authorship Model from: {auth_path.resolve()}")
    auth_model = joblib.load(auth_path)
    print(f"  -> Authorship model loaded successfully: {type(auth_model)}")
    print(f"     Steps: {list(auth_model.named_steps.keys())}")

    print(f"Loading Deception Model from: {dec_path.resolve()}")
    dec_model = joblib.load(dec_path)
    print(f"  -> Deception model loaded successfully: {type(dec_model)}")
    print(f"     Steps: {list(dec_model.named_steps.keys())}")

    # Load datasets
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    ott_df = pd.read_csv(DATA_DIR / "ott_deception_5fold.csv")

    # -------------------------------------------------------------------------
    # ITEM 1: Top 20 Features per Class
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ITEM 1: TOP 20 FEATURES PER CLASS")
    print("=" * 80)

    auth_top_features = extract_top_features(auth_model, pos_label_name="ai", neg_label_name="human", top_n=20)
    dec_top_features = extract_top_features(dec_model, pos_label_name="deceptive", neg_label_name="genuine", top_n=20)

    print("\n[Authorship Model (api/models/authorship_classical.joblib)]")
    print("Top 20 Features Pushing Toward HUMAN (Negative Weights):")
    for f in auth_top_features["negative_weights"]:
        print(f"  {f['feature']:<25} : {f['weight']:+.4f}")
    print("\nTop 20 Features Pushing Toward AI (Positive Weights):")
    for f in auth_top_features["positive_weights"]:
        print(f"  {f['feature']:<25} : {f['weight']:+.4f}")

    print("\n[Deception Model (api/models/deception_classical.joblib)]")
    print("Top 20 Features Pushing Toward GENUINE (Negative Weights):")
    for f in dec_top_features["negative_weights"]:
        print(f"  {f['feature']:<25} : {f['weight']:+.4f}")
    print("\nTop 20 Features Pushing Toward DECEPTIVE (Positive Weights):")
    for f in dec_top_features["positive_weights"]:
        print(f"  {f['feature']:<25} : {f['weight']:+.4f}")

    # -------------------------------------------------------------------------
    # ITEM 2: 10 Random Misclassified Examples (Seed 42)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ITEM 2: 10 RANDOM MISCLASSIFIED EXAMPLES (Seed 42)")
    print("=" * 80)

    # 2a. Authorship on Salminen test
    sal_test = test_df[test_df["source_dataset"] == "salminen"].reset_index(drop=True)
    sal_y_true = (sal_test["authorship_label"] == "ai").astype(int).values
    sal_y_pred = auth_model.predict(sal_test["text"])
    sal_y_prob = auth_model.predict_proba(sal_test["text"])[:, 1]

    sal_misc_idx = np.where(sal_y_true != sal_y_pred)[0]
    print(f"\nSalminen Test Total: {len(sal_test)} | Misclassified: {len(sal_misc_idx)}")
    rng_sal = np.random.RandomState(SEED)
    sample_sal_idx = rng_sal.choice(sal_misc_idx, size=min(10, len(sal_misc_idx)), replace=False)

    sal_misclassified_samples = []
    for idx in sample_sal_idx:
        row = sal_test.iloc[idx]
        item = {
            "row_id": str(row["row_id"]),
            "true_label": str(row["authorship_label"]),
            "predicted_label": "ai" if sal_y_pred[idx] == 1 else "human",
            "predicted_prob_ai": float(sal_y_prob[idx]),
            "text": str(row["text"])[:400],
        }
        sal_misclassified_samples.append(item)

    # 2b. Authorship on RAID test
    raid_test = test_df[test_df["source_dataset"] == "raid"].reset_index(drop=True)
    raid_y_true = (raid_test["authorship_label"] == "ai").astype(int).values
    raid_y_pred = auth_model.predict(raid_test["text"])
    raid_y_prob = auth_model.predict_proba(raid_test["text"])[:, 1]

    raid_misc_idx = np.where(raid_y_true != raid_y_pred)[0]
    print(f"RAID Test Total: {len(raid_test)} | Misclassified: {len(raid_misc_idx)}")
    rng_raid = np.random.RandomState(SEED)
    sample_raid_idx = rng_raid.choice(raid_misc_idx, size=min(10, len(raid_misc_idx)), replace=False)

    raid_misclassified_samples = []
    for idx in sample_raid_idx:
        row = raid_test.iloc[idx]
        item = {
            "row_id": str(row["row_id"]),
            "true_label": str(row["authorship_label"]),
            "predicted_label": "ai" if raid_y_pred[idx] == 1 else "human",
            "predicted_prob_ai": float(raid_y_prob[idx]),
            "text": str(row["text"])[:400],
        }
        raid_misclassified_samples.append(item)

    # 2c. Deception on Ott (5-fold stratified CV out-of-fold predictions)
    # The production saved model was trained on 100% of Ott rows and achieves 0 training errors.
    # Out-of-fold (OOF) cross-validation predictions are evaluated to examine genuine generalization errors.
    print(f"\nComputing 5-fold Stratified CV Out-of-Fold (OOF) predictions for Ott Deception...")
    ott_y_true = (ott_df["deception_label"] == "deceptive").astype(int).values
    oof_preds = np.zeros(len(ott_df), dtype=int)
    oof_probs = np.zeros(len(ott_df), dtype=float)

    for fold in range(5):
        train_idx = ott_df[ott_df["deception_cv_fold"] != fold].index
        val_idx = ott_df[ott_df["deception_cv_fold"] == fold].index

        fold_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True, min_df=2)),
            ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, dual=False, random_state=SEED), method="sigmoid", cv=3)),
        ])
        fold_pipe.fit(ott_df.loc[train_idx, "text"], ott_y_true[train_idx])
        oof_preds[val_idx] = fold_pipe.predict(ott_df.loc[val_idx, "text"])
        oof_probs[val_idx] = fold_pipe.predict_proba(ott_df.loc[val_idx, "text"])[:, 1]

    ott_misc_idx = np.where(ott_y_true != oof_preds)[0]
    print(f"Ott Deception Total: {len(ott_df)} | OOF Misclassified: {len(ott_misc_idx)}")
    rng_ott = np.random.RandomState(SEED)
    sample_ott_idx = rng_ott.choice(ott_misc_idx, size=min(10, len(ott_misc_idx)), replace=False)

    ott_misclassified_samples = []
    for idx in sample_ott_idx:
        row = ott_df.iloc[idx]
        item = {
            "row_id": str(row["row_id"]),
            "true_label": str(row["deception_label"]),
            "predicted_label": "deceptive" if oof_preds[idx] == 1 else "genuine",
            "predicted_prob_deceptive": float(oof_probs[idx]),
            "text": str(row["text"])[:400],
        }
        ott_misclassified_samples.append(item)

    # Print samples
    print("\n--- 10 Misclassified Salminen Test Examples ---")
    for i, s in enumerate(sal_misclassified_samples, 1):
        print(f"[{i}] Row ID: {s['row_id']} | True: {s['true_label']} | Pred: {s['predicted_label']} (P(AI)={s['predicted_prob_ai']:.4f})")
        print(f"    Text: {s['text']}\n")

    print("\n--- 10 Misclassified RAID Test Examples ---")
    for i, s in enumerate(raid_misclassified_samples, 1):
        print(f"[{i}] Row ID: {s['row_id']} | True: {s['true_label']} | Pred: {s['predicted_label']} (P(AI)={s['predicted_prob_ai']:.4f})")
        print(f"    Text: {s['text']}\n")

    print("\n--- 10 Misclassified Ott Deception Examples (5-Fold CV OOF) ---")
    for i, s in enumerate(ott_misclassified_samples, 1):
        print(f"[{i}] Row ID: {s['row_id']} | True: {s['true_label']} | Pred: {s['predicted_label']} (P(Deceptive)={s['predicted_prob_deceptive']:.4f})")
        print(f"    Text: {s['text']}\n")

    # -------------------------------------------------------------------------
    # ITEM 3: Bootstrap 95% Confidence Intervals (1,000 resamples, Seed 42)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ITEM 3: BOOTSTRAP 95% CONFIDENCE INTERVALS (1,000 resamples, Seed 42)")
    print("=" * 80)

    sal_bootstrap = bootstrap_macro_f1(sal_y_true, sal_y_pred, n_bootstraps=1000, seed=SEED)
    raid_bootstrap = bootstrap_macro_f1(raid_y_true, raid_y_pred, n_bootstraps=1000, seed=SEED)

    print(f"Salminen Test (N={len(sal_test)}):")
    print(f"  Observed Macro F1: {sal_bootstrap['observed_macro_f1']:.4f}")
    print(f"  Bootstrap Mean:   {sal_bootstrap['bootstrap_mean']:.4f} ± {sal_bootstrap['bootstrap_std']:.4f}")
    print(f"  95% CI:           [{sal_bootstrap['ci_95_lower']:.4f}, {sal_bootstrap['ci_95_upper']:.4f}]")

    print(f"\nRAID Test (N={len(raid_test)}):")
    print(f"  Observed Macro F1: {raid_bootstrap['observed_macro_f1']:.4f}")
    print(f"  Bootstrap Mean:   {raid_bootstrap['bootstrap_mean']:.4f} ± {raid_bootstrap['bootstrap_std']:.4f}")
    print(f"  95% CI:           [{raid_bootstrap['ci_95_lower']:.4f}, {raid_bootstrap['ci_95_upper']:.4f}]")

    # -------------------------------------------------------------------------
    # ITEM 5: Near-Duplicate Check (TF-IDF Cosine Similarity > 0.9)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ITEM 5: NEAR-DUPLICATE CHECK (Train vs Test Cosine Similarity > 0.9)")
    print("=" * 80)

    sal_dup_info = compute_near_duplicates(train_df, test_df, "salminen", threshold=0.9)
    raid_dup_info = compute_near_duplicates(train_df, test_df, "raid", threshold=0.9)

    # Evaluate Salminen before and after removing near duplicates
    sal_f1_before = sal_bootstrap["observed_macro_f1"]
    sal_clean_mask = ~sal_dup_info["near_dup_mask"]
    sal_y_true_clean = sal_y_true[sal_clean_mask]
    sal_y_pred_clean = sal_y_pred[sal_clean_mask]
    sal_f1_after = float(f1_score(sal_y_true_clean, sal_y_pred_clean, average="macro", zero_division=0))

    # Evaluate RAID before and after removing near duplicates
    raid_f1_before = raid_bootstrap["observed_macro_f1"]
    raid_clean_mask = ~raid_dup_info["near_dup_mask"]
    raid_y_true_clean = raid_y_true[raid_clean_mask]
    raid_y_pred_clean = raid_y_pred[raid_clean_mask]
    raid_f1_after = float(f1_score(raid_y_true_clean, raid_y_pred_clean, average="macro", zero_division=0))

    sal_near_dup_results = {
        "dataset": "salminen",
        "total_test_rows": len(sal_test),
        "near_duplicate_count": sal_dup_info["near_duplicate_count"],
        "near_duplicate_ratio": sal_dup_info["near_duplicate_ratio"],
        "macro_f1_before": sal_f1_before,
        "macro_f1_after": sal_f1_after,
        "macro_f1_diff": sal_f1_after - sal_f1_before,
    }

    raid_near_dup_results = {
        "dataset": "raid",
        "total_test_rows": len(raid_test),
        "near_duplicate_count": raid_dup_info["near_duplicate_count"],
        "near_duplicate_ratio": raid_dup_info["near_duplicate_ratio"],
        "macro_f1_before": raid_f1_before,
        "macro_f1_after": raid_f1_after,
        "macro_f1_diff": raid_f1_after - raid_f1_before,
    }

    print(f"Salminen:")
    print(f"  Test rows with max cosine sim > 0.9 to train: {sal_near_dup_results['near_duplicate_count']} / {sal_near_dup_results['total_test_rows']} ({sal_near_dup_results['near_duplicate_ratio']*100:.2f}%)")
    print(f"  Macro F1 Before: {sal_f1_before:.4f}")
    print(f"  Macro F1 After:  {sal_f1_after:.4f} (diff: {sal_near_dup_results['macro_f1_diff']:+.4f})")

    print(f"\nRAID:")
    print(f"  Test rows with max cosine sim > 0.9 to train: {raid_near_dup_results['near_duplicate_count']} / {raid_near_dup_results['total_test_rows']} ({raid_near_dup_results['near_duplicate_ratio']*100:.2f}%)")
    print(f"  Macro F1 Before: {raid_f1_before:.4f}")
    print(f"  Macro F1 After:  {raid_f1_after:.4f} (diff: {raid_near_dup_results['macro_f1_diff']:+.4f})")

    # -------------------------------------------------------------------------
    # Save Full Verification Report to JSON
    # -------------------------------------------------------------------------
    full_report = {
        "timestamp": datetime.now().isoformat(),
        "models_verified": {
            "authorship": {
                "path": str(auth_path),
                "type": str(type(auth_model)),
                "loaded_cleanly": True,
            },
            "deception": {
                "path": str(dec_path),
                "type": str(type(dec_model)),
                "loaded_cleanly": True,
            },
        },
        "item_1_top_features": {
            "authorship": auth_top_features,
            "deception": dec_top_features,
        },
        "item_2_misclassified_examples": {
            "authorship_salminen_test": {
                "total_misclassified": len(sal_misc_idx),
                "samples_count": len(sal_misclassified_samples),
                "samples": sal_misclassified_samples,
            },
            "authorship_raid_test": {
                "total_misclassified": len(raid_misc_idx),
                "samples_count": len(raid_misclassified_samples),
                "samples": raid_misclassified_samples,
            },
            "deception_ott": {
                "prediction_source": "5-fold stratified cross-validation out-of-fold (OOF) predictions",
                "note": "Saved model in api/models/deception_classical.joblib fits full Ott dataset with 0 errors; OOF predictions reflect generalization errors.",
                "total_misclassified": len(ott_misc_idx),
                "samples_count": len(ott_misclassified_samples),
                "samples": ott_misclassified_samples,
            },
        },
        "item_3_bootstrap_ci": {
            "salminen_test": sal_bootstrap,
            "raid_test": raid_bootstrap,
        },
        "item_5_near_duplicates": {
            "salminen": sal_near_dup_results,
            "raid": raid_near_dup_results,
        },
    }

    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Saved full verification report to: {REPORT_OUTPUT_PATH.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()

