import json
import os
from datetime import datetime
from pathlib import Path

# Matplotlib cache if needed
mpl_cache = Path("ml/reports/.matplotlib")
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_cache.resolve())

import joblib
import numpy as np
import pandas as pd

from ml.src.data.cleaning import clean_text

SYNTHETIC_DATA_PATH = Path("data/processed/synthetic_hotel_ai.csv")
OTT_DATA_PATH = Path("data/processed/ott_deception_5fold.csv")
MODEL_PATH = Path("api/models/authorship_classical_v2.joblib")
REPORT_PATH = Path("ml/reports/ott_positive_control_v2.json")

def main():
    print("=" * 80)
    print("ITEM 6: RE-RUN OTT POSITIVE CONTROL & OTT TEST (v2 Authorship Model)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    # 1. Load model
    print(f"Loading best saved v2 authorship model from: {MODEL_PATH.resolve()}")
    model = joblib.load(MODEL_PATH)
    print(f"  Model loaded: {type(model)} with steps: {list(model.named_steps.keys())}")

    # 2. Evaluate on synthetic_hotel_ai.csv (200 rows)
    print(f"\nLoading synthetic hotel AI dataset from: {SYNTHETIC_DATA_PATH.resolve()}")
    synth_df = pd.read_csv(SYNTHETIC_DATA_PATH)
    synth_df["text_clean"] = synth_df["text"].apply(clean_text)

    synth_preds = model.predict(synth_df["text_clean"])
    synth_probs = model.predict_proba(synth_df["text_clean"])[:, 1]
    synth_df["pred_prob_ai"] = synth_probs
    synth_df["is_pred_ai"] = (synth_preds == 1)

    total_synth = len(synth_df)
    synth_ai_count = int(synth_df["is_pred_ai"].sum())
    synth_ai_pct = float(synth_ai_count / total_synth * 100.0)
    synth_mean_prob = float(np.mean(synth_probs))
    synth_median_prob = float(np.median(synth_probs))

    pos_df = synth_df[synth_df["sentiment"] == "positive"]
    pos_total = len(pos_df)
    pos_ai_count = int(pos_df["is_pred_ai"].sum())
    pos_ai_pct = float(pos_ai_count / pos_total * 100.0)
    pos_mean_prob = float(np.mean(pos_df["pred_prob_ai"]))
    pos_median_prob = float(np.median(pos_df["pred_prob_ai"]))

    neg_df = synth_df[synth_df["sentiment"] == "negative"]
    neg_total = len(neg_df)
    neg_ai_count = int(neg_df["is_pred_ai"].sum())
    neg_ai_pct = float(neg_ai_count / neg_total * 100.0)
    neg_mean_prob = float(np.mean(neg_df["pred_prob_ai"]))
    neg_median_prob = float(np.median(neg_df["pred_prob_ai"]))

    # 3. Evaluate on all 1,596 Ott rows
    print(f"\nLoading Ott dataset from: {OTT_DATA_PATH.resolve()}")
    ott_df = pd.read_csv(OTT_DATA_PATH)
    ott_df["text_clean"] = ott_df["text"].apply(clean_text)

    ott_preds = model.predict(ott_df["text_clean"])
    ott_probs = model.predict_proba(ott_df["text_clean"])[:, 1]
    ott_df["pred_prob_ai"] = ott_probs
    ott_df["is_pred_ai"] = (ott_preds == 1)

    total_ott = len(ott_df)
    ott_ai_count = int(ott_df["is_pred_ai"].sum())
    ott_ai_pct = float(ott_ai_count / total_ott * 100.0)
    ott_mean_prob = float(np.mean(ott_probs))
    ott_median_prob = float(np.median(ott_probs))

    truthful_df = ott_df[ott_df["deception_label"] == "genuine"]
    truthful_total = len(truthful_df)
    truthful_ai_count = int(truthful_df["is_pred_ai"].sum())
    truthful_ai_pct = float(truthful_ai_count / truthful_total * 100.0)
    truthful_mean_prob = float(np.mean(truthful_df["pred_prob_ai"]))
    truthful_median_prob = float(np.median(truthful_df["pred_prob_ai"]))

    deceptive_df = ott_df[ott_df["deception_label"] == "deceptive"]
    deceptive_total = len(deceptive_df)
    deceptive_ai_count = int(deceptive_df["is_pred_ai"].sum())
    deceptive_ai_pct = float(deceptive_ai_count / deceptive_total * 100.0)
    deceptive_mean_prob = float(np.mean(deceptive_df["pred_prob_ai"]))
    deceptive_median_prob = float(np.median(deceptive_df["pred_prob_ai"]))

    # Statistical power assessment
    has_statistical_power = synth_ai_pct >= 50.0
    power_statement = (
        f"Based on the positive control detection rate of {synth_ai_pct:.2f}% (mean P(AI) = {synth_mean_prob:.4f}), "
        "the model exhibits very low sensitivity to AI-generated text in the hotel domain. "
        "Consequently, the Ott independence test lacks statistical power, and the low predicted AI rate on Ott "
        f"({ott_ai_pct:.2f}%) cannot be interpreted as confirming independence."
        if not has_statistical_power else
        f"The model detected {synth_ai_pct:.2f}% of synthetic AI reviews, demonstrating adequate statistical power."
    )

    report_data = {
        "timestamp": datetime.now().isoformat(),
        "model_version": "v2",
        "model_path": str(MODEL_PATH),
        "synthetic_positive_control": {
            "dataset_path": str(SYNTHETIC_DATA_PATH),
            "total_reviews": total_synth,
            "overall": {
                "predicted_ai_count": synth_ai_count,
                "predicted_ai_percentage": synth_ai_pct,
                "mean_predicted_prob_ai": synth_mean_prob,
                "median_predicted_prob_ai": synth_median_prob,
            },
            "positive_sentiment": {
                "total_reviews": pos_total,
                "predicted_ai_count": pos_ai_count,
                "predicted_ai_percentage": pos_ai_pct,
                "mean_predicted_prob_ai": pos_mean_prob,
                "median_predicted_prob_ai": pos_median_prob,
            },
            "negative_sentiment": {
                "total_reviews": neg_total,
                "predicted_ai_count": neg_ai_count,
                "predicted_ai_percentage": neg_ai_pct,
                "mean_predicted_prob_ai": neg_mean_prob,
                "median_predicted_prob_ai": neg_median_prob,
            },
        },
        "ott_held_out_evaluation": {
            "dataset_path": str(OTT_DATA_PATH),
            "total_reviews": total_ott,
            "overall": {
                "predicted_ai_count": ott_ai_count,
                "predicted_ai_percentage": ott_ai_pct,
                "mean_predicted_prob_ai": ott_mean_prob,
                "median_predicted_prob_ai": ott_median_prob,
            },
            "truthful_genuine": {
                "total_reviews": truthful_total,
                "predicted_ai_count": truthful_ai_count,
                "predicted_ai_percentage": truthful_ai_pct,
                "mean_predicted_prob_ai": truthful_mean_prob,
                "median_predicted_prob_ai": truthful_median_prob,
            },
            "deceptive": {
                "total_reviews": deceptive_total,
                "predicted_ai_count": deceptive_ai_count,
                "predicted_ai_percentage": deceptive_ai_pct,
                "mean_predicted_prob_ai": deceptive_mean_prob,
                "median_predicted_prob_ai": deceptive_median_prob,
            },
        },
        "statistical_power_assessment": {
            "has_statistical_power": has_statistical_power,
            "statement": power_statement,
        }
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print("\n" + "=" * 80)
    print("RESULTS: POSITIVE CONTROL ON SYNTHETIC HOTEL AI REVIEWS (v2)")
    print("=" * 80)
    print(f"Overall Synthetic AI Reviews (N={total_synth}):")
    print(f"  Predicted as AI: {synth_ai_count} / {total_synth} ({synth_ai_pct:.2f}%)")
    print(f"  Mean P(AI):      {synth_mean_prob:.4f}")
    print(f"  Median P(AI):    {synth_median_prob:.4f}")

    print(f"\nPositive Sentiment Reviews (N={pos_total}):")
    print(f"  Predicted as AI: {pos_ai_count} / {pos_total} ({pos_ai_pct:.2f}%)")
    print(f"  Mean P(AI):      {pos_mean_prob:.4f}")
    print(f"  Median P(AI):    {pos_median_prob:.4f}")

    print(f"\nNegative Sentiment Reviews (N={neg_total}):")
    print(f"  Predicted as AI: {neg_ai_count} / {neg_total} ({neg_ai_pct:.2f}%)")
    print(f"  Mean P(AI):      {neg_mean_prob:.4f}")
    print(f"  Median P(AI):    {neg_median_prob:.4f}")

    print("\n" + "=" * 80)
    print("RESULTS: OTT HELD-OUT ROWS EVALUATION (v2 Authorship Model)")
    print("=" * 80)
    print(f"Overall Ott Rows (N={total_ott}):")
    print(f"  Predicted as AI: {ott_ai_count} / {total_ott} ({ott_ai_pct:.2f}%)")
    print(f"  Mean P(AI):      {ott_mean_prob:.4f}")
    print(f"  Median P(AI):    {ott_median_prob:.4f}")

    print(f"\nTruthful / Genuine Rows (N={truthful_total}):")
    print(f"  Predicted as AI: {truthful_ai_count} / {truthful_total} ({truthful_ai_pct:.2f}%)")
    print(f"  Mean P(AI):      {truthful_mean_prob:.4f}")
    print(f"  Median P(AI):    {truthful_median_prob:.4f}")

    print(f"\nDeceptive Rows (N={deceptive_total}):")
    print(f"  Predicted as AI: {deceptive_ai_count} / {deceptive_total} ({deceptive_ai_pct:.2f}%)")
    print(f"  Mean P(AI):      {deceptive_mean_prob:.4f}")
    print(f"  Median P(AI):    {deceptive_median_prob:.4f}")

    print("\n" + "=" * 80)
    print("STATISTICAL POWER ASSESSMENT:")
    print("=" * 80)
    print(power_statement)
    print("=" * 80)
    print(f"Report saved to: {REPORT_PATH.resolve()}")

if __name__ == "__main__":
    main()
