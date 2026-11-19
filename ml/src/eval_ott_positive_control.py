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

DATA_PATH = Path("data/processed/synthetic_hotel_ai.csv")
MODEL_PATH = Path("api/models/authorship_classical.joblib")
REPORT_PATH = Path("ml/reports/ott_positive_control_v1.json")

def main():
    print("=" * 80)
    print("ITEM 6: POSITIVE CONTROL FOR OTT INDEPENDENCE TEST")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    # 1. Load model and data
    print(f"Loading best saved authorship model from: {MODEL_PATH.resolve()}")
    model = joblib.load(MODEL_PATH)
    print(f"  Model loaded: {type(model)} with steps: {list(model.named_steps.keys())}")

    print(f"Loading synthetic hotel AI dataset from: {DATA_PATH.resolve()}")
    df = pd.read_csv(DATA_PATH)
    print(f"  Dataset loaded: {len(df)} rows ({len(df[df['sentiment']=='positive'])} positive, {len(df[df['sentiment']=='negative'])} negative)")

    # 2. Run predictions
    preds = model.predict(df["text"])
    probs = model.predict_proba(df["text"])[:, 1]

    df["pred_label"] = np.where(preds == 1, "ai", "human")
    df["pred_prob_ai"] = probs
    df["is_pred_ai"] = (preds == 1)

    # 3. Compute metrics
    # Overall
    total_count = len(df)
    ai_count = int(df["is_pred_ai"].sum())
    ai_pct = float(ai_count / total_count * 100.0)
    mean_prob = float(np.mean(probs))
    median_prob = float(np.median(probs))

    # Positive-only
    pos_df = df[df["sentiment"] == "positive"]
    pos_total = len(pos_df)
    pos_ai_count = int(pos_df["is_pred_ai"].sum())
    pos_ai_pct = float(pos_ai_count / pos_total * 100.0)
    pos_mean_prob = float(np.mean(pos_df["pred_prob_ai"]))
    pos_median_prob = float(np.median(pos_df["pred_prob_ai"]))

    # Negative-only
    neg_df = df[df["sentiment"] == "negative"]
    neg_total = len(neg_df)
    neg_ai_count = int(neg_df["is_pred_ai"].sum())
    neg_ai_pct = float(neg_ai_count / neg_total * 100.0)
    neg_mean_prob = float(np.mean(neg_df["pred_prob_ai"]))
    neg_median_prob = float(np.median(neg_df["pred_prob_ai"]))

    # Interpretation
    is_low_detection = ai_pct < 50.0
    interpretation_note = (
        "The AI detection rate on known synthetic hotel reviews is low "
        f"({ai_pct:.2f}% detected as AI, mean P(AI) = {mean_prob:.4f}). "
        "Because the classical authorship model fails to reliably identify AI-generated text in the hotel domain, "
        "the Ott independence test lacks statistical power and the 0.19% result cannot be interpreted as confirming independence."
        if is_low_detection else
        f"The AI detection rate on known synthetic hotel reviews is {ai_pct:.2f}%."
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(MODEL_PATH),
        "dataset_path": str(DATA_PATH),
        "overall": {
            "total_reviews": total_count,
            "predicted_ai_count": ai_count,
            "predicted_ai_percentage": ai_pct,
            "mean_predicted_prob_ai": mean_prob,
            "median_predicted_prob_ai": median_prob,
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
        "statistical_power_assessment": {
            "is_detection_rate_low": is_low_detection,
            "interpretation": interpretation_note
        }
    }

    # Save to report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 4. Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS: POSITIVE CONTROL (200 Synthetic Hotel AI Reviews)")
    print("=" * 80)
    print(f"Overall Synthetic AI Reviews (N={total_count}):")
    print(f"  Predicted as AI: {ai_count} / {total_count} ({ai_pct:.2f}%)")
    print(f"  Mean P(AI):      {mean_prob:.4f}")
    print(f"  Median P(AI):    {median_prob:.4f}")

    print(f"\nPositive Sentiment Reviews (N={pos_total}):")
    print(f"  Predicted as AI: {pos_ai_count} / {pos_total} ({pos_ai_pct:.2f}%)")
    print(f"  Mean P(AI):      {pos_mean_prob:.4f}")
    print(f"  Median P(AI):    {pos_median_prob:.4f}")

    print(f"\nNegative Sentiment Reviews (N={neg_total}):")
    print(f"  Predicted as AI: {neg_ai_count} / {neg_total} ({neg_ai_pct:.2f}%)")
    print(f"  Mean P(AI):      {neg_mean_prob:.4f}")
    print(f"  Median P(AI):    {neg_median_prob:.4f}")

    print("\n" + "=" * 80)
    print("STATISTICAL POWER AUDIT FINDING:")
    print("=" * 80)
    print(interpretation_note)
    print("=" * 80)
    print(f"Results saved to: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
