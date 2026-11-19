import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from ml.src.data.cleaning import has_html_br, has_chat_preamble

DATA_DIR = Path("data/processed")
REPORTS_DIR = Path("ml/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = REPORTS_DIR / "cleaning_audit_v2.json"

FILES_TO_AUDIT = {
    "train.csv": DATA_DIR / "train.csv",
    "val.csv": DATA_DIR / "val.csv",
    "test.csv": DATA_DIR / "test.csv",
    "ott_deception_5fold.csv": DATA_DIR / "ott_deception_5fold.csv",
    "unified_dataset.csv": DATA_DIR / "unified_dataset.csv",
}

def audit_file(file_path: Path) -> dict:
    df = pd.read_csv(file_path)
    total_rows = len(df)

    df["has_br"] = df["text"].apply(has_html_br)
    df["has_preamble"] = df["text"].apply(has_chat_preamble)

    total_br = int(df["has_br"].sum())
    total_preamble = int(df["has_preamble"].sum())

    # Breakdown by source_dataset and authorship_label
    br_by_source_and_authorship = {}
    preamble_by_source_and_authorship = {}

    if "source_dataset" in df.columns and "authorship_label" in df.columns:
        for (src, auth), sub in df.groupby(["source_dataset", "authorship_label"]):
            key = f"{src}_{auth}"
            br_count = int(sub["has_br"].sum())
            preamble_count = int(sub["has_preamble"].sum())
            br_by_source_and_authorship[key] = {
                "source_dataset": src,
                "authorship_label": auth,
                "total_rows": len(sub),
                "br_count": br_count,
                "br_percentage": float(br_count / len(sub) * 100.0) if len(sub) > 0 else 0.0,
            }
            preamble_by_source_and_authorship[key] = {
                "source_dataset": src,
                "authorship_label": auth,
                "total_rows": len(sub),
                "preamble_count": preamble_count,
                "preamble_percentage": float(preamble_count / len(sub) * 100.0) if len(sub) > 0 else 0.0,
            }

    # Breakdown by deception_label if present
    br_by_deception = {}
    preamble_by_deception = {}
    if "deception_label" in df.columns:
        for dec, sub in df.groupby("deception_label"):
            br_count = int(sub["has_br"].sum())
            preamble_count = int(sub["has_preamble"].sum())
            br_by_deception[str(dec)] = {
                "deception_label": str(dec),
                "total_rows": len(sub),
                "br_count": br_count,
                "br_percentage": float(br_count / len(sub) * 100.0) if len(sub) > 0 else 0.0,
            }
            preamble_by_deception[str(dec)] = {
                "deception_label": str(dec),
                "total_rows": len(sub),
                "preamble_count": preamble_count,
                "preamble_percentage": float(preamble_count / len(sub) * 100.0) if len(sub) > 0 else 0.0,
            }

    return {
        "file_name": file_path.name,
        "total_rows": total_rows,
        "br_tags": {
            "total_count": total_br,
            "total_percentage": float(total_br / total_rows * 100.0) if total_rows > 0 else 0.0,
            "by_source_authorship": br_by_source_and_authorship,
            "by_deception_label": br_by_deception,
        },
        "chat_preambles": {
            "total_count": total_preamble,
            "total_percentage": float(total_preamble / total_rows * 100.0) if total_rows > 0 else 0.0,
            "by_source_authorship": preamble_by_source_and_authorship,
            "by_deception_label": preamble_by_deception,
        },
    }

def main():
    print("=" * 80)
    print("RUNNING SHARED CLEANING ARTIFACT AUDIT (v2)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    report = {
        "timestamp": datetime.now().isoformat(),
        "audit_results": {},
    }

    for name, path in FILES_TO_AUDIT.items():
        if not path.exists():
            print(f"File not found: {path}")
            continue
        print(f"\nAuditing: {name} ({path.resolve()})...")
        res = audit_file(path)
        report["audit_results"][name] = res
        print(f"  Total rows: {res['total_rows']:,}")
        print(f"  Rows with <br> tags: {res['br_tags']['total_count']:,} ({res['br_tags']['total_percentage']:.2f}%)")
        for k, v in res['br_tags']['by_source_authorship'].items():
            if v['br_count'] > 0:
                print(f"    - {k}: {v['br_count']}/{v['total_rows']} ({v['br_percentage']:.2f}%)")
        print(f"  Rows with chat preambles: {res['chat_preambles']['total_count']:,} ({res['chat_preambles']['total_percentage']:.2f}%)")
        for k, v in res['chat_preambles']['by_source_authorship'].items():
            if v['preamble_count'] > 0:
                print(f"    - {k}: {v['preamble_count']}/{v['total_rows']} ({v['preamble_percentage']:.2f}%)")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Audit report saved to: {OUTPUT_FILE.resolve()}")
    print("=" * 80)

if __name__ == "__main__":
    main()

