"""Inspect both independent data sources before transformations and modeling."""

from pathlib import Path
import json
import sys

# Allow running `python scripts/audit_datasets.py` without installing a package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retention_lab.data import load_retail, load_hillstrom  # noqa: E402


def retail_audit():
    df = load_retail(ROOT / "data/raw/online_retail_II.xlsx")
    # Potential duplicates are deliberately *reported*, not dropped.
    signature_cols = ["invoice", "stock_code", "quantity", "invoice_date", "unit_price", "customer_id"]
    duplicate_candidates = int(df.duplicated(subset=signature_cols, keep=False).sum())
    return {
        "source": "UCI Online Retail II",
        "rows": int(len(df)),
        "source_sheet_counts": {str(k): int(v) for k, v in df["source_sheet"].value_counts().items()},
        "first_invoice_date": str(df["invoice_date"].min()),
        "last_invoice_date": str(df["invoice_date"].max()),
        "missing_customer_id_rows": int(df["customer_id"].isna().sum()),
        "missing_invoice_dates": int(df["invoice_date"].isna().sum()),
        "missing_or_invalid_prices": int(df["unit_price"].isna().sum()),
        "cancelled_invoice_rows": int(df["is_cancellation"].sum()),
        "negative_quantity_rows": int(df["is_return_quantity"].sum()),
        "nonpositive_price_rows": int(df["unit_price"].le(0).sum()),
        "possible_duplicate_line_rows": duplicate_candidates,
        "note": "No records automatically removed; review possible sheet overlap and duplicates.",
    }


def experiment_audit():
    df = load_hillstrom(ROOT / "data/raw/hillstrom.csv")
    return {
        "source": "Hillstrom randomized email experiment",
        "rows": int(len(df)),
        "by_treatment": {
            str(arm): {
                "n": int(len(group)),
                "visits": int(group["visit"].sum()),
                "conversions": int(group["conversion"].sum()),
                "conversion_rate": round(float(group["conversion"].mean()), 6),
                "average_spend_per_assigned_customer": round(float(group["spend"].mean()), 6),
            }
            for arm, group in df.groupby("segment")
        },
        "note": "Descriptive only; inferential comparisons and confidence intervals come next.",
    }


if __name__ == "__main__":
    ROOT.joinpath("reports").mkdir(exist_ok=True)
    for filename, audit in (("retail_audit.json", retail_audit), ("hillstrom_audit.json", experiment_audit)):
        try:
            result = audit()
        except (FileNotFoundError, ValueError) as exc:
            print(f"Could not audit {filename}: {exc}")
            continue
        path = ROOT / "reports" / filename
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        print(f"Saved: {path}")
