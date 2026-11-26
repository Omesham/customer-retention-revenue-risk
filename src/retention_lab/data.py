"""Validate source schemas without silently dropping observations."""

from pathlib import Path

import pandas as pd

RETAIL_REQUIRED = {
    "invoice", "stock_code", "quantity", "invoice_date", "unit_price", "customer_id"
}
HILLSTROM_REQUIRED = {"segment", "visit", "conversion", "spend"}
# These are distinct randomized arms. Do not accidentally collapse both email types.
EXPERIMENT_ARMS = {"Mens E-Mail", "Womens E-Mail", "No E-Mail"}


# Both UCI Excel sheets use legacy headings; this also accepts newer dataset variants.
COLUMN_ALIASES = {
    "invoice": "invoice",
    "invoiceno": "invoice",
    "stockcode": "stock_code",
    "quantity": "quantity",
    "invoicedate": "invoice_date",
    "price": "unit_price",
    "unitprice": "unit_price",
    "customerid": "customer_id",
    "description": "description",
    "country": "country",
}


def load_retail(path: Path) -> pd.DataFrame:
    """Load *all* workbook sheets, retaining source provenance and return flags."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing Retail II workbook: {path}")

    # Every sheet is preserved, including potential date overlaps across sheets.
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    if not sheets:
        raise ValueError("Retail II workbook has no sheets")

    frames = []
    for sheet, data in sheets.items():
        # Normalize spaces, punctuation and capitalization for schema matching.
        normalized = {
            col: COLUMN_ALIASES.get("".join(ch for ch in str(col).lower() if ch.isalnum()))
            for col in data.columns
        }
        data = data.rename(columns={k: v for k, v in normalized.items() if v})
        missing = RETAIL_REQUIRED - set(data.columns)
        if missing:
            raise ValueError(f"Retail sheet {sheet!r} lacks required columns: {sorted(missing)}")

        data = data.copy()
        data["source_sheet"] = sheet
        data["invoice"] = data["invoice"].astype("string").str.strip()
        data["invoice_date"] = pd.to_datetime(data["invoice_date"], errors="coerce")
        data["quantity"] = pd.to_numeric(data["quantity"], errors="coerce")
        data["unit_price"] = pd.to_numeric(data["unit_price"], errors="coerce")
        data["customer_id"] = (
            data["customer_id"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
        )
        # Cancelled invoices conventionally start with C; negative quantities may also be returns.
        data["is_cancellation"] = data["invoice"].str.upper().str.startswith("C").fillna(False)
        data["is_return_quantity"] = data["quantity"] < 0
        frames.append(data)

    # Do NOT deduplicate here: identical line items can be valid and sheet overlap needs inspection.
    return pd.concat(frames, ignore_index=True)


def load_hillstrom(path: Path) -> pd.DataFrame:
    """Read randomized experiment data and explicitly verify treatment groups/outcomes."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing Hillstrom CSV: {path}")
    data = pd.read_csv(path)
    data.columns = [str(c).strip().lower() for c in data.columns]

    missing = HILLSTROM_REQUIRED - set(data.columns)
    if missing:
        raise ValueError(f"Hillstrom CSV lacks required columns: {sorted(missing)}")

    data["segment"] = data["segment"].astype("string").str.strip()
    present = set(data["segment"].dropna().unique())
    if not present <= EXPERIMENT_ARMS or present != EXPERIMENT_ARMS:
        raise ValueError(f"Expected all 3 Hillstrom arms {sorted(EXPERIMENT_ARMS)}; found {sorted(present)}")
    for col in ("visit", "conversion", "spend"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if data[["visit", "conversion", "spend"]].isna().any().any():
        raise ValueError("Hillstrom outcomes contain missing or invalid values")
    if not data["visit"].isin([0, 1]).all() or not data["conversion"].isin([0, 1]).all():
        raise ValueError("Hillstrom visit/conversion must be binary")
    if (data["spend"] < 0).any():
        raise ValueError("Hillstrom spend cannot be negative")
    return data
