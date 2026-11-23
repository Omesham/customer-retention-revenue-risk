from pathlib import Path
import pandas as pd

# Read only the customer and invoice columns from both workbook sheets.
root = Path(__file__).resolve().parents[1]
path = root / "data/raw/online_retail_II.xlsx"
sheets = pd.read_excel(path, sheet_name=None, usecols=["Invoice", "Customer ID"])

invoice_sets = []

for sheet_name, df in sheets.items():
    # Identify unique customer/invoice combinations in each sheet.
    df = df.dropna(subset=["Invoice", "Customer ID"]).copy()
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    pairs = set(zip(df["Customer ID"], df["Invoice"]))
    invoice_sets.append(pairs)
    print(f"{sheet_name}: {len(pairs):,} unique customer invoices")

# Find invoices belonging to the same customer on both sheets.
overlap = invoice_sets[0] & invoice_sets[1]
print(f"Customer invoices appearing on both sheets: {len(overlap):,}")
print(f"Examples: {list(sorted(overlap))[:5]}")