from collections import Counter
from pathlib import Path
import pandas as pd

# Load the fields needed to compare actual invoice contents.
root = Path(__file__).resolve().parents[1]
path = root / "data/raw/online_retail_II.xlsx"
columns = ["Customer ID", "Invoice", "InvoiceDate", "StockCode", "Quantity", "Price"]
sheets = pd.read_excel(path, sheet_name=None, usecols=columns)

keys = ["Customer ID", "Invoice"]
signatures = []

for sheet_name, df in sheets.items():
    # Standardize invoice identifiers before comparing sheets.
    df = df.dropna(subset=keys).copy()
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].fillna("").astype(str)

    # Preserve repeated line items when comparing invoice contents.
    grouped = {
        key: Counter(group[["InvoiceDate", "StockCode", "Quantity", "Price"]].itertuples(index=False, name=None))
        for key, group in df.groupby(keys)
    }

    signatures.append(grouped)

# Compare every customer/invoice combination found on both sheets.
overlap = signatures[0].keys() & signatures[1].keys()
identical = [key for key in overlap if signatures[0][key] == signatures[1][key]]
different = [key for key in overlap if signatures[0][key] != signatures[1][key]]

print(f"Overlapping invoices: {len(overlap):,}")
print(f"Identical invoices: {len(identical):,}")
print(f"Different invoice contents: {len(different):,}")
print(f"Examples requiring investigation: {sorted(different)[:5]}")