
import pandas as pd


def remove_cross_sheet_duplicates(df, workbook_path):
    # Identify invoices appearing on both workbook sheets.
    sheets = pd.read_excel(workbook_path, sheet_name=None, usecols=["Customer ID", "Invoice"])
    invoice_sets = []

    for sheet in sheets.values():
        sheet = sheet.dropna(subset=["Customer ID", "Invoice"]).copy()
        customers = sheet["Customer ID"].astype(str).str.replace(r"\.0$", "", regex=True)
        invoices = sheet["Invoice"].astype(str).str.strip()
        invoice_sets.append(set(zip(customers, invoices)))

    # Find customer/invoice combinations shared by both sheets.
    overlap = invoice_sets[0] & invoice_sets[1]
    print(f"Overlapping customer invoices: {len(overlap):,}")

    # Find matching invoices in our cleaned transaction data.
    data = df.reset_index(drop=True).copy()
    customers = data["customer_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    invoices = data["invoice"].astype(str).str.strip()

    mask = pd.MultiIndex.from_arrays([customers, invoices]).isin(overlap)
    matched = data.loc[mask]

    # Group identical invoice lines to count their occurrences.
    line_columns = ["customer_id", "invoice", "invoice_date", "stock_code", "quantity", "unit_price"]
    groups = matched.groupby(line_columns, dropna=False)
    counts = groups["invoice"].transform("size")

    # Investigate lines with odd counts before deleting anything.
    if (counts % 2 != 0).any():
        odd_lines = matched.loc[counts % 2 != 0, line_columns]

        print(f"Rows with odd counts: {len(odd_lines):,}")
        print("Line occurrence counts:")
        print(counts.value_counts().sort_index())
        print("Sample problematic lines:")
        print(odd_lines.head(5).to_string(index=False))

        raise ValueError("Duplicate counts changed after cleaning.")

    # Remove the extra copies while preserving legitimate repeated lines.
    duplicates = matched.index[groups.cumcount() >= counts // 2]
    print(f"Removed {len(duplicates):,} duplicated invoice lines")

    return data.drop(index=duplicates).reset_index(drop=True)
