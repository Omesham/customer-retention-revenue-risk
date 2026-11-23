def remove_missing_customer_ids(df):
    # Keep the original DataFrame unchanged.
    cleaned_df = df.dropna(subset=["customer_id"]).copy()

    return cleaned_df


def remove_cancellations(df):
    # Keep transactions that are not cancellations or returns.
    return df.loc[
        ~df["is_cancellation"] & ~df["is_return_quantity"]
    ].copy()

def remove_nonpositive_prices(df):
    # Keep only transactions with positive prices.
    return df.loc[df["unit_price"] > 0].copy()

def remove_duplicate_lines(df):
    # These columns identify the contents of an invoice line.
    columns = ["invoice", "stock_code", "quantity", "invoice_date", "unit_price", "customer_id"]

    # The source sheet is needed to distinguish cross-sheet duplicates.
    if "source_sheet" not in df.columns:
        raise ValueError("Missing source_sheet column.")

    data = df.copy()

    # Number identical occurrences within each sheet, starting at zero.
    # This preserves legitimate repeated invoice lines.
    data["_occurrence"] = data.groupby(["source_sheet"] + columns, dropna=False).cumcount()

    # Remove matching occurrences from other sheets, keeping one copy.
    before = len(data)
    data = data.drop_duplicates(subset=columns + ["_occurrence"], keep="first")

    print(f"Cross-sheet duplicate lines removed: {before - len(data):,}")

    # Remove the temporary column before returning the cleaned data.
    return data.drop(columns=["_occurrence"]).copy()

def clean_retail(df):
    """Prepare retail transactions for customer-level modeling."""

    # Exclude transactions without customer IDs.
    df = remove_missing_customer_ids(df)

    # Exclude cancellations and returns.
    df = remove_cancellations(df)

    # Exclude zero or negative prices.
    df = remove_nonpositive_prices(df)

    # Remove repeated transaction lines.
    df = remove_duplicate_lines(df)

    # Return the cleaned dataset.
    return df