def validate_retail_schema(df):
    # Required columns after loading and standardizing the dataset.
    required = {
        "invoice", "stock_code", "quantity",
        "invoice_date", "unit_price", "customer_id"
    }

    # Identify missing columns.
    missing = required - set(df.columns)

    # Stop if the dataset does not match our expected schema.
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")


def check_missing_values(df):
    return df.isna().sum().to_dict()
    

def check_returns(df):
    # Count cancelled invoices and negative-quantity transactions.
    return {
        "cancelled_rows": int(df["is_cancellation"].sum()),
        "negative_quantity_rows": int(df["is_return_quantity"].sum()),
    }

def inspect_negative_quantities(df):
    # Return transactions with negative quantities for inspection.
    return df.loc[
        df["quantity"] < 0,
        ["invoice", "stock_code", "quantity", "unit_price", "is_cancellation"]
    ].head(10)

def check_invalid_prices(df):
    # Count transactions with zero or negative prices.
    return {
        "zero_prices": int(df["unit_price"].eq(0).sum()),
        "negative_prices": int(df["unit_price"].lt(0).sum()),
    }

def check_duplicates(df):
    # Compare transaction details to identify possible duplicates.
    columns = [
        "invoice", "stock_code", "quantity",
        "invoice_date", "unit_price", "customer_id"
    ]

    # Return all matching rows for inspection.
    return df.loc[df.duplicated(subset=columns, keep=False)]