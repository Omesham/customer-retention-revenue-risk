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
    # Identify transaction lines using their invoice and product details.
    columns = [
        "invoice", "stock_code", "quantity",
        "invoice_date", "unit_price", "customer_id"
    ]

    # Keep the first occurrence of each matching transaction line.
    # Preserve the original DataFrame.
    return df.drop_duplicates(subset=columns, keep="first").copy()

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