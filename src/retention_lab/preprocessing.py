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