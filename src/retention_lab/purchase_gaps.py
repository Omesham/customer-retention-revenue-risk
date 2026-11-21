def calculate_purchase_gaps(df):
    # Keep one record per customer, invoice, and purchase date.
    purchases = df[["customer_id", "invoice", "invoice_date"]].drop_duplicates()

    # Sort each customer's purchases chronologically.
    purchases = purchases.sort_values(["customer_id", "invoice_date"])

    # Find each customer's previous purchase date.
    purchases["previous_purchase"] = purchases.groupby("customer_id")["invoice_date"].shift(1)

    # Calculate the gap between consecutive purchases in days.
    purchases["gap_days"] = (purchases["invoice_date"] - purchases["previous_purchase"]).dt.total_seconds() / 86400

    # Return purchase gaps, excluding the first purchase.
    return purchases["gap_days"].dropna()