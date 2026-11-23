import pandas as pd


def build_orders(df):
    # Keep valid purchases only.
    items = df.dropna(subset=["customer_id", "invoice", "invoice_date"]).copy()

    items["invoice"] = items["invoice"].astype(str).str.strip()
    items["invoice_date"] = pd.to_datetime(items["invoice_date"])

    items = items.loc[(items["quantity"] > 0) & (items["unit_price"] > 0) & (~items["invoice"].str.upper().str.startswith("C"))].copy()

    # Calculate revenue for each invoice line.
    items["line_spend"] = (items["quantity"] * items["unit_price"])

    # Each customer/invoice combination becomes one order.
    orders = items.groupby(["customer_id", "invoice"], as_index=False).agg(
        order_date=("invoice_date", "min"),
        order_spend=("line_spend", "sum"),
        order_units=("quantity", "sum"),
        order_lines=("invoice", "size")
    )

    return orders.sort_values(["customer_id", "order_date"]).reset_index(drop=True)