import pandas as pd


def split_scoring_window(orders, scoring_date):
    # Convert the scoring date into a pandas datetime.
    scoring_date = pd.Timestamp(scoring_date)

    # Use past orders to calculate customer features.
    history = orders[orders["order_date"] < scoring_date].copy()

    # Use the next 90 days of orders to determine churn.
    future = orders[(orders["order_date"] >= scoring_date) & (orders["order_date"] < scoring_date + pd.Timedelta(days=90))].copy()

    return history, future