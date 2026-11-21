import pandas as pd

from src.retention_lab.preprocessing import remove_duplicate_lines


def test_remove_duplicate_lines():
    # Two identical lines in invoice 100; invoice 101 is a separate reorder.
    df = pd.DataFrame({
        "invoice": ["100", "100", "101"],
        "stock_code": ["A", "A", "A"],
        "quantity": [2, 2, 2],
        "invoice_date": pd.to_datetime([
            "2026-01-01", "2026-01-01", "2026-01-05"
        ]),
        "unit_price": [5.0, 5.0, 5.0],
        "customer_id": ["42", "42", "42"],
    })

    # Remove the duplicate while preserving the separate reorder.
    result = remove_duplicate_lines(df)

    assert len(result) == 2
    assert result["invoice"].tolist() == ["100", "101"]