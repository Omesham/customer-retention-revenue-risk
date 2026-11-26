import pandas as pd

from src.retention_lab.preprocessing import remove_duplicate_lines


def test_remove_duplicate_lines():
    """Verify cross-sheet duplicate removal while preserving valid transactions."""

    df = pd.DataFrame({
        "invoice": ["100", "100", "100", "101"],
        "stock_code": ["A", "A", "A", "A"],
        "quantity": [2, 2, 2, 2],
        "invoice_date": pd.to_datetime([
            "2026-01-01", "2026-01-01",
            "2026-01-01", "2026-01-05"
        ]),
        "unit_price": [5.0, 5.0, 5.0, 5.0],
        "customer_id": ["42", "42", "42", "42"],
        "source_sheet": [
            "Year 2009-2010",
            "Year 2009-2010",
            "Year 2010-2011",
            "Year 2010-2011"
        ]
    })

    # Remove cross-sheet duplicates.
    result = remove_duplicate_lines(df)

    # Preserve both original invoice lines and the separate reorder.
    assert len(result) == 3
    assert (result["invoice"] == "100").sum() == 2
    assert (result["invoice"] == "101").sum() == 1