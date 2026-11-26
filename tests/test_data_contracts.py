"""Minimal reproducible input contracts; no private or external data required."""

import pandas as pd
import pytest

from retention_lab.data import load_retail, load_hillstrom


def test_retail_keeps_all_sheets_and_cancelled_transactions(tmp_path):
    # Include a valid order and a cancellation; neither must be silently dropped.
    columns = ["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
    first = pd.DataFrame([["10001", "A", 2, "2010-01-02", 5.0, 12345, "UK"]], columns=columns)
    second = pd.DataFrame([["C10002", "B", -1, "2011-01-02", 7.5, 12345, "UK"]], columns=columns)
    workbook = tmp_path / "retail.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        first.to_excel(writer, sheet_name="Year 2009-2010", index=False)
        second.to_excel(writer, sheet_name="Year 2010-2011", index=False)
    data = load_retail(workbook)
    assert len(data) == 2
    assert data["source_sheet"].nunique() == 2
    assert data["is_cancellation"].sum() == 1
    assert data["is_return_quantity"].sum() == 1
    assert set(data["customer_id"]) == {"12345"}


def test_hillstrom_preserves_three_randomized_groups(tmp_path):
    csv = tmp_path / "experiment.csv"
    pd.DataFrame({
        "segment": ["Mens E-Mail", "Womens E-Mail", "No E-Mail"],
        "visit": [1, 0, 1], "conversion": [1, 0, 0], "spend": [15.0, 0.0, 0.0]
    }).to_csv(csv, index=False)
    data = load_hillstrom(csv)
    assert len(data) == 3
    assert data["segment"].nunique() == 3


def test_missing_experimental_control_is_an_error(tmp_path):
    csv = tmp_path / "no_control.csv"
    pd.DataFrame({
        "segment": ["Mens E-Mail", "Womens E-Mail"],
        "visit": [1, 0], "conversion": [1, 0], "spend": [10.0, 0.0]
    }).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Expected all 3"):
        load_hillstrom(csv)
