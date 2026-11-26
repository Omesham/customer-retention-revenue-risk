import csv
import logging
from pathlib import Path

import psycopg

from src.retention_lab.config import load_config
from src.retention_lab.logger import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Load customer revenue risk predictions into PostgreSQL.

    Returns:
        None.

    Raises:
        Exception: If configuration, file reading, or database ingestion fails.
    """
    try:
        # Load environment settings before configuring logging.
        config = load_config()
        configure_logging()

        # Define the required project files.
        root = Path(__file__).resolve().parents[1]
        report_path = root / "reports/final_revenue_at_risk.csv"
        schema_path = root / "sql/schema.sql"

        # Verify that the required files exist.
        for path in [report_path, schema_path]:
            if not path.is_file():
                raise FileNotFoundError(f"File not found: {path}")

        # Insert predictions without overwriting existing records.
        insert_query = """
            INSERT INTO customer_revenue_risk (
                customer_id,
                scoring_date,
                churn_probability,
                estimated_90d_spend,
                revenue_at_risk_90d,
                expected_90d_revenue
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (customer_id, scoring_date)
            DO NOTHING;
        """

        # Read customer predictions from the CSV report.
        with report_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)

            records = [
                (
                    row["customer_id"],
                    row["scoring_date"],
                    row["churn_probability"],
                    row["estimated_90d_spend"],
                    row["revenue_at_risk_90d"],
                    row["expected_90d_revenue"]
                )
                for row in reader
            ]

        # Reject empty prediction reports.
        if not records:
            raise ValueError("Revenue risk report is empty.")

        logger.info("Records read: %s", len(records))

        # Connect using the centralized database configuration.
        with psycopg.connect(**config) as connection:
            with connection.cursor() as cursor:

                # Create the required database objects.
                cursor.execute(schema_path.read_text(encoding="utf-8"))

                # Insert customer predictions.
                cursor.executemany(insert_query, records)
                logger.info("Records inserted: %s", cursor.rowcount)

                # Verify the number of stored records.
                cursor.execute("SELECT COUNT(*) FROM customer_revenue_risk")
                total_records = cursor.fetchone()[0]

                logger.info("Total database records: %s", total_records)

        # Successful transactions are committed automatically.
        logger.info("Database ingestion completed successfully.")

    except Exception:
        # Record failures and their stack traces.
        logger.exception("Database ingestion failed.")
        raise


if __name__ == "__main__":
    main()