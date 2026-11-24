import csv
import os
import logging
from pathlib import Path

import psycopg


def main() -> None:
    """
    Create the PostgreSQL table and load customer revenue
    risk predictions from the final CSV report.

    Returns:
        None.
    """

    # Define project paths.
    root = Path(__file__).resolve().parents[1]
    report_path = root / "reports/final_revenue_at_risk.csv"
    schema_path = root / "sql/schema.sql"
    log_dir = root / "logs"

    # Create the logging directory if necessary.
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure terminal and file logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "load_revenue_risk.log"),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    try:
        # Retrieve database credentials from environment variables.
        connection_string = os.environ["DATABASE_URL"]

        # Verify that the required files exist.
        for path in [report_path, schema_path]:
            if not path.is_file():
                raise FileNotFoundError(f"File not found: {path}")

        # Define the SQL statement for inserting predictions.
        # Existing customer predictions will not be overwritten.
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

            # Prepare records in the same order as the SQL columns.
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

        logger.info("Records read: %s", len(records))

        # Establish a PostgreSQL connection.
        # Changes are committed on success or rolled back on failure.
        with psycopg.connect(connection_string) as connection:
            with connection.cursor() as cursor:

                # Create the database table and index.
                cursor.execute(schema_path.read_text(encoding="utf-8"))

                # Insert customer predictions in batches.
                cursor.executemany(insert_query, records)

                logger.info("Records inserted: %s", cursor.rowcount)

                # Verify the total number of stored records.
                cursor.execute("SELECT COUNT(*) FROM customer_revenue_risk")

                total_records = cursor.fetchone()[0]

                logger.info("Total database records: %s", total_records)

        logger.info("Database ingestion completed successfully.")

    except Exception:
        # Log the error and its stack trace.
        logger.exception("Database ingestion failed.")
        raise


# Execute the ingestion process.
if __name__ == "__main__":
    main()