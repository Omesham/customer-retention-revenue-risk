-- Store customer-level churn and revenue predictions.
CREATE TABLE IF NOT EXISTS customer_revenue_risk (
    customer_id VARCHAR(50) NOT NULL,
    scoring_date DATE NOT NULL,
    churn_probability DOUBLE PRECISION NOT NULL,
    estimated_90d_spend NUMERIC(18, 2) NOT NULL,
    revenue_at_risk_90d NUMERIC(18, 2) NOT NULL,
    expected_90d_revenue NUMERIC(18, 2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    -- Prevent duplicate predictions for the same customer and date.
    PRIMARY KEY (customer_id, scoring_date),

    -- Ensure churn probabilities are valid.
    CHECK (churn_probability BETWEEN 0 AND 1)
);

-- Speed up queries that identify high-risk customers.
CREATE INDEX IF NOT EXISTS idx_revenue_risk
ON customer_revenue_risk (revenue_at_risk_90d DESC);