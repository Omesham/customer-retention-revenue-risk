# Customer Retention & Revenue Risk — Data Science Portfolio

**Objective:** Predict 90-day purchase inactivity using historical retail transactions, estimate *illustrative* revenue at risk, and separately measure causal campaign lift using a real randomized email experiment.

## Research questions

1. Can historical purchasing behavior predict whether a customer makes **no purchase** during the next 90 days?
2. Which customer behaviors are associated with a higher risk of future purchase inactivity?
3. In a separate randomized experiment, did men's-merchandise or women's-merchandise emails increase conversion or revenue compared with no email?

## Datasets (two independent customer populations)

| Analysis | Source | Data | Primary outcome |
|---|---|---|---|
| Retention prediction | [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | 2009–2011 transaction lines, two sheets | Next-90-day purchase inactivity, constructed from future transactions |
| Experimentation | [Hillstrom / MineThatData](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) | 64,000 customers randomized to men's email, women's email, or no email | Observed 2-week conversion and spending |

**Important:** These datasets have different customers and cannot be merged at the customer level. The Hillstrom experiment does *not* validate the retention model. We will not claim that observational Retail II purchases establish the causal effect of discounts or email offers.

## Design and evaluation safeguards

- Raw source files remain unchanged and are excluded from Git (`data/raw/`).
- Audit cancellations, returns, missing customer identifiers, dates, sheet overlap and possible duplicates *before* designing customer features. Do not silently drop records.
- Define eligible customers and features using information available **on or before** each prediction date. Construct labels from the subsequent 90 days only.
- Use time-based evaluation with a gap that prevents training labels from extending into later evaluation periods. Also report a simple cohort-size baseline and logistic regression before comparing a custom PyTorch network.
- Evaluate PR-AUC, ROC-AUC, recall, precision and F1; evaluate threshold selection on validation data only. Compare revenue risk estimates against future holdout behavior and document assumptions.
- Analyze the original Hillstrom **three-arm** randomization against the no-email control. Report absolute lift with confidence intervals and adjusted p-values; distinguish statistical significance from practical significance.
- Deployment target: FastAPI, Docker and local Kubernetes. These follow validated research results, not the other way around.

## Repository map

```text
src/retention_lab/     Input schema, audit, and analysis code
scripts/               Data download and reproducible audit entry points
data/raw/              Input files, never committed
reports/               Audit and experiment outputs
sql/                   SQL analytics and feature transformations (next milestone)
notebooks/             EDA and visualizations (next milestone)
models/                Trained artifacts, never committed
k8s/                   Kubernetes manifests (later milestone)
tests/                 Automated dataset-validation tests
```

## Getting started

Use Python 3.11 or 3.12.

```bash
# Create a virtual environment separate from your system Python.
python -m venv .venv

# On Linux/macOS/WSL, activate it:
source .venv/bin/activate
# On Windows PowerShell, instead run: .venv\Scripts\Activate.ps1

# Install dependencies for data ingestion and audits.
python -m pip install -r requirements.txt

# Download official source files into data/raw (requires internet).
python scripts/download_datasets.py

# Check schema, quality, and treatment-arm balance; save audit JSON in reports/.
python scripts/audit_datasets.py

# Run dataset-contract tests.
python -m pytest -q
```

If the Hillstrom host is inaccessible, visit the [author's original dataset post](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html), download the CSV, and place it at `data/raw/hillstrom.csv` before running the audit. If the UCI download is blocked, download `online_retail_II.xlsx` from its dataset page to `data/raw/`.

## Work plan (deadline: Fri Nov 20, 2026)

- **Wed:** data acquisition, source audits, schema tests, data-quality rules.
- **Thu:** SQL features, retrospective labels, leakage-safe train/validation/test splits, logistic baseline, custom PyTorch model.
- **Fri:** model evaluation and interpretability, Hillstrom causal analysis, API/container/Kubernetes demonstration, reproducibility check, honest README results and updated resume bullets.

## Attribution

- Daqing Chen (2012), *Online Retail II*, UCI Machine Learning Repository, DOI: [10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D), CC BY 4.0.
- Kevin Hillstrom (2008), *MineThatData E-Mail Analytics and Data Mining Challenge*, randomized e-mail marketing challenge. Check original distribution terms before redistributing raw files. We do not commit either raw dataset.

No evaluation metrics or revenue-impact claims will be presented until they are actually measured.
