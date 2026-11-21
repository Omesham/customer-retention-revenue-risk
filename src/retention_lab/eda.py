import matplotlib.pyplot as plt
import pandas as pd

def plot_correlation_heatmap(df):
    # Calculate correlations between our features and churn.
    columns = ["recency_days", "frequency", "monetary", "churn"]
    correlations = df[columns].corr()

    # Create the heatmap.
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(correlations, vmin=-1, vmax=1)

    # Label the rows and columns.
    ax.set_xticks(range(len(columns)), labels=columns, rotation=45, ha="right")
    ax.set_yticks(range(len(columns)), labels=columns)

    # Display correlation values inside each cell.
    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, f"{correlations.iloc[i, j]:.2f}", ha="center", va="center")

    # Add the correlation scale and display the chart.
    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    plt.show()

def plot_recency_by_churn(df):
    # Separate customers by churn outcome.
    retained = df.loc[df["churn"] == 0, "recency_days"]
    churned = df.loc[df["churn"] == 1, "recency_days"]

    # Compare recency distributions between the two groups.
    plt.boxplot([retained, churned], tick_labels=["Retained", "Churned"])
    plt.ylabel("Days since last purchase")
    plt.title("Customer Recency by Churn Status")
    plt.show()

def plot_frequency_by_churn(df):
    # Separate purchase frequencies by churn status.
    retained = df.loc[df["churn"] == 0, "frequency"]
    churned = df.loc[df["churn"] == 1, "frequency"]

    # Compare the distributions using a box plot.
    plt.boxplot([retained, churned], tick_labels=["Retained", "Churned"])
    plt.ylabel("Number of purchases")
    plt.title("Purchase Frequency by Churn Status")
    # Zoom in on frequencies between 0 and 30 without removing data.
    plt.ylim(0, 30)
    plt.show()

def plot_monetary_by_churn(df):
    # Separate customer spending by churn status.
    retained = df.loc[df["churn"] == 0, "monetary"]
    churned = df.loc[df["churn"] == 1, "monetary"]

    # Compare spending distributions without displaying extreme outliers.
    plt.boxplot([retained, churned], tick_labels=["Retained", "Churned"], showfliers=False)
    plt.ylabel("Total customer spending")
    plt.title("Customer Spending by Churn Status")
    plt.show()

def plot_churn_rate_by_frequency(df):
    # Group customers by number of purchases.
    groups = pd.cut(df["frequency"], bins=[0, 2, 5, 10, float("inf")], labels=["1–2", "3–5", "6–10", "11+"])

    # Calculate the percentage of customers who churned in each group.
    churn_rates = df.groupby(groups, observed=True)["churn"].mean() * 100

    # Compare churn rates across purchase-frequency groups.
    churn_rates.plot(kind="bar")
    plt.xlabel("Number of purchases")
    plt.ylabel("Churn rate (%)")
    plt.title("Churn Rate by Purchase Frequency")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()