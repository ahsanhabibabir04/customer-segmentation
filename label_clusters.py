import pandas as pd

DATA_PATH = "data/segmented_customers.csv"

df = pd.read_csv(DATA_PATH)

cluster_names = {
    0: "Average Income - Average Spending",
    1: "High Income - High Spending",
    2: "Low Income - High Spending",
    3: "High Income - Low Spending",
    4: "Low Income - Low Spending",
}

df["Segment"] = df["Cluster"].map(cluster_names)
df.to_csv(DATA_PATH, index=False)

print("Customer segments labeled successfully!")
print()
print(df.groupby("Segment").agg(
    Customers=("CustomerID", "count"),
    Average_Age=("Age", "mean"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending=("Spending Score (1-100)", "mean")
).round(2).to_string())
