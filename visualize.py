import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/segmented_customers.csv")
OUTPUT_PATH = Path("models/customer_segments.png")

df = pd.read_csv(DATA_PATH)

plt.figure(figsize=(10, 6))

for segment, group in df.groupby("Segment"):
    plt.scatter(
        group["Annual Income (k$)"],
        group["Spending Score (1-100)"],
        label=segment,
        alpha=0.75
    )

plt.title("Customer Segmentation by Income and Spending")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(fontsize=8)
plt.grid(alpha=0.2)
plt.tight_layout()

OUTPUT_PATH.parent.mkdir(exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=150)
plt.close()

print("Visualization saved to:", OUTPUT_PATH)
