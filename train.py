import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

DATA_PATH = Path("data/mall_customers.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
silhouette_scores = []

for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    inertias.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

results = pd.DataFrame({
    "k": range(2, 11),
    "inertia": inertias,
    "silhouette_score": silhouette_scores
})

results.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

best_k = int(results.loc[results["silhouette_score"].idxmax(), "k"])

final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = final_model.fit_predict(X_scaled)

df["Cluster"] = clusters
df.to_csv(DATA_PATH.parent / "segmented_customers.csv", index=False)

joblib.dump(final_model, MODEL_DIR / "kmeans_model.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

plt.figure(figsize=(9, 5))
plt.plot(results["k"], results["inertia"], marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Customer Segmentation")
plt.xticks(range(2, 11))
plt.tight_layout()
plt.savefig(MODEL_DIR / "elbow_method.png", dpi=150)
plt.close()

print("Dataset shape:", df.shape)
print("\nModel comparison:")
print(results.round(4).to_string(index=False))
print("\nSelected number of clusters:", best_k)
print("Silhouette score:", round(silhouette_score(X_scaled, clusters), 4))
print("\nCluster sizes:")
print(df["Cluster"].value_counts().sort_index())
print("\nTraining completed successfully!")
