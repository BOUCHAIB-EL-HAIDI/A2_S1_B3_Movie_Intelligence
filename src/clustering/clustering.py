
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "processed" / "movies_features.json"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
VISUALIZATION_DIR = BASE_DIR / "data" / "visualizations"


def load_data():
    df = pd.read_json(INPUT_FILE)
    return df


def prepare_financial_data(df):
    features = ["budget", "revenue"]

    financial_data = df[features].copy()
    financial_data = financial_data.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    financial_data = imputer.fit_transform(financial_data)

    scaler = StandardScaler()
    financial_matrix = scaler.fit_transform(financial_data)

    return financial_matrix


def create_clusters(financial_matrix):
    model = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(financial_matrix)
    score = silhouette_score(financial_matrix, labels)

    return labels, score


def analyze_clusters(df, labels):
    results = df[
        ["movie_id", "title", "budget", "revenue"]
    ].copy()

    results["cluster"] = labels

    print("\n" + "=" * 50)
    print("FINANCIAL CLUSTERING")
    print("=" * 50)

    print("\nSilhouette score:", round(
        silhouette_score(
            prepare_financial_data(df), labels
        ), 3
    ))

    print("\nCluster sizes:")
    print(results["cluster"].value_counts().sort_index())

    print("\nCluster medians:")
    print(
        results.groupby("cluster")[["budget", "revenue"]]
        .median()
        .round(2)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        OUTPUT_DIR / "financial_clusters.csv",
        index=False
    )

    return results


def plot_clusters(results):
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))

    for cluster_id in sorted(results["cluster"].unique()):
        cluster_data = results[
            results["cluster"] == cluster_id
        ]

        plt.scatter(
            cluster_data["budget"],
            cluster_data["revenue"],
            label=f"Cluster {cluster_id}",
            alpha=0.6
        )

    plt.xlabel("Budget")
    plt.ylabel("Revenue")
    plt.title("Financial Clustering — Budget vs Revenue")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = VISUALIZATION_DIR / "financial_clusters.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"\nVisualization saved to: {output_path}")


def main():
    df = load_data()

    financial_matrix = prepare_financial_data(df)
    labels, score = create_clusters(financial_matrix)

    results = analyze_clusters(df, labels)
    plot_clusters(results)

    print(f"\nFinal silhouette score: {score:.3f}")
    print("Number of clusters: 2")


if __name__ == "__main__":
    main()
