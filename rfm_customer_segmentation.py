import logging
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    num_customers: int = 350
    min_transactions: int = 3
    max_transactions: int = 8
    start_date: str = "2024-01-01"
    end_date: str = "2025-12-31"
    analysis_date: str = "2026-01-01"
    output_dir: str = "output"
    random_state: int = 42
    min_k: int = 4
    max_k: int = 8


SEGMENT_ORDER = [
    "Champions",
    "Loyal Customers",
    "At-Risk",
    "Lost Customers",
]


def ensure_output_dir(path: str) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def validate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"CustomerID", "OrderDate", "TransactionAmount"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if df.empty:
        raise ValueError("The transactions dataframe is empty.")

    cleaned_df = df.copy()
    cleaned_df["OrderDate"] = pd.to_datetime(cleaned_df["OrderDate"], errors="coerce")
    cleaned_df["TransactionAmount"] = pd.to_numeric(
        cleaned_df["TransactionAmount"], errors="coerce"
    )

    null_count = cleaned_df[list(required_columns)].isna().sum().sum()
    if null_count > 0:
        LOGGER.warning("Dropping %s rows containing empty or invalid values.", null_count)
        cleaned_df = cleaned_df.dropna(subset=list(required_columns))

    cleaned_df = cleaned_df[cleaned_df["TransactionAmount"] > 0]
    if cleaned_df.empty:
        raise ValueError("No valid transactions remain after cleaning.")

    return cleaned_df


def save_dataframe(df: pd.DataFrame, output_path: Path, description: str) -> Path:
    df.to_csv(output_path, index=False)
    LOGGER.info("Saved %s to %s", description, output_path)
    return output_path


def simulate_transactions(config: Config) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_state)
    customer_ids = [f"CUST{idx:04d}" for idx in range(1, config.num_customers + 1)]

    start_ts = pd.Timestamp(config.start_date)
    end_ts = pd.Timestamp(config.end_date)
    num_days = (end_ts - start_ts).days

    transactions = []
    for customer_id in customer_ids:
        transaction_count = rng.integers(
            config.min_transactions, config.max_transactions + 1
        )

        # Create different customer value bands to make clustering meaningful.
        value_band = rng.choice(
            ["high", "mid", "low"], p=[0.2, 0.5, 0.3]
        )
        if value_band == "high":
            amount_mean, amount_std = 250, 55
        elif value_band == "mid":
            amount_mean, amount_std = 120, 30
        else:
            amount_mean, amount_std = 55, 18

        for _ in range(transaction_count):
            order_offset = int(rng.integers(0, num_days + 1))
            order_date = start_ts + pd.Timedelta(days=order_offset)
            amount = max(5, round(rng.normal(amount_mean, amount_std), 2))
            transactions.append(
                {
                    "CustomerID": customer_id,
                    "OrderDate": order_date,
                    "TransactionAmount": amount,
                }
            )

    transactions_df = pd.DataFrame(transactions)
    if len(transactions_df) < 1000:
        raise ValueError(
            "Synthetic data generation produced fewer than 1,000 transactions."
        )

    LOGGER.info("Generated %s synthetic transactions.", len(transactions_df))
    return validate_transactions(transactions_df)


def build_rfm_features(
    transactions_df: pd.DataFrame, analysis_date: str
) -> pd.DataFrame:
    analysis_ts = pd.Timestamp(analysis_date)
    if transactions_df.empty:
        raise ValueError("Cannot compute RFM features from an empty dataframe.")

    rfm_df = (
        transactions_df.groupby("CustomerID")
        .agg(
            LastOrderDate=("OrderDate", "max"),
            Frequency=("OrderDate", "count"),
            Monetary=("TransactionAmount", "sum"),
        )
        .reset_index()
    )
    rfm_df["Recency"] = (analysis_ts - rfm_df["LastOrderDate"]).dt.days
    rfm_df["Monetary"] = rfm_df["Monetary"].round(2)

    if rfm_df[["Recency", "Frequency", "Monetary"]].isna().any().any():
        raise ValueError("RFM feature calculation produced null values.")

    return rfm_df[["CustomerID", "Recency", "Frequency", "Monetary"]]


def score_rfm_features(rfm_df: pd.DataFrame) -> pd.DataFrame:
    scored_df = rfm_df.copy()

    # Recency is inverse-scored because lower recency is better.
    scored_df["R_Score"] = pd.qcut(
        scored_df["Recency"].rank(method="first", ascending=True),
        q=4,
        labels=[4, 3, 2, 1],
    ).astype(int)
    scored_df["F_Score"] = pd.qcut(
        scored_df["Frequency"].rank(method="first", ascending=True),
        q=4,
        labels=[1, 2, 3, 4],
    ).astype(int)
    scored_df["M_Score"] = pd.qcut(
        scored_df["Monetary"].rank(method="first", ascending=True),
        q=4,
        labels=[1, 2, 3, 4],
    ).astype(int)
    scored_df["RFM_Score"] = (
        scored_df["R_Score"] + scored_df["F_Score"] + scored_df["M_Score"]
    )
    return scored_df


def validate_config(config: Config) -> None:
    if config.num_customers <= 0:
        raise ValueError("num_customers must be greater than zero.")
    if config.min_transactions <= 0:
        raise ValueError("min_transactions must be greater than zero.")
    if config.max_transactions < config.min_transactions:
        raise ValueError("max_transactions must be greater than or equal to min_transactions.")
    if pd.Timestamp(config.analysis_date) <= pd.Timestamp(config.end_date):
        raise ValueError("analysis_date must be later than end_date.")


def scale_features(
    rfm_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, StandardScaler]:
    feature_columns = ["Recency", "Frequency", "Monetary"]
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(rfm_df[feature_columns])
    scaled_df = pd.DataFrame(scaled_array, columns=feature_columns, index=rfm_df.index)
    return scaled_df, scaler


def find_optimal_k(
    scaled_df: pd.DataFrame, min_k: int, max_k: int, random_state: int
) -> Tuple[int, Dict[int, float]]:
    if min_k < 2 or max_k <= min_k:
        raise ValueError("Invalid k range supplied for elbow calculation.")

    inertias: Dict[int, float] = {}
    k_values = list(range(min_k, min(max_k, len(scaled_df) - 1) + 1))
    if len(k_values) < 2:
        raise ValueError("Not enough samples to evaluate multiple cluster sizes.")

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            random_state=random_state,
        )
        model.fit(scaled_df)
        inertias[k] = model.inertia_

    points = np.array([[k, inertias[k]] for k in k_values], dtype=float)
    start_point = points[0]
    end_point = points[-1]
    line_vector = end_point - start_point
    line_vector_norm = np.linalg.norm(line_vector)
    if line_vector_norm == 0:
        optimal_k = k_values[0]
        return optimal_k, inertias

    distances = []
    for point in points:
        point_vector = point - start_point
        distance = abs(
            (line_vector[0] * point_vector[1]) - (line_vector[1] * point_vector[0])
        ) / line_vector_norm
        distances.append(distance)

    optimal_index = int(np.argmax(distances))
    optimal_k = k_values[optimal_index]
    LOGGER.info("Selected k=%s using the elbow method.", optimal_k)
    return optimal_k, inertias


def train_kmeans(
    scaled_df: pd.DataFrame, n_clusters: int, random_state: int
) -> KMeans:
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=20,
        random_state=random_state,
    )
    model.fit(scaled_df)
    return model


def assign_segment_labels(rfm_df: pd.DataFrame) -> pd.DataFrame:
    cluster_summary = (
        rfm_df.groupby("Cluster")
        .agg(
            Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"),
            Monetary=("Monetary", "mean"),
            R_Score=("R_Score", "mean"),
            F_Score=("F_Score", "mean"),
            M_Score=("M_Score", "mean"),
            RFM_Score=("RFM_Score", "mean"),
        )
        .reset_index()
    )

    cluster_summary["QualityScore"] = (
        (0.4 * cluster_summary["R_Score"])
        + (0.3 * cluster_summary["F_Score"])
        + (0.3 * cluster_summary["M_Score"])
    ).round(2)
    cluster_summary["SegmentLabel"] = "Loyal Customers"

    champion_idx = cluster_summary["QualityScore"].idxmax()
    lost_idx = cluster_summary["QualityScore"].idxmin()

    cluster_summary.loc[champion_idx, "SegmentLabel"] = "Champions"
    cluster_summary.loc[lost_idx, "SegmentLabel"] = "Lost Customers"

    at_risk_mask = (
        cluster_summary.index.isin([champion_idx, lost_idx]) == False
    ) & (
        (cluster_summary["R_Score"] <= 2.75)
        & (
            (cluster_summary["F_Score"] >= 2.5)
            | (cluster_summary["M_Score"] >= 2.5)
            | (cluster_summary["RFM_Score"] >= 7.0)
        )
    )
    cluster_summary.loc[at_risk_mask, "SegmentLabel"] = "At-Risk"

    cluster_summary = cluster_summary.sort_values(
        by=["QualityScore", "RFM_Score", "Monetary", "Frequency"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    segment_mapping: Dict[int, str] = dict(
        zip(cluster_summary["Cluster"], cluster_summary["SegmentLabel"])
    )

    labeled_df = rfm_df.copy()
    labeled_df["Segment"] = labeled_df["Cluster"].map(segment_mapping)

    score_map = dict(zip(cluster_summary["Cluster"], cluster_summary["QualityScore"]))
    labeled_df["ClusterQualityScore"] = labeled_df["Cluster"].map(score_map)
    return labeled_df


def print_marketing_strategy_report(segmented_df: pd.DataFrame) -> None:
    strategies = {
        "Champions": "Reward with VIP perks, early-access campaigns, and referral incentives.",
        "Loyal Customers": "Upsell with product bundles, loyalty points, and personalized recommendations.",
        "At-Risk": "Send a 20% discount coupon and a limited-time reactivation email.",
        "Lost Customers": "Run a win-back campaign with a strong introductory offer and feedback survey.",
    }

    print("\n" + "=" * 72)
    print("MARKETING STRATEGY REPORT")
    print("=" * 72)
    for segment, strategy in strategies.items():
        segment_size = (segmented_df["Segment"] == segment).sum()
        print(f"For {segment} customers ({segment_size} total): {strategy}")
    print("=" * 72 + "\n")


def build_segment_summary(segmented_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        segmented_df.groupby("Segment")
        .agg(
            CustomerCount=("CustomerID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
            AvgRFMScore=("RFM_Score", "mean"),
        )
        .reset_index()
    )

    summary_df["Segment"] = pd.Categorical(
        summary_df["Segment"], categories=SEGMENT_ORDER, ordered=True
    )
    summary_df = summary_df.sort_values("Segment").reset_index(drop=True)
    numeric_columns = [
        "AvgRecency",
        "AvgFrequency",
        "AvgMonetary",
        "AvgRFMScore",
    ]
    summary_df[numeric_columns] = summary_df[numeric_columns].round(2)
    return summary_df


def plot_elbow_curve(inertias: Dict[int, float], output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(list(inertias.keys()), list(inertias.values()), marker="o", linewidth=2)
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "elbow_curve.png", dpi=300)
    plt.close()


def plot_customer_clusters(segmented_df: pd.DataFrame, output_dir: Path) -> None:
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(
        segmented_df[["RecencyScaled", "FrequencyScaled", "MonetaryScaled"]]
    )

    plot_df = segmented_df.copy()
    plot_df["PC1"] = components[:, 0]
    plot_df["PC2"] = components[:, 1]
    plot_df["Segment"] = pd.Categorical(
        plot_df["Segment"], categories=SEGMENT_ORDER, ordered=True
    )

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="Segment",
        size="Monetary",
        hue_order=SEGMENT_ORDER,
        palette={
            "Champions": "#2E8B57",
            "Loyal Customers": "#1F77B4",
            "At-Risk": "#FF8C00",
            "Lost Customers": "#C0392B",
        },
        alpha=0.85,
        sizes=(40, 220),
    )
    plt.title("Customer Segments Visualized with PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "customer_segments.png", dpi=300)
    plt.close()


def export_for_bi(segmented_df: pd.DataFrame, output_dir: Path) -> Path:
    export_columns = [
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary",
        "R_Score",
        "F_Score",
        "M_Score",
        "RFM_Score",
        "Cluster",
        "Segment",
        "ClusterQualityScore",
    ]
    export_df = segmented_df[export_columns].sort_values(
        by=["Segment", "Monetary"], ascending=[True, False]
    )
    export_path = output_dir / "customer_segments_bi_ready.csv"
    return save_dataframe(export_df, export_path, "BI-ready customer segments")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end e-commerce customer segmentation pipeline using RFM analysis."
    )
    parser.add_argument("--num-customers", type=int, default=350)
    parser.add_argument("--min-transactions", type=int, default=3)
    parser.add_argument("--max-transactions", type=int, default=8)
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--end-date", type=str, default="2025-12-31")
    parser.add_argument("--analysis-date", type=str, default="2026-01-01")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-k", type=int, default=4)
    parser.add_argument("--max-k", type=int, default=8)
    args = parser.parse_args()

    return Config(
        num_customers=args.num_customers,
        min_transactions=args.min_transactions,
        max_transactions=args.max_transactions,
        start_date=args.start_date,
        end_date=args.end_date,
        analysis_date=args.analysis_date,
        output_dir=args.output_dir,
        random_state=args.random_state,
        min_k=args.min_k,
        max_k=args.max_k,
    )


def run_pipeline(config: Config) -> pd.DataFrame:
    validate_config(config)
    output_dir = ensure_output_dir(config.output_dir)

    transactions_df = simulate_transactions(config)
    save_dataframe(
        transactions_df.sort_values(["CustomerID", "OrderDate"]).reset_index(drop=True),
        output_dir / "synthetic_transactions.csv",
        "synthetic transactions",
    )

    rfm_df = build_rfm_features(transactions_df, config.analysis_date)
    rfm_df = score_rfm_features(rfm_df)
    scaled_df, _ = scale_features(rfm_df)

    optimal_k, inertias = find_optimal_k(
        scaled_df, config.min_k, config.max_k, config.random_state
    )
    model = train_kmeans(scaled_df, optimal_k, config.random_state)

    segmented_df = rfm_df.copy()
    segmented_df["Cluster"] = model.labels_
    segmented_df["RecencyScaled"] = scaled_df["Recency"]
    segmented_df["FrequencyScaled"] = scaled_df["Frequency"]
    segmented_df["MonetaryScaled"] = scaled_df["Monetary"]

    segmented_df = assign_segment_labels(segmented_df)

    plot_elbow_curve(inertias, output_dir)
    plot_customer_clusters(segmented_df, output_dir)
    export_for_bi(segmented_df, output_dir)
    save_dataframe(
        build_segment_summary(segmented_df),
        output_dir / "segment_summary.csv",
        "segment summary",
    )
    print_marketing_strategy_report(segmented_df)

    return segmented_df


def main() -> None:
    try:
        config = parse_args()
        final_df = run_pipeline(config)
        LOGGER.info("Pipeline completed successfully.")
        LOGGER.info("\n%s", final_df.head(10).to_string(index=False))
    except Exception as exc:
        LOGGER.exception("Pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
