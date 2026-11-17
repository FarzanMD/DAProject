"""Run the complete leakage-safe Phase 2 feature pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

if __package__:
    from .anomaly_detection import fit_demand_profiles, flag_demand_anomalies
    from .clustering import fit_zone_clusters
    from .features import add_cluster_labels, add_temporal_features, prepare_modeling_frame
    from .preprocessing import RAW_COLUMNS, clean_and_aggregate
else:
    from anomaly_detection import fit_demand_profiles, flag_demand_anomalies
    from clustering import fit_zone_clusters
    from features import add_cluster_labels, add_temporal_features, prepare_modeling_frame
    from preprocessing import RAW_COLUMNS, clean_and_aggregate

LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT = Path("data/yellow_tripdata_2024-01.parquet")
DEFAULT_OUTPUT = Path("data/processed/hourly_demand_features.parquet")


def run_phase2(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build features and train-only cluster/anomaly labels; save final parquet."""
    trips = pd.read_parquet(input_path, columns=list(RAW_COLUMNS))
    hourly, attrition = clean_and_aggregate(trips)

    labels, cluster_metrics, _ = fit_zone_clusters(hourly)
    features = add_cluster_labels(add_temporal_features(hourly), labels)

    train_end = pd.Timestamp("2024-01-22")
    anomaly_profiles = fit_demand_profiles(hourly, train_end=train_end)
    anomaly_results = flag_demand_anomalies(hourly, anomaly_profiles)
    features = features.merge(
        anomaly_results[["PULocationID", "pickup_hour", "anomaly_z_score", "is_anomaly"]],
        on=["PULocationID", "pickup_hour"],
        how="left",
        validate="one_to_one",
    )
    if features["is_anomaly"].isna().any():
        raise ValueError("Anomaly scoring did not cover every hourly row.")
    features["is_anomaly"] = features["is_anomaly"].astype(bool)
    features = prepare_modeling_frame(features)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output, index=False)
    LOGGER.info("Saved %s feature rows to %s", f"{len(features):,}", output)
    return features, attrition, cluster_metrics, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    features, attrition, cluster_metrics, _ = run_phase2(args.input, args.output)
    print("\nData attrition:")
    print(attrition.to_string(index=False))
    print("\nCluster selection metrics:")
    print(cluster_metrics.to_string(index=False))
    print(f"\nFinal shape: {features.shape}")


if __name__ == "__main__":
    main()
