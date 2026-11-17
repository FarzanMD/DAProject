"""Train-only spatial demand profiling and zone clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ZONE = "PULocationID"
TIME = "pickup_hour"
TRAIN_END = pd.Timestamp("2024-01-22")


def build_zone_profiles(
    hourly: pd.DataFrame, train_end: pd.Timestamp = TRAIN_END
) -> pd.DataFrame:
    """Build 24-hour demand shape and mean trip distance from training only."""
    train = hourly.loc[pd.to_datetime(hourly[TIME]).lt(train_end)].copy()
    if train.empty:
        raise ValueError("Training data is empty; cannot profile zones.")
    train["hour"] = pd.to_datetime(train[TIME]).dt.hour
    profile = train.pivot_table(
        index=ZONE,
        columns="hour",
        values="trip_count",
        aggfunc="mean",
        fill_value=0,
    ).reindex(columns=range(24), fill_value=0)
    profile.columns = [f"demand_hour_{hour:02d}" for hour in profile.columns]
    distance_columns = [
        column for column in train.columns if column in ("mean_trip_distance",)
    ]
    if distance_columns:
        distance = (
            train.assign(
                _trip_distance_sum=train["mean_trip_distance"] * train["trip_count"]
            )
            .groupby(ZONE)
            .agg(
                _trip_distance_sum=("_trip_distance_sum", "sum"),
                _trip_count=("trip_count", "sum"),
            )
            .eval("_trip_distance_sum / _trip_count")
            .rename("mean_trip_distance_profile")
        )
    else:
        distance = pd.Series(0.0, index=profile.index, name="mean_trip_distance_profile")
    return profile.join(distance).reset_index()


def select_cluster_count(
    profiles: pd.DataFrame, candidates: range = range(3, 6), random_state: int = 42
) -> tuple[pd.DataFrame, int]:
    """Score candidate K values with inertia and silhouette; return best K."""
    feature_columns = [
        column for column in profiles.columns if column != ZONE
    ]
    matrix = profiles[feature_columns].to_numpy(dtype=float)
    if len(matrix) < 3:
        raise ValueError("At least three zones are needed for clustering.")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    scores = []
    for k in candidates:
        if k >= len(profiles):
            continue
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(scaled)
        scores.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(scaled, labels),
            }
        )
    if not scores:
        raise ValueError("No valid cluster counts for the available zones.")
    metrics = pd.DataFrame(scores)
    best_k = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
    return metrics, best_k


def fit_zone_clusters(
    hourly: pd.DataFrame,
    n_clusters: int | None = None,
    train_end: pd.Timestamp = TRAIN_END,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit scaler and K-Means on train profiles, then label every known zone."""
    profiles = build_zone_profiles(hourly, train_end=train_end)
    metrics, best_k = select_cluster_count(profiles, random_state=random_state)
    selected_k = best_k if n_clusters is None else n_clusters
    if selected_k not in metrics["k"].tolist():
        raise ValueError(
            f"n_clusters must be one of {metrics['k'].tolist()}, got {selected_k}."
        )
    feature_columns = [column for column in profiles if column != ZONE]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(profiles[feature_columns].to_numpy(dtype=float))
    model = KMeans(n_clusters=selected_k, random_state=random_state, n_init=10)
    labels = profiles[[ZONE]].copy()
    labels["zone_cluster"] = model.fit_predict(scaled)
    labels["cluster_name"] = _describe_clusters(profiles, labels)
    return labels, metrics, profiles


def _describe_clusters(profiles: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    summary = profiles.merge(labels, on=ZONE)
    demand_columns = [column for column in profiles if column.startswith("demand_hour_")]
    summary["mean_demand"] = summary[demand_columns].mean(axis=1)
    means = summary.groupby("zone_cluster")["mean_demand"].mean()
    low_demand_cutoff = means.quantile(1 / 3)
    peak_hours = summary.groupby("zone_cluster")[demand_columns].mean().idxmax(axis=1)
    names = {}
    for cluster, peak_column in peak_hours.items():
        hour = int(peak_column.rsplit("_", 1)[1])
        if means.loc[cluster] <= low_demand_cutoff:
            activity = "lower-demand"
        else:
            activity = "higher-demand"
        if 6 <= hour <= 10:
            period = "morning-peak"
        elif 16 <= hour <= 20:
            period = "evening-peak"
        elif 0 <= hour <= 5:
            period = "overnight"
        else:
            period = "midday"
        names[cluster] = f"{activity} {period}"
    return labels["zone_cluster"].map(names)
