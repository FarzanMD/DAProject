"""Leakage-safe temporal and lag features for hourly demand panels."""

from __future__ import annotations

import numpy as np
import pandas as pd

ZONE = "PULocationID"
TIME = "pickup_hour"
TARGET = "trip_count"


def add_temporal_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Add calendar encodings, target lags, and past-only rolling statistics."""
    required = {ZONE, TIME, TARGET}
    missing = required.difference(hourly.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = hourly.copy()
    result[TIME] = pd.to_datetime(result[TIME])
    result = result.sort_values([ZONE, TIME]).reset_index(drop=True)
    timestamps = result[TIME]
    result["hour"] = timestamps.dt.hour
    result["day_of_week"] = timestamps.dt.dayofweek
    result["is_weekend"] = timestamps.dt.dayofweek.isin([5, 6]).astype("int8")

    hour_angle = 2 * np.pi * result["hour"] / 24
    dow_angle = 2 * np.pi * result["day_of_week"] / 7
    result["hour_sin"] = np.sin(hour_angle)
    result["hour_cos"] = np.cos(hour_angle)
    result["dow_sin"] = np.sin(dow_angle)
    result["dow_cos"] = np.cos(dow_angle)

    grouped = result.groupby(ZONE, sort=False)[TARGET]
    for lag in (1, 2, 24, 168):
        result[f"lag_{lag}h"] = grouped.shift(lag)
    shifted = grouped.shift(1)
    result["rolling_mean_3h"] = shifted.groupby(result[ZONE], sort=False).transform(
        lambda values: values.rolling(3, min_periods=3).mean()
    )
    result["rolling_std_3h"] = shifted.groupby(result[ZONE], sort=False).transform(
        lambda values: values.rolling(3, min_periods=3).std()
    )
    return result


def add_cluster_labels(
    features: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Attach train-fitted zone cluster IDs to all rows."""
    if not {ZONE, "zone_cluster"}.issubset(labels.columns):
        raise ValueError("labels must contain PULocationID and zone_cluster")
    mapping = labels.drop_duplicates(ZONE).set_index(ZONE)["zone_cluster"]
    result = features.copy()
    result["zone_cluster"] = result[ZONE].map(mapping).astype("Int64")
    if result["zone_cluster"].isna().any():
        missing_zones = result.loc[result["zone_cluster"].isna(), ZONE].unique()
        raise ValueError(f"No cluster label for zones: {missing_zones.tolist()}")
    result["zone_cluster"] = result["zone_cluster"].astype("int16")
    return result


def prepare_modeling_frame(
    features: pd.DataFrame, warmup_hours: int = 168
) -> pd.DataFrame:
    """Remove startup rows that lack a full weekly lag and rolling history."""
    result = features.sort_values([ZONE, TIME]).copy()
    first_hour = result.groupby(ZONE)[TIME].transform("min")
    elapsed_hours = (result[TIME] - first_hour).dt.total_seconds() / 3600
    result = result.loc[elapsed_hours.ge(warmup_hours)].copy()

    lag_columns = [
        "lag_1h",
        "lag_2h",
        "lag_24h",
        "lag_168h",
        "rolling_mean_3h",
        "rolling_std_3h",
    ]
    if result[lag_columns].isna().any().any():
        raise ValueError("Lag features still contain missing values after warmup.")
    return result.reset_index(drop=True)
