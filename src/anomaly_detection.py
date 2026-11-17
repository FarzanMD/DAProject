"""Training-profile Z-score anomaly detection for hourly zone demand."""

from __future__ import annotations

import numpy as np
import pandas as pd

ZONE = "PULocationID"
TIME = "pickup_hour"
TARGET = "trip_count"


def fit_demand_profiles(
    hourly: pd.DataFrame,
    train_end: pd.Timestamp = pd.Timestamp("2024-01-22"),
) -> pd.DataFrame:
    """Fit weekday/hour demand mean and standard deviation on training rows."""
    train = hourly.loc[pd.to_datetime(hourly[TIME]).lt(train_end)].copy()
    if train.empty:
        raise ValueError("Training data is empty; cannot fit anomaly profiles.")
    timestamps = pd.to_datetime(train[TIME])
    train["day_of_week"] = timestamps.dt.dayofweek
    train["hour"] = timestamps.dt.hour
    return (
        train.groupby([ZONE, "day_of_week", "hour"])[TARGET]
        .agg(profile_mean="mean", profile_std="std", profile_count="count")
        .reset_index()
        .fillna({"profile_std": 0})
    )


def flag_demand_anomalies(
    hourly: pd.DataFrame,
    profiles: pd.DataFrame,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Score rows against training-only weekday/hour profiles."""
    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    result = hourly.copy()
    timestamps = pd.to_datetime(result[TIME])
    result["day_of_week"] = timestamps.dt.dayofweek
    result["hour"] = timestamps.dt.hour
    result = result.merge(
        profiles,
        on=[ZONE, "day_of_week", "hour"],
        how="left",
        validate="many_to_one",
    )
    if result["profile_mean"].isna().any():
        raise ValueError("Some hourly rows have no matching training anomaly profile.")
    deviation = result[TARGET] - result["profile_mean"]
    usable_std = result["profile_std"].replace(0, np.nan)
    result["anomaly_z_score"] = (deviation / usable_std).fillna(0)
    result["is_anomaly"] = result["anomaly_z_score"].abs().ge(threshold)
    return result
