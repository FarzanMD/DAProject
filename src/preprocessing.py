"""Trip-level cleaning and hourly demand aggregation for NYC TLC data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)
START = pd.Timestamp("2024-01-01")
END_EXCLUSIVE = pd.Timestamp("2024-02-01")
TRAIN_END_EXCLUSIVE = pd.Timestamp("2024-01-22")
PICKUP = "tpep_pickup_datetime"
DROPOFF = "tpep_dropoff_datetime"
ZONE = "PULocationID"
RAW_COLUMNS = (
    PICKUP,
    DROPOFF,
    ZONE,
    "fare_amount",
    "total_amount",
    "trip_distance",
    "passenger_count",
)


def clean_and_aggregate(
    trips: pd.DataFrame,
    start: pd.Timestamp = START,
    end_exclusive: pd.Timestamp = END_EXCLUSIVE,
    train_end_exclusive: pd.Timestamp = TRAIN_END_EXCLUSIVE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean trips, then return a continuous hourly panel and attrition report.

    The zone universe is derived from cleaned training trips only. This avoids
    introducing validation-only zones into the feature space.
    """
    required = {
        PICKUP,
        DROPOFF,
        ZONE,
        "fare_amount",
        "total_amount",
        "trip_distance",
        "passenger_count",
    }
    missing_columns = required.difference(trips.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    data = trips.copy()
    initial_count = len(data)
    attrition = [{"stage": "raw", "removed": 0, "remaining": initial_count}]

    data[PICKUP] = pd.to_datetime(data[PICKUP], errors="coerce")
    data[DROPOFF] = pd.to_datetime(data[DROPOFF], errors="coerce")

    def retain(mask: pd.Series, stage: str) -> None:
        nonlocal data
        previous_count = len(data)
        data = data.loc[mask].copy()
        removed = previous_count - len(data)
        attrition.append(
            {"stage": stage, "removed": removed, "remaining": len(data)}
        )
        LOGGER.info(
            "%s: removed %s rows; %s remain",
            stage,
            f"{removed:,}",
            f"{len(data):,}",
        )

    retain(data[PICKUP].ge(start) & data[PICKUP].lt(end_exclusive), "pickup_date")
    data["trip_duration_min"] = (
        data[DROPOFF] - data[PICKUP]
    ).dt.total_seconds() / 60
    retain(
        data["fare_amount"].gt(0) & data["total_amount"].gt(0),
        "positive_amounts",
    )
    retain(
        data["trip_duration_min"].ge(1) & data["trip_duration_min"].le(180),
        "valid_duration_1_to_180_min",
    )
    retain(
        data["trip_distance"].gt(0) & data["trip_distance"].le(100),
        "valid_distance_0_to_100_miles",
    )
    retain(
        data[ZONE].notna()
        & data[ZONE].between(1, 263)
        & ~data[ZONE].isin([264, 265]),
        "valid_pickup_zone",
    )

    data[ZONE] = data[ZONE].astype("int16")
    data["passenger_count"] = data["passenger_count"].fillna(1)
    data["pickup_hour"] = data[PICKUP].dt.floor("h")
    data["train_period"] = data[PICKUP].lt(train_end_exclusive)
    active_zones = sorted(
        data.loc[data["train_period"], ZONE].unique().tolist()
    )
    if not active_zones:
        raise ValueError("No valid training trips remain after cleaning.")

    trip_features = [
        column
        for column in (
            "fare_amount",
            "total_amount",
            "trip_distance",
            "trip_duration_min",
            "passenger_count",
        )
        if column in data.columns
    ]
    aggregated = (
        data.loc[data[ZONE].isin(active_zones)]
        .groupby([ZONE, "pickup_hour"], observed=True)
        .agg(
            trip_count=(PICKUP, "size"),
            **{f"mean_{column}": (column, "mean") for column in trip_features},
        )
    )
    hours = pd.date_range(start, end_exclusive, freq="h", inclusive="left")
    full_index = pd.MultiIndex.from_product(
        [active_zones, hours], names=[ZONE, "pickup_hour"]
    )
    hourly = aggregated.reindex(full_index).reset_index()
    hourly["trip_count"] = hourly["trip_count"].fillna(0).astype("int32")
    mean_columns = [column for column in hourly if column.startswith("mean_")]
    hourly[mean_columns] = hourly[mean_columns].fillna(0)
    hourly["train_period"] = hourly["pickup_hour"].lt(train_end_exclusive)

    return hourly, pd.DataFrame(attrition)


def build_processed_dataset(
    input_path: str | Path,
    output_path: str | Path = "data/processed/hourly_demand.parquet",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read raw parquet, clean/aggregate trips, and save the hourly panel."""
    trips = pd.read_parquet(input_path, columns=list(RAW_COLUMNS))
    hourly, attrition = clean_and_aggregate(trips)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(output, index=False)
    LOGGER.info("Saved hourly demand to %s", output)
    return hourly, attrition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/yellow_tripdata_2024-01.parquet",
        help="Path to the raw January 2024 TLC parquet file.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/hourly_demand.parquet",
        help="Path for the cleaned, aggregated hourly parquet file.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _, attrition = build_processed_dataset(args.input, args.output)
    print(attrition.to_string(index=False))


if __name__ == "__main__":
    main()
