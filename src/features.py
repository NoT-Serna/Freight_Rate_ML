"""
Feature engineering for freight rate prediction.

Design principle: this module must run identically over
train_test.csv, validation.csv, and december_chart_inputs.csv.
No fitting decisions should be made on validation data — any
statistic used to impute or scale (e.g. medians) is computed on
TRAIN ONLY and passed in / reused at predict time.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 1. Cleaning (impute, never drop — validation/december need every row kept)
# --------------------------------------------------------------------------

def fit_clean_stats(train_df: pd.DataFrame) -> dict:
    """Compute imputation statistics from TRAIN ONLY. Reuse on val/december."""
    return {
        "weight_median": train_df.loc[train_df["weight"] >= 0, "weight"].median(),
        "market_index_median": train_df["market_index"].median(),
    }


def clean(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()

    # Negative weight is a data-entry sign error, not a valid observation —
    # take the absolute value rather than dropping (we can't drop from
    # validation/december, and abs() preserves the shipment's magnitude).
    df["weight"] = df["weight"].abs()

    # Fill remaining missing weight / market_index with the TRAIN median.
    df["weight"] = df["weight"].fillna(stats["weight_median"])
    df["market_index"] = df["market_index"].fillna(stats["market_index_median"])

    return df


# --------------------------------------------------------------------------
# 2. Feature engineering
# --------------------------------------------------------------------------

# Anchor date for the linear trend feature. Use the TRAIN min date so
# "days_since_start" is consistent and comparable across train/val/december.
ANCHOR_DATE = pd.Timestamp("2025-01-01")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # --- Linear trend: lets the model extrapolate a drift beyond Oct ---
    df["days_since_start"] = (df["date"] - ANCHOR_DATE).dt.days

    # --- Cyclical seasonality: Dec sits smoothly next to Jan, not an
    #     unseen bucket the way a one-hot "month" column would be ---
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    dow = df["date"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # --- Core numeric signal ---
    # distance is kept as-is (your strongest linear driver, ~0.91 corr)
    # weight / market_index / quote_signal kept as-is after cleaning

    # --- Equipment: low cardinality, one-hot is safe (seen in both sets) ---
    df = pd.get_dummies(df, columns=["equipment"], prefix="equip")

    # --- Deliberately NOT used as model inputs ---
    # pickup / delivery / pickup_lat / pickup_lon / delivery_lat / delivery_lon:
    # validation has cities unseen in training, so any city-identity feature
    # (one-hot, target-encoded lane average, or raw lat/lon which is really
    # just a proxy for city identity on this synthetic map) would be undefined
    # for those rows. `distance` already captures the geographic driver in a
    # way that generalizes to unseen city pairs.

    return df


FEATURE_COLUMNS = [
    "distance", "weight", "market_index", "quote_signal",
    "days_since_start", "doy_sin", "doy_cos", "dow_sin", "dow_cos",
    "equip_Dry Van", "equip_Flatbed", "equip_Reefer",
]


def prepare(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """One call to go from raw CSV columns to model-ready feature matrix."""
    df = clean(df, stats)
    df = engineer_features(df)
    # Ensure all expected dummy columns exist even if a category is
    # absent from this particular slice (e.g. december inputs are all
    # Dry Van, so equip_Flatbed/equip_Reefer must still be added as 0).
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURE_COLUMNS]