import pandas as pd


def time_split(df: pd.DataFrame, train_end: str = "2025-08-31", val_start: str = "2025-09-01"):
    """
    Train: Jan-Aug 2025
    Validation (holdout): Sep-Oct 2025
    """
    dates = pd.to_datetime(df["date"])
    train_df = df[dates <= train_end].reset_index(drop=True)
    val_df = df[dates >= val_start].reset_index(drop=True)
    return train_df, val_df
