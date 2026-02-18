import pandas as pd

def create_lag_features(df: pd.DataFrame):
    df["vpd_lag2"] = df.groupby(["y", "x"])["vpd"].shift(2)
    df["temp_lag2"] = df.groupby(["y", "x"])["temp"].shift(2)
    df["precip_lag2"] = df.groupby(["y", "x"])["precip"].shift(2)
    df["vpd_ghm_interaction"] = df["vpd_lag2"] * df["ghm"]
    df["month"] = df["valid_time"].dt.month
    return df