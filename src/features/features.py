import pandas as pd

def create_lag_features(df: pd.DataFrame):
    df["vpd_lag2"] = df.groupby(["y", "x"])["vpd"].shift(2)
    df["temp_lag2"] = df.groupby(["y", "x"])["temp"].shift(2)
    df["precip_lag2"] = df.groupby(["y", "x"])["precip"].shift(2)
    df["vpd_ghm_interaction"] = df["vpd_lag2"] * df["ghm"]
    df["month"] = df["valid_time"].dt.month
    return df

def prepare_features(df: pd.DataFrame):
    df = df.sort_values(["y", "x", "valid_time"])        
    df = create_lag_features(df)
    
    df = df.dropna(subset=[
        "temp_lag2",
        "vpd_lag2",
        "precip_lag2"
    ])
    
    return df