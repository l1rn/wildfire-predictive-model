import matplotlib.pyplot as plt
from pandas import DataFrame

def plot_month_map(
    df: DataFrame,
    year: int,
    month: int,
    title: str,
    save_path: str    
):
    subset = df[
        (df["valid_time"].dt.year == year) &
        (df["valid_time"].dt.month == month)
    ]
    print("Total X:", df["x"].nunique())
    print("Total Y:", df["y"].nunique())
    
    print("Unique X:", subset["x"].nunique())
    print("Unique Y:", subset["y"].nunique())
        
    risk_map = subset.pivot(
        index="y",
        columns="x",
        values="fire_probability"
    )
    
    risk_map = risk_map.sort_index().sort_index(axis=1)
    print(risk_map.isna().sum().sum())

    plt.figure(figsize=(12, 6))
    plt.imshow(risk_map, origin="lower", vmin=0, vmax=1)
    plt.colorbar(label="Wildfire Probability")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()