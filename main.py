from src.data.data_loader import load_master_dataset
from src.models.train_model import train_model, evaluate_model, generate_forecast
from src.models.models import get_xgboost, get_random_forest

from src.visualization import maps
from src import preprocessing
from src.models import cross_validation
import src.features.features as ft
from src.data import split

import sys
from src.config import PROCESSED_DIR 

def build_xgb(train):
    scale_pos_weight = len(train) / train["fire"].sum()
    return get_xgboost(scale_pos_weight)

def main():
    # run_gee_pipeline
    try:
        # ds = preprocessing.process_data()
        # preprocessing.upload_dataset_to_parquet(ds)
        df = load_master_dataset()
        print("Loaded: ", df.shape)

        df = ft.prepare_features(df)
        train, test, future = split.temporal_split(df)    
        
        print("Train: ", train.shape)
        print("Test: ", test.shape)
        
        features1 = [
            "temp",
            "vpd",
            "precip",
            "vpd_ghm_interaction",

            "dem",
            "landcover",
            "ghm",
        ]
        
        rf = get_random_forest()

        # cross_validation.temporal_cross_validation(
            # df,
            # features1,
            # build_xgb
        # )
                
        X_train = df[features1]
        y_train = df["fire"]
        
        xgb = get_xgboost(len(y_train) / y_train.sum())
        
        main_model = train_model(
            model=xgb, X_train=X_train, y_train=y_train
        )
        
        probs = evaluate_model(
            model=main_model, X_test=X_train, y_test=y_train, features=features1
        )
        
        future = generate_forecast(
            model=main_model, df=future, features=features1
        )

        maps.plot_month_map(
            future,
            year=2026,
            month=1,
            title="Wildfire Forecast – January 2026",
            save_path="wildfire_risk_jan_2026.jpg"
        )
        print("=== Random Forest ===")
    finally:
        print("Cleaning processes")
if __name__ == "__main__":
    main()
    sys.exit(1)    