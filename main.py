from src.data.data_loader import load_master_dataset
from src.models.train_model import train_model, evaluate_model, generate_forecast
from src.models.models import get_xgboost, get_random_forest

from src.visualization import maps
import src.features.features as ft
from src.data import split

import sys
from src.config import PROCESSED_DIR 

def main():
    # run_gee_pipeline
    try:
        # ds = process_data()
        # upload_dataset_to_parquet(ds)
        df = load_master_dataset()
        print("Loaded: ", df.shape)

        df = ft.prepare_features(df)
        train, test, future = split.temporal_split(df)    
        
        print("Train: ", train.shape)
        print("Test: ", test.shape)
        
        features = [
            "temp",
            "vpd",
            "precip",
            "dem",
            "landcover",
            "ghm",
            "vpd_ghm_interaction"
        ]
        
        X_train = train[features]
        y_train = train["fire"]
        
        X_test = test[features]
        y_test = test["fire"]
        
        xgb_model = get_xgboost(len(y_train) / y_train.sum())
        rf = get_random_forest()
        
        main_model = train_model(
            model=xgb_model, X_train=X_train, y_train=y_train
        )
        
        probs = evaluate_model(
            model=main_model, X_test=X_test, y_test=y_test, features=features
        )
        
        future = generate_forecast(
            model=main_model, df=future, features=features
        )

        maps.plot_month_map(
            future,
            year=2026,
            month=1,
            title="Wildfire Forecast – January 2026",
            save_path="wildfire_risk_jan_2026.jpg"
        )
        # print("=== Random Forest ===")
        # train_and_evaluate(model=rf, 
        #                    X_train=X_train, 
        #                    y_train=y_train, 
        #                    X_test=X_test, 
        #                    y_test=y_test,
        #                    features=features,
        #                    test=df)
    finally:
        print("Cleaning processes")
if __name__ == "__main__":
    main()
    sys.exit(1)    