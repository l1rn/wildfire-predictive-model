from src.preprocessing import process_data, upload_dataset_to_parquet
from src.models.train_model import train_and_evaluate
from src.models.models import get_xgboost, get_random_forest
import sys
import pandas as pd
from src.config import PROCESSED_DIR 

def main():
    # run_gee_pipeline
    try:
        # ds = process_data()
        # upload_dataset_to_parquet(ds)
        df = pd.read_parquet(f"{PROCESSED_DIR}/khmao_master.parquet")
        df = df.reset_index()
        
        print("Loaded: ", df.shape)
        
        train = df[df["year"] <= 2024]
        test = df[df["year"] >= 2025]
        
        print("Train: ", train.shape)
        print("Test: ", test.shape)
        
        features = [
            "temp",
            "vpd",
            "precip",
            "dem",
            "landcover",
            "ghm"
        ]
        
        X_train = train[features]
        y_train = train["fire"]
        
        X_test = test[features]
        y_test = test["fire"]
        
        xgb_model = get_xgboost(len(y_train) / y_train.sum())
        rf = get_random_forest()
        
        print("=== XGBoost ===")
        train_and_evaluate(model=xgb_model, 
                           X_train=X_train, 
                           y_train=y_train, 
                           X_test=X_test, 
                           y_test=y_test,
                           features=features)
        
        print("=== Random Forest ===")
        train_and_evaluate(model=rf, 
                           X_train=X_train, 
                           y_train=y_train, 
                           X_test=X_test, 
                           y_test=y_test,
                           features=features)
    finally:
        print("Cleaning processes")
if __name__ == "__main__":
    main()
    sys.exit(1)    