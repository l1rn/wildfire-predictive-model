from src.preprocessing import process_data, upload_dataset_to_parquet
from src.models.train_model import train_and_evaluate
import sys
import pandas as pd
from src.config import PROCESSED_DIR 

def main():
    # run_gee_pipeline
    try:
        ds = process_data()
        upload_dataset_to_parquet(ds)
        # train_model_on_master_table()        
    finally:
        print("Cleaning processes")
if __name__ == "__main__":
    main()
    sys.exit(1)    