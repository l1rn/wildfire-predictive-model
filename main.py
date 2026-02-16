from src.preprocessing import process_data
import sys

def main():
    # run_gee_pipeline
    try:
        process_data()
    finally:
        print("Cleaning processes")
if __name__ == "__main__":
    main()
    sys.exit(1)    