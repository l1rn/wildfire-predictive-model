import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, roc_auc_score
from src.config import PROCESSED_DIR
import matplotlib.pyplot as plt

def train_and_evaluate():
    df = pd.read_parquet(f"{PROCESSED_DIR}/khmao_master.parquet")
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
    
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    
    auc = roc_auc_score(y_test, probs)
    
    print("ROC-AUC: ", auc)
    print(classification_report(y_test, preds))
    
    importance = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=False)
    
    print(importance)