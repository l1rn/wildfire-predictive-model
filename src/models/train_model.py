import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from src.config import PROCESSED_DIR

def train_model_on_master_table():
    df = pd.read_parquet(f"{PROCESSED_DIR}/khmao_master.parquet")
    print("Loaded: ", df.shape)
    df = df.reset_index()
    
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    df["year"] = df["valid_time"].dt.year
    
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
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        n_jobs=1,
        random_state=42
    )
    
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