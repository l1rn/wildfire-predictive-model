from sklearn.metrics import roc_auc_score
import numpy as np

def temporal_cross_validation(
    df,
    features,
    model_builder
):
    print(features)
    years = sorted(df["year"].unique())
    results = []
    
    for test_year in years[2:-1]:
        train = df[df["year"] < test_year]
        test = df[df["year"] == test_year]
        
        model = model_builder(train)
        
        model.fit(train[features], train["fire"])
        probs = model.predict_proba(test[features])[:, 1]
        
        auc = roc_auc_score(test["fire"], probs)
        results.append(auc)

        print(f"Year {test_year} ROC-AUC: {auc:.4f}")
        
    print("\nMean ROC-AUC:", np.mean(results))
    print("Std ROC-AUC:", np.std(results))

    return results