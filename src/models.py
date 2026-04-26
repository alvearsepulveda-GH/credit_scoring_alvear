import json
import pickle
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

SEED = 42

MODELOS_CONFIG = {
    "Random Forest": (
        RandomForestClassifier(random_state=SEED),
        {"n_estimators": [100, 200], "max_depth": [4, 6, None]},
    ),
    "XGBoost": (
        XGBClassifier(random_state=SEED, eval_metric="logloss", verbosity=0),
        {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
    ),
    "Logistic Regression": (
        LogisticRegression(random_state=SEED, max_iter=1000),
        {"C": [0.01, 0.1, 1, 10]},
    ),
}

def train_all_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Entrena modelos mediante GridSearchCV y retorna los mejores estimadores."""
    best_models = {}
    for nombre, (model, params) in MODELOS_CONFIG.items():
        gs = GridSearchCV(model, params, cv=5, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_train, y_train)
        best_models[nombre] = gs.best_estimator_
    return best_models

def evaluate_models(models: dict, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """Genera un ranking de modelos basado en la métrica AUC-ROC."""
    results = []
    for nombre, model in models.items():
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        results.append({"Modelo": nombre, "AUC": auc})
    return pd.DataFrame(results).sort_values(by="AUC", ascending=False)

def save_model(model, path: str, metadata: dict) -> None:
    """Guarda el modelo serializado y su metadata en la ruta especificada."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    
    with open(p / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    metadata["saved_at"] = date.today().isoformat()
    with open(p / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
