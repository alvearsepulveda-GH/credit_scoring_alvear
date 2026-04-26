import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def auc_roc(model, X: np.ndarray, y: np.ndarray) -> float:
    """Calcula el área bajo la curva ROC para un modelo y dataset dados."""
    return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))

def costo_total(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    umbral: float,
    c_fn: float = 500,
    c_fp: float = 100
) -> float:
    """Calcula el costo económico total de las predicciones según un umbral."""
    y_pred = (y_prob >= umbral).astype(int)
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(fn * c_fn + fp * c_fp)

def build_scorecard(
    model_lr,
    woe_tables: dict,
    base_score: int = 300,
    pdo: int = 50
) -> pd.DataFrame:
    """Transforma los coeficientes de Regresión Logística en un sistema de puntaje."""
    factor = pdo / np.log(2)
    # Asumiendo odds base de 1 para simplificar según instrucciones estándar
    offset = base_score - factor * np.log(1)
    
    intercepto = model_lr.intercept_[0]
    coefs = model_lr.coef_[0]
    n_vars = len(woe_tables)
    
    scorecard_list = []
    for i, (feat, table) in enumerate(woe_tables.items()):
        temp_table = table.copy()
        temp_table["feature"] = feat
        # Fórmula: -(coef * woe + intercepto/n_vars) * factor + (offset/n_vars)
        temp_table["puntos"] = -(coefs[i] * temp_table["woe"] + intercepto/n_vars) * factor + (offset/n_vars)
        scorecard_list.append(temp_table[["feature", "bin", "woe", "puntos"]])
        
    return pd.concat(scorecard_list, ignore_index=True)