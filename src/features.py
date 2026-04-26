import numpy as np
import pandas as pd

def compute_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str,
    bins: int = 10
) -> tuple[pd.DataFrame, float]:
    """Calcula las tablas de WoE y el valor de IV para una variable continua."""
    df = df[[feature, target]].copy()
    df["bin"] = pd.qcut(df[feature], q=bins, duplicates="drop")
    
    grouped = df.groupby("bin", observed=True)[target].agg(["count", "sum"])
    grouped.columns = ["n_obs", "n_events"]
    grouped["n_non_events"] = grouped["n_obs"] - grouped["n_events"]
    
    total_events = grouped["n_events"].sum()
    total_non_events = grouped["n_non_events"].sum()
    
    grouped["dist_events"] = grouped["n_events"] / total_events
    grouped["dist_non_events"] = grouped["n_non_events"] / total_non_events
    
    # Reemplazo de ceros para evitar log(0) o división por cero
    grouped["dist_events"] = grouped["dist_events"].replace(0, 0.0001)
    grouped["dist_non_events"] = grouped["dist_non_events"].replace(0, 0.0001)
    
    grouped["woe"] = np.log(grouped["dist_events"] / grouped["dist_non_events"])
    grouped["iv_bin"] = (grouped["dist_events"] - grouped["dist_non_events"]) * grouped["woe"]
    
    iv = grouped["iv_bin"].sum()
    return grouped.reset_index()[["bin", "n_events", "n_non_events", "woe", "iv_bin"]], iv

def select_features_by_iv(
    df: pd.DataFrame,
    target: str,
    threshold: float = 0.1
) -> list[str]:
    """Retorna la lista de variables cuyo IV supera el umbral definido."""
    selected_features = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target)
    
    for col in numeric_cols:
        _, iv = compute_woe_iv(df, col, target)
        if iv >= threshold:
            selected_features.append(col)
    return selected_features

def build_woe_tables(
    df: pd.DataFrame,
    features: list[str],
    target: str
) -> dict[str, pd.DataFrame]:
    """Genera un diccionario con las tablas WoE para cada variable seleccionada."""
    return {feat: compute_woe_iv(df, feat, target)[0] for feat in features}

def transform_woe(
    df: pd.DataFrame,
    woe_tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Reemplaza cada feature por su valor WoE manejando duplicados y valores fuera de rango."""
    df_woe = pd.DataFrame(index=df.index)
    
    for feat, table in woe_tables.items():
        # Extraemos los intervalos y los valores WoE
        intervals = table["bin"].tolist()
        woe_values = table["woe"].values
        
        # Ajuste de límites para evitar NaNs (cubrir -inf a inf)
        new_bins = [intervals[0].left] + [i.right for i in intervals]
        new_bins[0] = -float('inf')
        new_bins[-1] = float('inf')
        
        # SOLUCIÓN AL ERROR: 
        # Usamos pd.cut para obtener los índices de los bins y luego mapeamos los valores WoE.
        # Esto evita el conflicto de 'labels must be unique'.
        bin_indices = pd.cut(
            df[feat], 
            bins=new_bins, 
            include_lowest=True, 
            labels=False  # Retorna enteros (0, 1, 2...) en lugar de las etiquetas duplicadas
        )
        
        # Mapeamos los índices a los valores reales de WoE
        df_woe[f"{feat}_woe"] = bin_indices.map(lambda x: woe_values[x] if pd.notnull(x) else 0).astype(float)
        
    return df_woe.fillna(0)