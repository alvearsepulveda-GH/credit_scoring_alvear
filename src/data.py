import pandas as pd
import sys
import types

# --- PARCHE DE COMPATIBILIDAD ---
# Inyectamos un pkg_resources falso para engañar a ydata_profiling
try:
    import pkg_resources
except ImportError:
    mock_pkg = types.ModuleType("pkg_resources")
    mock_pkg.get_distribution = lambda x: types.SimpleNamespace(version="0.0.0")
    sys.modules["pkg_resources"] = mock_pkg

from ydata_profiling import ProfileReport
# ... el resto de tus funciones (load_raw, etc.)




REQUIRED_COLUMNS = [
    "Age", "Employ", "Address", "Income",
    "Creddebt", "OthDebt", "MonthlyLoad", "Default"
]

def load_raw(path: str, **kwargs) -> pd.DataFrame:
    """Carga el dataset crudo desde un CSV."""
    df = pd.read_csv(path, **kwargs)
    validate_schema(df)
    return df

def validate_schema(df: pd.DataFrame) -> None:
    """Lanza ValueError si falta alguna columna requerida."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes: {missing}")

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features derivadas y elimina filas con NaN."""
    df = df.copy()
    df["OthDebtRatio"] = df["OthDebt"] / df["Income"]
    return df.dropna()

def generate_eda_report(df: pd.DataFrame, output_file: str = "report_eda.html") -> None:
    """Genera un ProfileReport detallado y lo guarda como HTML."""
    report = ProfileReport(
        df, 
        title="Análisis Exploratorio - Credit Scoring",
        explorative=True
    )
    report.to_file(output_file)