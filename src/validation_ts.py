from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TRAIN_COLOR = "#2196F3"
TEST_COLOR = "#FF5722"
GAP_COLOR = "#9E9E9E"


@dataclass(frozen=True)
class WalkForwardConfig:
    initial_train: int = 504
    test_size: int = 63
    step: int = 21
    gap: int = 5


def configure_plots() -> None:
    """Aplica una configuracion visual consistente para el notebook."""
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        }
    )


def load_market_returns(
    tickers: list[str] | tuple[str, ...],
    start: str = "2010-01-01",
    end: str = "2024-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descarga precios ajustados desde Yahoo Finance y calcula retornos diarios."""
    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].dropna(how="all")
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]}).dropna(how="all")

    returns = prices.pct_change().dropna()
    return prices, returns


def autocorrelation_summary(returns: pd.Series) -> pd.DataFrame:
    """Resume autocorrelacion simple de retorno y proxies de volatilidad."""
    return pd.DataFrame(
        {
            "serie": ["retorno", "retorno_absoluto", "retorno_cuadrado"],
            "acf_lag_1": [
                returns.autocorr(lag=1),
                returns.abs().autocorr(lag=1),
                returns.pow(2).autocorr(lag=1),
            ],
        }
    )


def build_basic_next_return_dataset(returns: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Crea un dataset simple para predecir direccion del retorno siguiente."""
    df = pd.DataFrame(index=returns.index)
    df["ret_lag1"] = returns.shift(1)
    df["vol_20"] = returns.rolling(20).std()
    df["target_next_up"] = (returns.shift(-1) > 0).astype(int)
    df = df.dropna()
    return df[["ret_lag1", "vol_20"]], df["target_next_up"]


def compare_random_vs_temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compara un split aleatorio contra una validacion out-of-time."""
    model = LogisticRegression(max_iter=2000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=random_state
    )
    random_auc = _fit_auc(model, X_train, y_train, X_test, y_test)

    cut = int(len(X) * (1 - test_size))
    temporal_auc = _fit_auc(model, X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:])

    return pd.DataFrame(
        {
            "validacion": ["Split aleatorio", "Out-of-time temporal"],
            "auc": [random_auc, temporal_auc],
        }
    )


def scaler_leakage_example(returns: pd.Series, train_ratio: float = 0.80) -> dict[str, float]:
    """Muestra como cambia el escalamiento cuando se ajusta con todo el dataset."""
    values = returns.dropna().to_numpy().reshape(-1, 1)
    split_idx = int(len(values) * train_ratio)

    bad_scaler = StandardScaler()
    good_scaler = StandardScaler()
    bad_scaler.fit(values)
    good_scaler.fit(values[:split_idx])

    return {
        "media_scaler_con_leakage": float(bad_scaler.mean_[0]),
        "media_scaler_correcto": float(good_scaler.mean_[0]),
        "n_train": split_idx,
        "n_test": len(values) - split_idx,
    }


def rolling_leakage_example(returns: pd.Series, window: int = 10) -> pd.DataFrame:
    """Contrasta rolling centrado incorrecto contra rolling historico desplazado."""
    df = returns.to_frame("retorno")
    df["rolling_centrado_incorrecto"] = df["retorno"].rolling(window=window, center=True).mean()
    df["rolling_historico"] = df["retorno"].rolling(window=window, center=False).mean()
    df["rolling_historico_lag1"] = df["rolling_historico"].shift(1)
    return df.dropna().head(10)


def label_leakage_example(returns: pd.Series) -> pd.DataFrame:
    """Compara un target mal definido con un target futuro correctamente desplazado."""
    df = returns.to_frame("ret_t")
    df["label_incorrecto"] = (df["ret_t"] > 0).astype(int)
    df["label_correcto"] = (df["ret_t"].shift(-1) > 0).astype(int)
    df["feature_ret_lag1"] = df["ret_t"].shift(1)
    return df.dropna().head(10)


def build_time_series_features(returns: pd.Series, horizon: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    """Construye features temporales usando solo informacion disponible hasta t."""
    r = returns.copy()
    df = pd.DataFrame(index=r.index)

    for lag in [1, 2, 3, 5, 10]:
        df[f"ret_lag_{lag}"] = r.shift(lag)

    shifted = r.shift(1)
    for window in [5, 10, 20, 60]:
        df[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        df[f"rolling_std_{window}"] = shifted.rolling(window).std()

    df["rolling_skew_20"] = shifted.rolling(20).skew()
    df["rsi_14"] = _rsi(shifted, 14)
    df["target_next_up"] = (r.shift(-horizon) > 0).astype(int)
    df = df.dropna()
    return df.drop(columns="target_next_up"), df["target_next_up"]


def compare_cv_strategies(X: pd.DataFrame, y: pd.Series, n_splits: int = 10) -> pd.DataFrame:
    """Compara K-Fold aleatorio con TimeSeriesSplit usando AUC por fold."""
    pipe = make_logistic_pipeline(C=0.1)
    rows = []
    strategies = {
        "K-Fold aleatorio": KFold(n_splits=n_splits, shuffle=True, random_state=42),
        "TimeSeriesSplit": TimeSeriesSplit(n_splits=n_splits),
    }

    for name, splitter in strategies.items():
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
            auc = _fit_auc(
                clone(pipe),
                X.iloc[train_idx],
                y.iloc[train_idx],
                X.iloc[test_idx],
                y.iloc[test_idx],
            )
            rows.append({"estrategia": name, "fold": fold, "auc": auc})

    return pd.DataFrame(rows)


def make_lagged_direction_dataset(returns: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Dataset minimo para walk-forward: retorno rezagado predice direccion siguiente."""
    X = returns.shift(1).to_frame("ret_lag_1").dropna()
    y = (returns.shift(-1) > 0).astype(int).reindex(X.index).dropna()
    return X.align(y, join="inner", axis=0)


def walk_forward_auc(
    X: pd.DataFrame,
    y: pd.Series,
    config: WalkForwardConfig = WalkForwardConfig(),
) -> pd.DataFrame:
    """Evalua AUC usando ventanas walk-forward con gap temporal."""
    pipe = make_logistic_pipeline(C=0.1)
    rows = []
    end = config.initial_train

    while end + config.gap + config.test_size <= len(X):
        X_train = X.iloc[:end]
        y_train = y.iloc[:end]
        X_test = X.iloc[end + config.gap : end + config.gap + config.test_size]
        y_test = y.iloc[end + config.gap : end + config.gap + config.test_size]

        auc = _fit_auc(clone(pipe), X_train, y_train, X_test, y_test)
        rows.append(
            {
                "fold": len(rows) + 1,
                "test_start": X_test.index[0].strftime("%Y-%m"),
                "n_train": len(X_train),
                "auc": auc,
            }
        )
        end += config.step

    return pd.DataFrame(rows)


def nested_walk_forward_auc(
    X: pd.DataFrame,
    y: pd.Series,
    config: WalkForwardConfig = WalkForwardConfig(),
    param_grid: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """Hace tuning interno temporal y evaluacion externa out-of-time."""
    if param_grid is None:
        param_grid = {"model__C": [0.01, 0.1, 1.0, 10.0]}

    pipe = make_logistic_pipeline()
    rows = []
    end = config.initial_train

    while end + config.gap + config.test_size <= len(X):
        X_train = X.iloc[:end]
        y_train = y.iloc[:end]
        X_test = X.iloc[end + config.gap : end + config.gap + config.test_size]
        y_test = y.iloc[end + config.gap : end + config.gap + config.test_size]

        grid = GridSearchCV(
            clone(pipe),
            param_grid=param_grid,
            cv=TimeSeriesSplit(n_splits=3),
            scoring="roc_auc",
        )
        grid.fit(X_train, y_train)
        outer_auc = roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1])

        rows.append(
            {
                "fold": len(rows) + 1,
                "test_start": X_test.index[0].strftime("%Y-%m"),
                "n_train": len(X_train),
                "best_C": grid.best_params_["model__C"],
                "inner_auc": grid.best_score_,
                "outer_auc": outer_auc,
            }
        )
        end += config.step

    result = pd.DataFrame(rows)
    result["brecha_inner_outer"] = result["inner_auc"] - result["outer_auc"]
    return result


def summarize_walk_forward(results: pd.DataFrame, auc_column: str = "auc") -> pd.DataFrame:
    """Resume estabilidad de resultados walk-forward."""
    return pd.DataFrame(
        {
            "metrica": ["AUC medio", "AUC std", "Folds sobre 0.5"],
            "valor": [
                results[auc_column].mean(),
                results[auc_column].std(),
                (results[auc_column] > 0.5).mean(),
            ],
        }
    )


def plot_split_strategies(n_samples: int = 60, n_splits: int = 5) -> plt.Figure:
    """Visualiza K-Fold aleatorio versus TimeSeriesSplit."""
    dummy = np.arange(n_samples).reshape(-1, 1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 5))
    splitters = [
        ("K-Fold aleatorio", KFold(n_splits=n_splits, shuffle=True, random_state=42)),
        ("TimeSeriesSplit / Walk-Forward", TimeSeriesSplit(n_splits=n_splits)),
    ]

    for ax, (title, splitter) in zip(axes, splitters):
        for fold, (train_idx, test_idx) in enumerate(splitter.split(dummy)):
            ax.barh(fold, len(train_idx), left=train_idx[0], height=0.6, color=TRAIN_COLOR, alpha=0.8)
            ax.barh(fold, len(test_idx), left=test_idx[0], height=0.6, color=TEST_COLOR, alpha=0.8)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Indice temporal")
        ax.set_ylabel("Fold")

    fig.suptitle("Comparacion de estrategias de validacion", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_cv_comparison(scores: pd.DataFrame) -> plt.Figure:
    """Grafica distribucion de AUC por estrategia de validacion."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=scores, x="estrategia", y="auc", ax=ax)
    ax.axhline(0.5, ls="--", color="gray", lw=1)
    ax.set_title("K-Fold vs TimeSeriesSplit")
    ax.set_xlabel("")
    ax.set_ylabel("AUC ROC")
    fig.tight_layout()
    return fig


def plot_gap_scenarios(
    n_obs: int = 120,
    train_size: int = 60,
    test_size: int = 20,
    rolling_window: int = 20,
) -> plt.Figure:
    """Visualiza el rol del gap entre train y test."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 5))
    scenarios = [(0, "Sin gap"), (rolling_window, f"Gap = {rolling_window}")]

    for ax, (gap, label) in zip(axes, scenarios):
        test_start = train_size + gap
        ax.barh(0, train_size, left=0, height=0.5, color=TRAIN_COLOR, alpha=0.8, label="Train")
        ax.barh(0, test_size, left=test_start, height=0.5, color=TEST_COLOR, alpha=0.8, label="Test")
        if gap:
            ax.barh(0, gap, left=train_size, height=0.5, color=GAP_COLOR, alpha=0.7, label="Gap")
        else:
            ax.axvspan(train_size - rolling_window, train_size + rolling_window, alpha=0.25, color="red")
        ax.set_xlim(0, n_obs)
        ax.set_yticks([])
        ax.set_xlabel("Indice temporal")
        ax.set_title(label, fontweight="bold")
        ax.legend(loc="lower right")

    fig.tight_layout()
    return fig


def plot_walk_forward(results: pd.DataFrame) -> plt.Figure:
    """Grafica AUC por fold walk-forward."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(results["test_start"], results["auc"], marker="o", ms=4, color="steelblue")
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="No-skill")
    tick_step = max(1, len(results) // 8)
    ax.set_xticks(range(0, len(results), tick_step))
    ax.set_xticklabels(results["test_start"].iloc[::tick_step], rotation=30)
    ax.set_ylabel("AUC ROC")
    ax.set_title("Walk-forward: AUC por fold")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_nested_results(results: pd.DataFrame) -> plt.Figure:
    """Grafica resultados de nested walk-forward."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    folds = results["fold"]

    axes[0, 0].plot(folds, results["inner_auc"], marker="o", ms=4, label="Inner", color="steelblue")
    axes[0, 0].plot(folds, results["outer_auc"], marker="o", ms=4, label="Outer", color="tomato")
    axes[0, 0].axhline(0.5, ls="--", color="gray", lw=1)
    axes[0, 0].set_title("AUC inner vs outer")
    axes[0, 0].legend()

    axes[0, 1].bar(folds, results["brecha_inner_outer"], color="steelblue")
    axes[0, 1].axhline(0.05, ls="--", color="gray", lw=1)
    axes[0, 1].set_title("Brecha inner - outer")

    results["best_C"].value_counts().sort_index().plot(kind="bar", ax=axes[1, 0], color="steelblue")
    axes[1, 0].set_title("Mejor C por fold")
    axes[1, 0].tick_params(axis="x", rotation=0)

    sns.boxplot(data=results[["inner_auc", "outer_auc"]], ax=axes[1, 1])
    axes[1, 1].axhline(0.5, ls="--", color="gray", lw=1)
    axes[1, 1].set_title("Distribucion de AUC")

    fig.suptitle("Nested walk-forward temporal", fontweight="bold")
    fig.tight_layout()
    return fig


def make_logistic_pipeline(C: float = 1.0) -> Pipeline:
    """Pipeline con escalamiento dentro del entrenamiento para evitar leakage."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, C=C)),
        ]
    )


def _fit_auc(
    model: Pipeline | LogisticRegression,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> float:
    model.fit(X_train, y_train)
    return float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))


def _rsi(returns: pd.Series, window: int = 14) -> pd.Series:
    gain = returns.clip(lower=0).rolling(window).mean()
    loss = (-returns.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))
