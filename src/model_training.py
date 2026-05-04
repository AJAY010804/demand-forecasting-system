"""
src/model_training.py
----------------------
Trains Linear Regression, Random Forest, XGBoost.
Selects best model by R². Saves models, scaler, feature columns, results JSON.
Industry standard: function-based, no top-level execution, importable.
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model  import LinearRegression
from sklearn.ensemble      import RandomForestRegressor
from sklearn.metrics       import r2_score, mean_absolute_error, mean_squared_error
from xgboost               import XGBRegressor

logger = logging.getLogger(__name__)

_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_HERE)
_MODELS_DIR = os.path.join(_ROOT, "models")
_OUTPUT_DIR = os.path.join(_ROOT, "outputs")


# ── Model definitions ─────────────────────────────────────────────────────────

def get_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42,
            verbosity=0,
            objective="reg:squarederror",
        ),
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def _train_one(name, model, X_train, X_test, y_train, y_test) -> tuple:
    """Train a single model and return (metrics_dict, predictions, trained_model)."""
    logger.info("Training %s ...", name)
    t0 = time.time()

    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_test), 0, None)

    elapsed = time.time() - t0
    r2   = r2_score(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    mse  = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    metrics = {"R2": float(r2), "MAE": float(mae),
               "MSE": float(mse), "RMSE": float(rmse), "Time": float(elapsed)}

    logger.info("  Done in %.1fs  R²=%.4f  MAE=%.2f  RMSE=%.2f", elapsed, r2, mae, rmse)
    return metrics, preds, model


def train_all_models(X_train, X_test, y_train, y_test) -> tuple:
    """
    Train all models. Returns (results, trained_models, predictions).

    Returns
    -------
    results        : dict  {model_name: {R2, MAE, MSE, RMSE, Time}}
    trained_models : dict  {model_name: fitted_model}
    predictions    : dict  {model_name: np.ndarray}
    """
    models_dict    = get_models()
    results        = {}
    trained_models = {}
    predictions    = {}

    for name, model in models_dict.items():
        metrics, preds, fitted = _train_one(name, model, X_train, X_test, y_train, y_test)
        results[name]        = metrics
        trained_models[name] = fitted
        predictions[name]    = preds

    return results, trained_models, predictions


# ── Ensemble ──────────────────────────────────────────────────────────────────

def build_ensemble(predictions: dict, results: dict, y_test) -> tuple:
    """Weighted average ensemble (weights ∝ R² score)."""
    total_r2 = sum(r["R2"] for r in results.values())
    weights  = {n: r["R2"] / total_r2 for n, r in results.items()}

    ensemble_pred = sum(w * predictions[n] for n, w in weights.items())
    ensemble_pred = np.clip(ensemble_pred, 0, None)

    metrics = {
        "R2":   float(r2_score(y_test, ensemble_pred)),
        "MAE":  float(mean_absolute_error(y_test, ensemble_pred)),
        "MSE":  float(mean_squared_error(y_test, ensemble_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, ensemble_pred))),
        "Time": 0.0,
        "Weights": {n: round(w, 4) for n, w in weights.items()},
    }
    logger.info("Ensemble  R²=%.4f  MAE=%.2f  RMSE=%.2f",
                metrics["R2"], metrics["MAE"], metrics["RMSE"])
    return metrics, ensemble_pred


# ── Save artefacts ────────────────────────────────────────────────────────────

def save_artefacts(
    trained_models: dict,
    best_model_name: str,
    scaler,
    feature_columns: list,
    results: dict,
) -> None:
    """Save all model artefacts and results JSON."""
    os.makedirs(_MODELS_DIR, exist_ok=True)
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    # Individual models
    for name, model in trained_models.items():
        fname = name.replace(" ", "_").lower() + ".pkl"
        joblib.dump(model, os.path.join(_MODELS_DIR, fname))
        logger.info("Saved model: %s", fname)

    # Best model alias
    joblib.dump(trained_models[best_model_name],
                os.path.join(_MODELS_DIR, "best_model.pkl"))

    # Scaler + feature columns (CRITICAL for prediction pipeline)
    joblib.dump(scaler,          os.path.join(_MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_columns, os.path.join(_MODELS_DIR, "feature_columns.pkl"))
    logger.info("Saved scaler and feature_columns")

    # Results JSON (used by Flask dashboard)
    with open(os.path.join(_OUTPUT_DIR, "model_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Saved model_results.json")


# ── Plots ─────────────────────────────────────────────────────────────────────

def generate_plots(
    best_model_name: str,
    best_model,
    feature_columns: list,
    predictions: dict,
    y_test,
) -> None:
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=feature_columns).nlargest(15)
        plt.figure(figsize=(10, 5))
        fi.sort_values().plot(kind="barh", color="steelblue")
        plt.title(f"Feature Importance — {best_model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(_OUTPUT_DIR, "feature_importance.png"), dpi=150)
        plt.close()

    # Actual vs Predicted (sampled scatter)
    preds = predictions[best_model_name]
    sample_idx = np.random.choice(len(y_test), min(3000, len(y_test)), replace=False)
    y_s = np.array(y_test)[sample_idx]
    p_s = preds[sample_idx]

    plt.figure(figsize=(6, 6))
    plt.scatter(y_s, p_s, alpha=0.3, s=8, color="coral")
    lim = [0, max(y_s.max(), p_s.max())]
    plt.plot(lim, lim, "k--", linewidth=1, label="Perfect prediction")
    plt.xlabel("Actual Weekly Sales ($)")
    plt.ylabel("Predicted Weekly Sales ($)")
    plt.title(f"{best_model_name} — Actual vs Predicted")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(_OUTPUT_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close()

    # Model comparison bar chart
    logger.info("Plots saved to: %s", _OUTPUT_DIR)