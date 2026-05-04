"""
main.py
-------
Runs the full ML training pipeline end-to-end:
Load → Clean → EDA → Feature Engineering → Preprocess → Train → Evaluate → Save

Usage:
    python main.py
    python main.py --skip-eda        # skip EDA plots (faster)
    python main.py --data-dir /path  # custom data directory
"""

import os
import sys
import json
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.load_data       import load_data
from src.data_processing import clean_data, feature_engineering, preprocess_data
from src.eda             import perform_eda
from src.model_training  import (
    train_all_models, build_ensemble, save_artefacts, generate_plots
)


def main(skip_eda: bool = False, data_dir: str = None) -> None:

    logger.info("=" * 60)
    logger.info("  DEMAND FORECASTING — TRAINING PIPELINE")
    logger.info("=" * 60)

    # ── 1. Load ───────────────────────────────────────────────────────
    kwargs = {"raw_dir": data_dir} if data_dir else {}
    df = load_data(**kwargs)
    logger.info("Step 1/7  Load        ✓  shape=%s", df.shape)

    # ── 2. Clean ──────────────────────────────────────────────────────
    df = clean_data(df)
    logger.info("Step 2/7  Clean       ✓  shape=%s", df.shape)

    # ── 3. EDA ────────────────────────────────────────────────────────
    if not skip_eda:
        perform_eda(df)
        logger.info("Step 3/7  EDA         ✓  plots saved to outputs/")
    else:
        logger.info("Step 3/7  EDA         — skipped")

    # ── 4. Feature engineering ────────────────────────────────────────
    df = feature_engineering(df)
    logger.info("Step 4/7  Features    ✓  shape=%s", df.shape)

    # ── 5. Preprocess ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, scaler, feature_columns = preprocess_data(df)
    logger.info("Step 5/7  Preprocess  ✓  train=%s  test=%s  features=%d",
                X_train.shape, X_test.shape, len(feature_columns))

    # ── 6. Train ──────────────────────────────────────────────────────
    results, trained_models, predictions = train_all_models(
        X_train, X_test, y_train, y_test
    )

    # Add ensemble
    ens_metrics, ens_preds = build_ensemble(predictions, results, y_test)
    results["Ensemble"]        = ens_metrics
    predictions["Ensemble"]    = ens_preds
    logger.info("Step 6/7  Train       ✓  models=%s", list(trained_models.keys()))

    # ── 7. Select best & save ─────────────────────────────────────────
    # Best single model (not ensemble) for the pkl alias
    best_name = max(
        (n for n in results if n != "Ensemble"),
        key=lambda n: results[n]["R2"]
    )
    logger.info("Step 7/7  Best model  ✓  %s  R²=%.4f",
                best_name, results[best_name]["R2"])

    save_artefacts(trained_models, best_name, scaler, feature_columns, results)
    generate_plots(best_name, trained_models[best_name],
                   feature_columns, predictions, y_test)

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("  FINAL RESULTS")
    logger.info("  %-22s  %8s  %10s  %10s", "Model", "R²", "MAE", "RMSE")
    logger.info("  %s", "-" * 55)
    for name, r in results.items():
        marker = " ← BEST" if name == best_name else ""
        logger.info("  %-22s  %8.4f  $%9.0f  $%9.0f%s",
                    name, r["R2"], r["MAE"], r["RMSE"], marker)
    logger.info("=" * 60)
    logger.info("Training pipeline complete. Run  python run.py  to start the web app.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the demand forecasting pipeline.")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA plots")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to directory containing raw CSVs")
    args = parser.parse_args()
    main(skip_eda=args.skip_eda, data_dir=args.data_dir)