"""
src/load_data.py
----------------
Loads and merges the Walmart raw CSVs into a single DataFrame.
Industry standard: path-agnostic, validated, logged.
"""

import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Resolve paths relative to this file ──────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(_HERE)
_RAW_DIR  = os.path.join(_ROOT, "data", "raw")

REQUIRED_FILES = ["train.csv", "features.csv", "stores.csv"]


def _validate_files(raw_dir: str) -> None:
    """Raise FileNotFoundError if any required CSV is missing."""
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(raw_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing raw data files in '{raw_dir}': {missing}\n"
            "Download from: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data"
        )


def load_data(raw_dir: str = _RAW_DIR) -> pd.DataFrame:
    """
    Load and merge train.csv + features.csv + stores.csv.

    Parameters
    ----------
    raw_dir : str
        Directory that contains the three raw CSVs.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all columns from the three sources.
    """
    _validate_files(raw_dir)

    logger.info("Loading raw CSVs from: %s", raw_dir)

    train    = pd.read_csv(os.path.join(raw_dir, "train.csv"))
    features = pd.read_csv(os.path.join(raw_dir, "features.csv"))
    stores   = pd.read_csv(os.path.join(raw_dir, "stores.csv"))

    logger.info("train    shape : %s", train.shape)
    logger.info("features shape : %s", features.shape)
    logger.info("stores   shape : %s", stores.shape)

    # Merge on shared keys
    df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    df = df.merge(stores,   on="Store",                          how="left")

    logger.info("Merged shape   : %s", df.shape)
    logger.info("Columns        : %s", df.columns.tolist())

    # Quick null report
    nulls = df.isnull().sum()
    null_report = nulls[nulls > 0]
    if not null_report.empty:
        logger.info("Columns with nulls:\n%s", null_report.to_string())

    return df


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    print(df.head(3))
    print("Shape      :", df.shape)
    print("Columns    :", df.columns.tolist())
    print("Duplicates :", df.duplicated().sum())
    print("Nulls:\n",   df.isnull().sum()[df.isnull().sum() > 0])