"""
src/data_processing.py
-----------------------
Clean → Feature Engineer → Preprocess
Industry standard: pure functions, no side effects, no inplace bugs.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# ── 1. CLEAN ──────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw merged DataFrame.

    Steps
    -----
    1. Drop exact duplicates
    2. Parse Date column to datetime
    3. Fill MarkDown1-5 nulls with 0  (no promotion = 0, not missing)
    4. Fill CPI / Unemployment with column median
    5. Fill remaining numeric nulls with median
    6. Fill remaining categorical nulls with mode
    7. Remove rows with negative Weekly_Sales (returns / corrections)
    8. Cap Weekly_Sales at 99th percentile (outlier handling)
    9. Encode IsHoliday as int
    """
    df = df.copy()

    # 1. Duplicates
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Dropped %d duplicate rows", before - len(df))

    # 2. Date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    # 3. MarkDowns  ← fill with 0, NOT median
    markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    for col in markdown_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 4. CPI / Unemployment
    for col in ["CPI", "Unemployment"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 5. Remaining numeric nulls
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        if col not in markdown_cols + ["CPI", "Unemployment"]:
            df[col] = df[col].fillna(df[col].median())

    # 6. Categorical nulls
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])   # ← fixed inplace bug

    # 7. Remove negative Weekly_Sales
    if "Weekly_Sales" in df.columns:
        before = len(df)
        df = df[df["Weekly_Sales"] >= 0].copy()
        logger.info("Removed %d rows with negative Weekly_Sales", before - len(df))

    # 8. Cap outliers at 99th percentile
    if "Weekly_Sales" in df.columns:
        cap = df["Weekly_Sales"].quantile(0.99)
        df["Weekly_Sales"] = df["Weekly_Sales"].clip(upper=cap)
        logger.info("Weekly_Sales capped at 99th pct: $%.2f", cap)

    # 9. IsHoliday → int
    if "IsHoliday" in df.columns:
        df["IsHoliday"] = df["IsHoliday"].astype(int)

    logger.info("clean_data output shape: %s", df.shape)
    return df


# ── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all model features from the cleaned DataFrame.

    Features created
    ----------------
    Time    : Year, Month, Week, Day, DayOfWeek, IsWeekend, Season
    Lag     : lag_1_week, lag_4_week, lag_52_week
    Rolling : rolling_mean_4, rolling_std_4
    Encode  : one-hot for Store Type (A/B/C)
    Drop    : Date column (all info extracted)
    """
    df = df.copy()

    # ── Time features ──────────────────────────────────────────
    if "Date" in df.columns:
        df["Year"]      = df["Date"].dt.year
        df["Month"]     = df["Date"].dt.month
        df["Week"]      = df["Date"].dt.isocalendar().week.astype(int)
        df["Day"]       = df["Date"].dt.day
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    # ── Season (retail-standard US seasons) ────────────────────
    def _season(month: int) -> int:
        if month in [12, 1, 2]:  return 1   # Winter
        if month in [3,  4, 5]:  return 2   # Spring
        if month in [6,  7, 8]:  return 3   # Summer
        return 4                             # Fall

    if "Month" in df.columns:
        df["Season"] = df["Month"].apply(_season)

    # ── Lag features (per Store × Dept) ────────────────────────
    if "Store" in df.columns and "Dept" in df.columns and "Weekly_Sales" in df.columns:
        df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

        grp = df.groupby(["Store", "Dept"])["Weekly_Sales"]

        df["lag_1_week"]  = grp.shift(1)
        df["lag_4_week"]  = grp.shift(4)
        df["lag_52_week"] = grp.shift(52)

        df["rolling_mean_4"] = (
            grp.transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
        )
        df["rolling_std_4"] = (
            grp.transform(lambda x: x.shift(1).rolling(4, min_periods=1).std())
        )
        df["rolling_std_4"] = df["rolling_std_4"].fillna(0)

    # ── One-hot encode Store Type (A / B / C) ──────────────────
    if "Type" in df.columns:
        dummies = pd.get_dummies(df["Type"], prefix="Type").astype(int)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=["Type"])

    # ── Drop Date (all info extracted) ─────────────────────────
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    logger.info("feature_engineering output shape: %s", df.shape)
    return df


# ── 3. PREPROCESS ─────────────────────────────────────────────────────────────

def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "Weekly_Sales",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Final preprocessing before model training.

    Steps
    -----
    1. Drop rows with NaN (created by lag features)
    2. Separate X / y
    3. Temporal train/test split  (shuffle=False — no data leakage)
    4. MinMaxScale numeric features
    5. Return X_train, X_test, y_train, y_test, scaler, feature_columns

    Returns
    -------
    tuple : (X_train, X_test, y_train, y_test, scaler, feature_columns)
    """
    df = df.copy()

    # 1. Drop NaN rows (lag features introduce NaN at group boundaries)
    before = len(df)
    df = df.dropna(subset=["lag_1_week", "lag_4_week", "lag_52_week"])
    logger.info("Dropped %d rows with NaN lag features", before - len(df))

    # 2. Split X / y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Ensure all X columns are numeric
    non_num = X.select_dtypes(exclude="number").columns.tolist()
    if non_num:
        logger.warning("Dropping non-numeric columns from X: %s", non_num)
        X = X.drop(columns=non_num)

    feature_columns = X.columns.tolist()

    # 3. Temporal split — shuffle=False prevents future leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )

    # 4. Scale  (fit on train only, transform both)
    scale_cols = [
        "Size", "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    ]
    scale_cols = [c for c in scale_cols if c in X_train.columns]

    scaler = MinMaxScaler()
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols]  = scaler.transform(X_test[scale_cols])

    logger.info(
        "preprocess_data: train=%s  test=%s  features=%d",
        X_train.shape, X_test.shape, len(feature_columns)
    )
    return X_train, X_test, y_train, y_test, scaler, feature_columns