"""
src/eda.py
----------
Generates and saves all EDA plots to outputs/.
No hardcoded absolute paths — uses relative paths from project root.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

logger = logging.getLogger(__name__)

_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_HERE)
_OUTPUT_DIR = os.path.join(_ROOT, "outputs")

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def _save(filename: str) -> None:
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(_OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", path)


def perform_eda(df: pd.DataFrame) -> None:
    """Run all EDA plots and save to outputs/."""
    df = df.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    # ── 1. Sales over time ─────────────────────────────────────────────────
    plt.figure(figsize=(14, 5))
    sales_ot = df.groupby("Date")["Weekly_Sales"].sum() / 1e6
    plt.plot(sales_ot.index, sales_ot.values, color="steelblue", linewidth=1.5)
    plt.fill_between(sales_ot.index, sales_ot.values, alpha=0.15, color="steelblue")

    holiday_dates = df[df["IsHoliday"].astype(str).isin(["True", "1"])]["Date"].unique()
    if len(holiday_dates):
        plt.vlines(holiday_dates, ymin=sales_ot.min(), ymax=sales_ot.max(),
                   color="red", alpha=0.15, linewidth=0.8, label="Holiday")
        plt.legend()

    plt.title("Total Weekly Sales Over Time (All Stores)", fontsize=14)
    plt.xlabel("Date"); plt.ylabel("Weekly Sales (Millions $)")
    plt.tight_layout()
    _save("sales_trend.png")
    plt.close()
    logger.info("Insight: Clear seasonal peaks in Nov–Dec each year.")

    # ── 2. Sales by month ──────────────────────────────────────────────────
    df["Month"] = df["Date"].dt.month
    monthly = df.groupby("Month")["Weekly_Sales"].mean()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(month_names, monthly.values, color="steelblue", alpha=0.85)
    plt.bar_label(bars, fmt="$%.0f", fontsize=8, padding=3)
    plt.title("Average Weekly Sales by Month", fontsize=13)
    plt.xlabel("Month"); plt.ylabel("Avg Weekly Sales ($)")
    plt.tight_layout()
    _save("sales_by_month.png")
    plt.close()

    # ── 3. Sales by store type ─────────────────────────────────────────────
    if "Type" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        type_sales = df.groupby("Type")["Weekly_Sales"].mean()
        axes[0].bar(type_sales.index, type_sales.values,
                    color=["steelblue", "coral", "mediumseagreen"])
        for i, v in enumerate(type_sales.values):
            axes[0].text(i, v + 500, f"${v:,.0f}", ha="center", fontsize=9)
        axes[0].set_title("Avg Weekly Sales by Store Type")
        axes[0].set_xlabel("Store Type"); axes[0].set_ylabel("Avg Weekly Sales ($)")

        axes[1].bar(month_names, monthly.values, color="steelblue", alpha=0.8)
        axes[1].set_title("Avg Weekly Sales by Month")
        axes[1].set_xlabel("Month"); axes[1].set_ylabel("Avg Weekly Sales ($)")
        plt.tight_layout()
        _save("sales_by_store_type.png")
        plt.close()

    # ── 4. Holiday effect ──────────────────────────────────────────────────
    holiday_sales = df.groupby("IsHoliday")["Weekly_Sales"].mean()
    plt.figure(figsize=(6, 4))
    labels = [str(k) for k in holiday_sales.index]
    bars = plt.bar(labels, holiday_sales.values, color=["steelblue", "coral"], width=0.4)
    plt.bar_label(bars, fmt="$%.0f", padding=3, fontsize=10, fontweight="bold")
    plt.title("Holiday vs Non-Holiday — Avg Weekly Sales")
    plt.xlabel("Is Holiday"); plt.ylabel("Avg Weekly Sales ($)")
    plt.tight_layout()
    _save("sales_by_holiday_effect.png")
    plt.close()

    # ── 5. Correlation heatmap ─────────────────────────────────────────────
    plt.figure(figsize=(10, 7))
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # Keep a sensible subset for readability
    key_cols = [c for c in
                ["Weekly_Sales", "Size", "Temperature", "Fuel_Price",
                 "CPI", "Unemployment", "IsHoliday"]
                if c in num_cols]
    corr = df[key_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5, annot_kws={"size": 10})
    plt.title("Correlation Heatmap — Key Feature Relationships", fontsize=13)
    plt.tight_layout()
    _save("sales_correlation_heatmap.png")
    plt.close()
    logger.info("Insight: Store Size shows strongest positive correlation with Weekly_Sales.")

    # ── 6. Sales distribution ──────────────────────────────────────────────
    plt.figure(figsize=(10, 4))
    clipped = df["Weekly_Sales"].clip(upper=df["Weekly_Sales"].quantile(0.99))
    plt.hist(clipped, bins=60, color="coral", edgecolor="white", alpha=0.85)
    plt.axvline(df["Weekly_Sales"].median(), color="navy", linestyle="--",
                label=f"Median: ${df['Weekly_Sales'].median():,.0f}")
    plt.axvline(df["Weekly_Sales"].mean(), color="green", linestyle="--",
                label=f"Mean: ${df['Weekly_Sales'].mean():,.0f}")
    plt.title("Weekly Sales Distribution (99th pct cap)")
    plt.xlabel("Weekly Sales ($)"); plt.ylabel("Frequency")
    plt.legend(); plt.tight_layout()
    _save("sales_distribution.png")
    plt.close()

    logger.info("EDA complete. All plots saved to: %s", _OUTPUT_DIR)