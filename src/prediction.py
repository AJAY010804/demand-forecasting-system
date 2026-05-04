"""
src/prediction.py
-----------------
Prediction pipeline — mirrors the training pipeline exactly.
Used by Flask API to serve real-time and batch predictions.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib

from src.data_processing import clean_data, feature_engineering

logger = logging.getLogger(__name__)

_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_HERE)
_MODELS_DIR = os.path.join(_ROOT, "models")


# ── Load artefacts once at module level ───────────────────────────────────────

def _load_artefacts():
    model_path   = os.path.join(_MODELS_DIR, "best_model.pkl")
    scaler_path  = os.path.join(_MODELS_DIR, "scaler.pkl")
    columns_path = os.path.join(_MODELS_DIR, "feature_columns.pkl")

    missing = [p for p in [model_path, scaler_path, columns_path]
               if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Model artefacts missing: {missing}\nRun main.py first."
        )

    model           = joblib.load(model_path)
    scaler          = joblib.load(scaler_path)
    feature_columns = joblib.load(columns_path)
    logger.info("Artefacts loaded: model=%s  features=%d",
                type(model).__name__, len(feature_columns))
    return model, scaler, feature_columns


try:
    _MODEL, _SCALER, _FEATURE_COLUMNS = _load_artefacts()
    _ARTEFACTS_LOADED = True
except FileNotFoundError as e:
    logger.warning("Prediction artefacts not found. Run main.py first.\n%s", e)
    _MODEL = _SCALER = _FEATURE_COLUMNS = None
    _ARTEFACTS_LOADED = False

# Scaler columns (must match training)
_SCALE_COLS = ["Size", "Temperature", "Fuel_Price", "CPI", "Unemployment",
               "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]


# ── Single prediction ─────────────────────────────────────────────────────────

def predict_sales(input_data: dict) -> dict:
    """
    Predict weekly sales for a single row of input.

    Parameters
    ----------
    input_data : dict
        Raw field values matching the training dataset schema.
        Required keys: Store, Dept, Date, IsHoliday, Temperature,
                       Fuel_Price, CPI, Unemployment.
        Optional: MarkDown1-5, Size, Type.

    Returns
    -------
    dict : {prediction, low_estimate, high_estimate, model_used}
    """
    if not _ARTEFACTS_LOADED:
        raise RuntimeError("Model not trained yet. Run main.py first.")

    df = pd.DataFrame([input_data])

    # Fill any missing columns with safe defaults before pipeline
    _DEFAULTS = {
        "MarkDown1": 0, "MarkDown2": 0, "MarkDown3": 0, "MarkDown4": 0, "MarkDown5": 0,
        "Temperature": 60.0, "Fuel_Price": 3.4, "CPI": 211.0, "Unemployment": 7.0,
        "Size": 151315, "Type": "A", "IsHoliday": 0,
        "lag_1_week": 20000.0, "lag_4_week": 18000.0,
        "lag_52_week": 19000.0, "rolling_mean_4": 19500.0, "rolling_std_4": 500.0,
    }
    for col, default in _DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    # ── Mirror training pipeline ───────────────────────────────
    df = clean_data(df)
    df = feature_engineering(df)

    # Drop target if accidentally included
    if "Weekly_Sales" in df.columns:
        df = df.drop(columns=["Weekly_Sales"])

    # Apply scaler to the same columns used in training
    scale_cols_present = [c for c in _SCALE_COLS if c in df.columns]
    if scale_cols_present:
        df[scale_cols_present] = _SCALER.transform(df[scale_cols_present])

    # Align columns — fill missing with 0, drop extras
    df = df.reindex(columns=_FEATURE_COLUMNS, fill_value=0)

    pred = float(max(0, _MODEL.predict(df)[0]))
    mae  = 1017.46   # replace with loaded MAE from model_results.json if preferred

    return {
        "prediction":    round(pred, 2),
        "low_estimate":  round(max(0, pred - mae), 2),
        "high_estimate": round(pred + mae, 2),
        "model_used":    type(_MODEL).__name__,
    }


# ── Batch prediction ──────────────────────────────────────────────────────────

def predict_batch(records: list[dict]) -> list[dict]:
    """
    Predict for a list of input dicts. Returns a list of result dicts.
    """
    return [predict_sales(r) for r in records]


# ── Rolling forecast ──────────────────────────────────────────────────────────

def rolling_forecast(base_input: dict, n_weeks: int = 12) -> list[dict]:
    """
    Generate an autoregressive rolling forecast for n_weeks ahead.
    Each prediction becomes the lag_1_week for the next iteration.

    Parameters
    ----------
    base_input : dict  — starting conditions (same schema as predict_sales)
    n_weeks    : int   — number of weeks to forecast

    Returns
    -------
    list of dicts: [{week, prediction, low_estimate, high_estimate}, ...]
    """
    if not _ARTEFACTS_LOADED:
        raise RuntimeError("Model not trained yet. Run main.py first.")

    results   = []
    current   = dict(base_input)
    lag1      = float(base_input.get("lag_1_week", base_input.get("Weekly_Sales", 20000)))
    lag4      = float(base_input.get("lag_4_week", lag1 * 0.95))
    lag52     = float(base_input.get("lag_52_week", lag1 * 0.98))
    rolling   = float(base_input.get("rolling_mean_4", lag1 * 0.97))
    mae       = 1017.46

    for week in range(1, n_weeks + 1):
        current["lag_1_week"]    = lag1
        current["lag_4_week"]    = lag4
        current["lag_52_week"]   = lag52
        current["rolling_mean_4"] = rolling
        current["rolling_std_4"] = abs(lag1 - rolling) * 0.5

        # Advance week number
        current_week = int(current.get("Week", 1))
        current["Week"] = (current_week % 52) + 1

        try:
            res = predict_sales(current)
            pred = res["prediction"]
        except Exception as e:
            logger.error("Forecast week %d failed: %s", week, e)
            pred = lag1   # fallback

        results.append({
            "week":          week,
            "prediction":    round(pred, 2),
            "low_estimate":  round(max(0, pred - mae), 2),
            "high_estimate": round(pred + mae, 2),
        })

        # Roll forward lag features
        lag4    = lag1
        lag1    = pred
        rolling = rolling * 0.75 + pred * 0.25

    return results


# ── Business Insights Engine ─────────────────────────────────────────────────

def generate_insights(prediction: float, input_data: dict, forecast: list[dict]) -> dict:
    """
    Generate business insights, staffing recommendations, and action items
    from a single prediction + rolling forecast.
    """
    lag1    = float(input_data.get("lag_1_week",    prediction))
    lag4    = float(input_data.get("lag_4_week",    prediction))
    lag52   = float(input_data.get("lag_52_week",   prediction))
    rolling = float(input_data.get("rolling_mean_4", prediction))
    holiday = int(input_data.get("IsHoliday", 0))
    is_holiday = bool(holiday)

    try:
        from datetime import datetime
        date_str = input_data.get("Date", "")
        month = datetime.strptime(date_str, "%Y-%m-%d").month if date_str else 6
    except Exception:
        month = 6

    is_peak_season  = month in [11, 12]
    is_slow_season  = month in [1, 2]

    # Trend signals
    wow_pct  = ((prediction - lag1)  / lag1  * 100) if lag1  > 0 else 0
    yoy_pct  = ((prediction - lag52) / lag52 * 100) if lag52 > 0 else 0
    vs_roll  = ((prediction - rolling) / rolling * 100) if rolling > 0 else 0

    # Forecast trend — is it going up or down over next 4 weeks?
    if len(forecast) >= 4:
        f_start = forecast[0]["prediction"]
        f_end   = forecast[3]["prediction"]
        forecast_trend_pct = ((f_end - f_start) / f_start * 100) if f_start > 0 else 0
    else:
        forecast_trend_pct = 0

    # Peak forecast week
    peak_week = max(forecast, key=lambda x: x["prediction"]) if forecast else None

    insights   = []   # {icon, title, detail, type: info|success|warning|danger}
    actions    = []   # {priority, action, reason}
    staffing   = {}   # {decision, level, reason, details}

    # ── Sales trend insight ───────────────────────────────────────────────────
    if wow_pct >= 10:
        insights.append({"icon": "📈", "title": "Strong Week-on-Week Growth",
            "detail": f"Sales up {wow_pct:.1f}% vs last week — momentum is building.",
            "type": "success"})
    elif wow_pct <= -10:
        insights.append({"icon": "📉", "title": "Week-on-Week Decline",
            "detail": f"Sales down {abs(wow_pct):.1f}% vs last week — monitor closely.",
            "type": "warning"})
    else:
        insights.append({"icon": "➡️", "title": "Stable Week-on-Week",
            "detail": f"Sales {'+' if wow_pct >= 0 else ''}{wow_pct:.1f}% vs last week — demand is steady.",
            "type": "info"})

    # ── Year-on-year insight ──────────────────────────────────────────────────
    if yoy_pct >= 5:
        insights.append({"icon": "🏆", "title": "Year-on-Year Growth",
            "detail": f"Up {yoy_pct:.1f}% vs same week last year — strong annual performance.",
            "type": "success"})
    elif yoy_pct <= -5:
        insights.append({"icon": "⚠️", "title": "Year-on-Year Decline",
            "detail": f"Down {abs(yoy_pct):.1f}% vs same week last year — investigate root cause.",
            "type": "danger"})

    # ── Holiday insight ───────────────────────────────────────────────────────
    if is_holiday:
        insights.append({"icon": "🎉", "title": "Holiday Week Detected",
            "detail": "Holiday weeks historically drive 10–15% higher sales. Ensure full stock and staff.",
            "type": "success"})

    # ── Seasonal insight ──────────────────────────────────────────────────────
    if is_peak_season:
        insights.append({"icon": "🛍️", "title": "Peak Season (Nov–Dec)",
            "detail": "You are in the highest-demand period of the year. Maximise inventory and promotions.",
            "type": "success"})
    elif is_slow_season:
        insights.append({"icon": "🌨️", "title": "Slow Season (Jan–Feb)",
            "detail": "Post-holiday slowdown expected. Focus on clearance sales and cost reduction.",
            "type": "info"})

    # ── Forecast trend insight ────────────────────────────────────────────────
    if forecast_trend_pct >= 8:
        insights.append({"icon": "🚀", "title": "Forecast: Demand Rising",
            "detail": f"Next 4 weeks show +{forecast_trend_pct:.1f}% growth trend. Prepare supply chain now.",
            "type": "success"})
    elif forecast_trend_pct <= -8:
        insights.append({"icon": "🔻", "title": "Forecast: Demand Falling",
            "detail": f"Next 4 weeks show {forecast_trend_pct:.1f}% decline. Consider promotions to stimulate demand.",
            "type": "warning"})

    if peak_week and peak_week["week"] > 1:
        insights.append({"icon": "📅", "title": f"Peak Demand at Week {peak_week['week']}",
            "detail": f"Highest forecasted sales of ${peak_week['prediction']:,.0f} expected at week {peak_week['week']}. Plan ahead.",
            "type": "info"})

    # ── Action items ──────────────────────────────────────────────────────────
    if vs_roll >= 15:
        actions.append({"priority": "HIGH", "action": "Increase inventory orders immediately",
            "reason": f"Predicted sales {vs_roll:.0f}% above 4-week average — risk of stockout."})
    elif vs_roll <= -15:
        actions.append({"priority": "MEDIUM", "action": "Reduce next inventory order",
            "reason": f"Predicted sales {abs(vs_roll):.0f}% below 4-week average — avoid overstock."})
    else:
        actions.append({"priority": "LOW", "action": "Maintain current inventory levels",
            "reason": "Demand is within normal range of recent average."})

    if is_holiday or is_peak_season:
        actions.append({"priority": "HIGH", "action": "Run targeted promotions & markdowns",
            "reason": "Holiday/peak season is the highest ROI window for promotions."})

    if yoy_pct <= -10:
        actions.append({"priority": "HIGH", "action": "Investigate pricing & competitor activity",
            "reason": f"Sales down {abs(yoy_pct):.1f}% year-on-year — structural decline signal."})

    if forecast_trend_pct >= 8:
        actions.append({"priority": "MEDIUM", "action": "Accelerate supplier orders for next 4 weeks",
            "reason": "Forecast shows rising demand — lead time planning needed now."})

    if is_slow_season:
        actions.append({"priority": "MEDIUM", "action": "Launch clearance / discount campaign",
            "reason": "Slow season — move excess inventory before spring restocking."})

    # ── Staffing recommendation ───────────────────────────────────────────────
    if is_holiday or is_peak_season:
        staffing = {
            "decision": "HIRE",
            "level": "high",
            "badge": "Hire Temporary Staff",
            "reason": "Holiday/peak season demand requires additional floor and checkout staff.",
            "details": [
                "Schedule maximum available part-time staff",
                "Hire seasonal/temporary workers for 4–8 weeks",
                "Extend store hours if applicable",
                "Brief all staff on holiday promotions and stock locations",
            ]
        }
    elif vs_roll >= 20:
        staffing = {
            "decision": "HIRE",
            "level": "medium",
            "badge": "Add Part-Time Staff",
            "reason": f"Predicted demand is {vs_roll:.0f}% above recent average — current staff may be insufficient.",
            "details": [
                "Add 1–2 part-time shifts for the week",
                "Ensure stock replenishment team is at full capacity",
                "Consider overtime for key departments",
            ]
        }
    elif vs_roll <= -20 and not is_holiday:
        staffing = {
            "decision": "REDUCE",
            "level": "medium",
            "badge": "Reduce Staff Hours",
            "reason": f"Predicted demand is {abs(vs_roll):.0f}% below recent average — overstaffing risk.",
            "details": [
                "Reduce part-time shifts for the week",
                "Reassign staff to restocking or training",
                "Do not renew expiring temporary contracts this week",
                "Review if any permanent roles are underutilised",
            ]
        }
    elif yoy_pct <= -15:
        staffing = {
            "decision": "REVIEW",
            "level": "high",
            "badge": "Review Workforce Size",
            "reason": f"Sales down {abs(yoy_pct):.1f}% year-on-year — sustained decline may require restructuring.",
            "details": [
                "Review permanent headcount vs sales volume",
                "Consider not replacing departing staff",
                "Identify roles that can be consolidated",
                "Consult HR before any permanent reductions",
            ]
        }
    else:
        staffing = {
            "decision": "MAINTAIN",
            "level": "low",
            "badge": "Maintain Current Staff",
            "reason": "Demand is stable — no staffing changes recommended this week.",
            "details": [
                "Keep current shift schedule as planned",
                "Ensure no unplanned absences go uncovered",
                "Standard operations — no action required",
            ]
        }

    return {
        "insights":          insights,
        "actions":           actions,
        "staffing":          staffing,
        "wow_pct":           round(wow_pct, 1),
        "yoy_pct":           round(yoy_pct, 1),
        "vs_rolling_pct":    round(vs_roll, 1),
        "forecast_trend_pct": round(forecast_trend_pct, 1),
        "is_holiday":        is_holiday,
        "is_peak_season":    is_peak_season,
        "is_slow_season":    is_slow_season,
        "peak_week":         peak_week,
    }


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "Store": 1, "Dept": 1,
        "Date": "2013-12-20",
        "IsHoliday": 1,
        "Temperature": 40.0,
        "Fuel_Price": 3.2,
        "CPI": 211.0,
        "Unemployment": 7.0,
        "MarkDown1": 5000, "MarkDown2": 0, "MarkDown3": 0,
        "MarkDown4": 0,    "MarkDown5": 0,
        "Size": 151315,    "Type": "A",
    }

    result = predict_sales(sample)
    print("\n=== PREDICTION RESULT ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n=== 4-WEEK FORECAST ===")
    forecast = rolling_forecast(sample, n_weeks=4)
    for f in forecast:
        print(f"  Week {f['week']}: ${f['prediction']:,.2f}")