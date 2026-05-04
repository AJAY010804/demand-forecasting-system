"""
src/insights.py
---------------
Business insights engine.
Generates actionable recommendations from prediction + forecast data.
No ML — pure rule-based logic on top of model outputs.
"""

from datetime import datetime, timedelta


# ── Seasonal helpers ──────────────────────────────────────────────────────────

_PEAK_MONTHS    = {11, 12}          # Nov, Dec
_HOLIDAY_WEEKS  = {47, 48, 49, 50, 51, 52, 1}   # Thanksgiving → New Year


def _month_from_date(date_str: str) -> int:
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").month
    except Exception:
        return datetime.now().month


def _week_from_date(date_str: str) -> int:
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").isocalendar()[1]
    except Exception:
        return datetime.now().isocalendar()[1]


# ── Single-prediction insights ────────────────────────────────────────────────

def generate_insights(input_data: dict, result: dict) -> dict:
    """
    Generate business insights for a single prediction.

    Returns
    -------
    dict with keys:
        summary        — one-line headline
        demand_signal  — 'high' | 'normal' | 'low'
        insights       — list of {icon, title, detail, level}
        inventory      — {action, reason, level}
        staffing       — {action, reason, level, change_pct}
        sales_signals  — list of {label, value, change, direction}
    """
    pred      = result["prediction"]
    rolling   = float(input_data.get("rolling_mean_4", pred))
    lag1      = float(input_data.get("lag_1_week",     pred))
    lag52     = float(input_data.get("lag_52_week",    pred))
    is_holiday= int(input_data.get("IsHoliday", 0))
    date_str  = str(input_data.get("Date", ""))
    month     = _month_from_date(date_str)
    week      = _week_from_date(date_str)

    # Demand gap vs rolling average
    gap_pct   = ((pred - rolling) / rolling * 100) if rolling > 0 else 0
    yoy_pct   = ((pred - lag52)   / lag52   * 100) if lag52  > 0 else 0
    wow_pct   = ((pred - lag1)    / lag1    * 100) if lag1   > 0 else 0

    is_peak   = month in _PEAK_MONTHS
    is_hol_wk = week in _HOLIDAY_WEEKS or bool(is_holiday)

    # ── Demand signal ─────────────────────────────────────────────────────────
    if gap_pct >= 15:
        demand_signal = "high"
    elif gap_pct <= -15:
        demand_signal = "low"
    else:
        demand_signal = "normal"

    # ── Insights list ─────────────────────────────────────────────────────────
    insights = []

    # Trend vs rolling
    if gap_pct >= 20:
        insights.append({"icon": "📈", "title": "Strong Demand Surge",
            "detail": f"Predicted sales are {gap_pct:+.1f}% above your 4-week average. Prepare for high traffic.",
            "level": "success"})
    elif gap_pct >= 10:
        insights.append({"icon": "↗️", "title": "Above-Average Demand",
            "detail": f"Sales expected {gap_pct:+.1f}% above rolling average. Good week ahead.",
            "level": "info"})
    elif gap_pct <= -20:
        insights.append({"icon": "📉", "title": "Significant Demand Drop",
            "detail": f"Sales predicted {gap_pct:+.1f}% below average. Consider running promotions.",
            "level": "danger"})
    elif gap_pct <= -10:
        insights.append({"icon": "↘️", "title": "Below-Average Demand",
            "detail": f"Sales expected {gap_pct:+.1f}% below rolling average. Monitor closely.",
            "level": "warning"})
    else:
        insights.append({"icon": "➡️", "title": "Stable Demand",
            "detail": f"Sales within ±10% of rolling average ({gap_pct:+.1f}%). Normal week expected.",
            "level": "info"})

    # YoY
    if yoy_pct >= 10:
        insights.append({"icon": "🏆", "title": "Year-over-Year Growth",
            "detail": f"Sales up {yoy_pct:+.1f}% vs same week last year. Strong growth trajectory.",
            "level": "success"})
    elif yoy_pct <= -10:
        insights.append({"icon": "⚠️", "title": "Year-over-Year Decline",
            "detail": f"Sales down {abs(yoy_pct):.1f}% vs same week last year. Investigate root cause.",
            "level": "warning"})

    # Holiday
    if is_hol_wk:
        insights.append({"icon": "🎄", "title": "Holiday Week Detected",
            "detail": "Holiday weeks historically drive 10–20% higher sales. Ensure full stock and staffing.",
            "level": "info"})

    # Peak season
    if is_peak:
        insights.append({"icon": "🔥", "title": "Peak Season (Nov–Dec)",
            "detail": "You are in the highest-demand period of the year. Maximise inventory and staff levels.",
            "level": "warning"})

    # Markdown opportunity
    total_markdown = sum(float(input_data.get(f"MarkDown{i}", 0)) for i in range(1, 6))
    if demand_signal == "low" and total_markdown == 0:
        insights.append({"icon": "🏷️", "title": "Consider Running Promotions",
            "detail": "Demand is below average and no markdowns are active. A targeted promotion could lift sales.",
            "level": "warning"})
    elif total_markdown > 5000:
        insights.append({"icon": "💰", "title": "Active Promotions Detected",
            "detail": f"${total_markdown:,.0f} in markdowns active. Monitor margin impact vs volume lift.",
            "level": "info"})

    # ── Inventory recommendation ───────────────────────────────────────────────
    if demand_signal == "high" or is_hol_wk or is_peak:
        inventory = {
            "action": "Stock Up",
            "reason": "High demand expected. Increase inventory by 15–25% to avoid stockouts.",
            "level":  "success",
            "icon":   "📦"
        }
    elif demand_signal == "low":
        inventory = {
            "action": "Reduce Stock Orders",
            "reason": "Low demand predicted. Avoid over-ordering — reduce replenishment by 10–20%.",
            "level":  "warning",
            "icon":   "🔻"
        }
    else:
        inventory = {
            "action": "Maintain Current Levels",
            "reason": "Demand is stable. Continue standard replenishment schedule.",
            "level":  "info",
            "icon":   "✅"
        }

    # ── Staffing recommendation ────────────────────────────────────────────────
    if gap_pct >= 20 or (is_hol_wk and is_peak):
        staffing = {
            "action":     "Hire Temporary Staff",
            "reason":     "Very high demand expected. Bring in temporary workers and extend shift hours.",
            "level":      "success",
            "icon":       "👥",
            "change_pct": "+20–30%"
        }
    elif gap_pct >= 10 or is_hol_wk or is_peak:
        staffing = {
            "action":     "Schedule Extra Shifts",
            "reason":     "Above-average demand. Add part-time shifts and ensure full floor coverage.",
            "level":      "info",
            "icon":       "🕐",
            "change_pct": "+10–20%"
        }
    elif gap_pct <= -20:
        staffing = {
            "action":     "Reduce Staff Hours",
            "reason":     "Significantly low demand. Reduce part-time hours and defer non-essential shifts.",
            "level":      "danger",
            "icon":       "📋",
            "change_pct": "-15–25%"
        }
    elif gap_pct <= -10:
        staffing = {
            "action":     "Trim Part-Time Hours",
            "reason":     "Below-average demand. Scale back part-time schedules to control labour costs.",
            "level":      "warning",
            "icon":       "⏱️",
            "change_pct": "-10–15%"
        }
    else:
        staffing = {
            "action":     "Maintain Current Workforce",
            "reason":     "Demand is stable. No staffing changes needed this week.",
            "level":      "info",
            "icon":       "👤",
            "change_pct": "0%"
        }

    # ── Sales signals summary row ──────────────────────────────────────────────
    sales_signals = [
        {"label": "vs Last Week",    "value": f"${pred:,.0f}", "change": f"{wow_pct:+.1f}%",
         "direction": "up" if wow_pct >= 0 else "down"},
        {"label": "vs 4-Wk Avg",     "value": f"${pred:,.0f}", "change": f"{gap_pct:+.1f}%",
         "direction": "up" if gap_pct >= 0 else "down"},
        {"label": "vs Last Year",    "value": f"${pred:,.0f}", "change": f"{yoy_pct:+.1f}%",
         "direction": "up" if yoy_pct >= 0 else "down"},
    ]

    # ── Summary headline ──────────────────────────────────────────────────────
    if demand_signal == "high":
        summary = f"High-demand week ahead — predicted ${pred:,.0f} ({gap_pct:+.1f}% above average)"
    elif demand_signal == "low":
        summary = f"Slow week expected — predicted ${pred:,.0f} ({gap_pct:+.1f}% below average)"
    else:
        summary = f"Stable week predicted — ${pred:,.0f} in expected sales"

    return {
        "summary":       summary,
        "demand_signal": demand_signal,
        "insights":      insights,
        "inventory":     inventory,
        "staffing":      staffing,
        "sales_signals": sales_signals,
        "gap_pct":       round(gap_pct, 1),
        "yoy_pct":       round(yoy_pct, 1),
        "wow_pct":       round(wow_pct, 1),
    }


# ── Forecast-level insights ───────────────────────────────────────────────────

def generate_forecast_insights(forecast: list[dict], base_input: dict) -> dict:
    """
    Generate insights across a multi-week rolling forecast.

    Parameters
    ----------
    forecast   : list of {week, prediction, low_estimate, high_estimate}
    base_input : original input dict

    Returns
    -------
    dict with keys:
        trend          — 'upward' | 'downward' | 'stable'
        peak_week      — week number with highest predicted sales
        low_week       — week number with lowest predicted sales
        total_revenue  — sum of all predicted sales
        avg_weekly     — average weekly prediction
        insights       — list of {icon, title, detail, level}
        staffing_plan  — list of {week, action, level} per week
        inventory_plan — list of {week, action, level} per week
    """
    if not forecast:
        return {}

    preds      = [w["prediction"] for w in forecast]
    avg        = sum(preds) / len(preds)
    rolling    = float(base_input.get("rolling_mean_4", avg))
    is_holiday = int(base_input.get("IsHoliday", 0))
    date_str   = str(base_input.get("Date", ""))
    month      = _month_from_date(date_str)

    peak_week  = max(forecast, key=lambda w: w["prediction"])
    low_week   = min(forecast, key=lambda w: w["prediction"])

    # Trend: compare first half vs second half
    mid        = len(preds) // 2
    first_avg  = sum(preds[:mid]) / mid if mid else avg
    second_avg = sum(preds[mid:]) / (len(preds) - mid) if (len(preds) - mid) else avg
    trend_pct  = ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0

    if trend_pct >= 5:
        trend = "upward"
    elif trend_pct <= -5:
        trend = "downward"
    else:
        trend = "stable"

    # ── Forecast insights ─────────────────────────────────────────────────────
    insights = []

    if trend == "upward":
        insights.append({"icon": "📈", "title": "Growing Sales Trend",
            "detail": f"Sales are projected to grow {trend_pct:+.1f}% over the forecast period. Plan for increasing demand.",
            "level": "success"})
    elif trend == "downward":
        insights.append({"icon": "📉", "title": "Declining Sales Trend",
            "detail": f"Sales are projected to decline {abs(trend_pct):.1f}% over the forecast period. Consider promotions.",
            "level": "warning"})
    else:
        insights.append({"icon": "➡️", "title": "Stable Sales Forecast",
            "detail": "Sales are expected to remain relatively stable across the forecast window.",
            "level": "info"})

    insights.append({"icon": "🏆", "title": f"Peak Sales: Week {peak_week['week']}",
        "detail": f"Highest predicted sales of ${peak_week['prediction']:,.0f}. Maximise stock and staffing for this week.",
        "level": "info"})

    insights.append({"icon": "⬇️", "title": f"Lowest Sales: Week {low_week['week']}",
        "detail": f"Lowest predicted sales of ${low_week['prediction']:,.0f}. Good week to run promotions or reduce costs.",
        "level": "warning"})

    if month in _PEAK_MONTHS:
        insights.append({"icon": "🔥", "title": "Peak Season Active",
            "detail": "Forecast covers Nov–Dec peak season. Ensure supply chain and staffing are fully prepared.",
            "level": "warning"})

    revenue_vs_avg = ((avg - rolling) / rolling * 100) if rolling > 0 else 0
    if revenue_vs_avg >= 10:
        insights.append({"icon": "💵", "title": "Above-Average Revenue Period",
            "detail": f"Forecast average ${avg:,.0f}/week is {revenue_vs_avg:+.1f}% above your baseline.",
            "level": "success"})
    elif revenue_vs_avg <= -10:
        insights.append({"icon": "💸", "title": "Below-Average Revenue Period",
            "detail": f"Forecast average ${avg:,.0f}/week is {abs(revenue_vs_avg):.1f}% below your baseline. Review strategy.",
            "level": "danger"})

    # ── Per-week staffing & inventory plans ───────────────────────────────────
    staffing_plan  = []
    inventory_plan = []

    for w in forecast:
        p       = w["prediction"]
        wk_gap  = ((p - avg) / avg * 100) if avg > 0 else 0

        if wk_gap >= 15:
            s_action, s_level = "Hire / Extra Shifts", "success"
            i_action, i_level = "Stock Up (+20%)",     "success"
        elif wk_gap >= 5:
            s_action, s_level = "Add Part-Time Shifts", "info"
            i_action, i_level = "Stock Up (+10%)",      "info"
        elif wk_gap <= -15:
            s_action, s_level = "Reduce Hours",         "danger"
            i_action, i_level = "Reduce Orders (-20%)", "danger"
        elif wk_gap <= -5:
            s_action, s_level = "Trim Part-Time",       "warning"
            i_action, i_level = "Reduce Orders (-10%)", "warning"
        else:
            s_action, s_level = "Maintain Staff",       "info"
            i_action, i_level = "Normal Replenishment", "info"

        staffing_plan.append( {"week": w["week"], "action": s_action, "level": s_level, "pred": p})
        inventory_plan.append({"week": w["week"], "action": i_action, "level": i_level, "pred": p})

    return {
        "trend":          trend,
        "trend_pct":      round(trend_pct, 1),
        "peak_week":      peak_week,
        "low_week":       low_week,
        "total_revenue":  round(sum(preds), 2),
        "avg_weekly":     round(avg, 2),
        "insights":       insights,
        "staffing_plan":  staffing_plan,
        "inventory_plan": inventory_plan,
    }
