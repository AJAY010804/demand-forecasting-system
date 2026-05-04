"""
web_app/routes.py
"""

import os
import io
import json
import logging

import pandas as pd
from flask import (
    render_template, request, jsonify,
    redirect, url_for, flash, send_file, send_from_directory
)

logger = logging.getLogger(__name__)

_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_HERE)
_OUTPUT_DIR = os.path.join(_ROOT, "outputs")
_MODELS_DIR = os.path.join(_ROOT, "models")
_RESULTS_DIR = os.path.join(_HERE, "results")

ALLOWED_EXTENSIONS = {"csv"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _load_model_results():
    path = os.path.join(_OUTPUT_DIR, "model_results.json")
    if not os.path.exists(path):
        return {"available": False, "models": {}, "best_model": None}
    with open(path) as f:
        results = json.load(f)
    best_model = max(results, key=lambda x: results[x]["R2"])
    return {"available": True, "models": results, "best_model": best_model}


def _parse_input(form):
    return {
        "Store":          int(form.get("store", 1)),
        "Dept":           int(form.get("dept", 1)),
        "Date":           form.get("date", "2024-01-05"),
        "IsHoliday":      int(form.get("is_holiday", 0)),
        "Temperature":    float(form.get("temperature", 60)),
        "Fuel_Price":     float(form.get("fuel_price", 3.4)),
        "CPI":            float(form.get("cpi", 211)),
        "Unemployment":   float(form.get("unemployment", 7.0)),
        "MarkDown1":      float(form.get("markdown1", 0)),
        "MarkDown2":      float(form.get("markdown2", 0)),
        "MarkDown3":      float(form.get("markdown3", 0)),
        "MarkDown4":      float(form.get("markdown4", 0)),
        "MarkDown5":      float(form.get("markdown5", 0)),
        "Size":           int(form.get("size", 151315)),
        "Type":           form.get("store_type", "A"),
        "lag_1_week":     float(form.get("lag1", 20000)),
        "lag_4_week":     float(form.get("lag4", 18000)),
        "lag_52_week":    float(form.get("lag52", 19000)),
        "rolling_mean_4": float(form.get("rolling", 19500)),
    }


def register_routes(app) -> None:

    # ── Index ─────────────────────────────────────────────────────────────────
    @app.route("/")
    def index():
        data = _load_model_results()
        return render_template("index.html",
                               trained=data["available"],
                               results=data["models"],
                               best=data["best_model"])

    # ── Dashboard ─────────────────────────────────────────────────────────────
    @app.route("/dashboard")
    def dashboard():
        data = _load_model_results()
        return render_template("dashboard.html",
                               model_results=data["models"],
                               best_model=data["best_model"])

    # ── Predict ───────────────────────────────────────────────────────────────
    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        if request.method == "GET":
            return render_template("predict.html")
        try:
            from src.prediction import predict_sales, rolling_forecast, generate_insights
            input_data = _parse_input(request.form)
            result     = predict_sales(input_data)
            fc         = rolling_forecast(input_data, n_weeks=12)
            insights   = generate_insights(result["prediction"], input_data, fc)
            return render_template("result.html",
                                   result=result, input_data=input_data,
                                   forecast=fc, insights=insights)
        except RuntimeError as e:
            flash(str(e), "error")
            return redirect(url_for("predict"))
        except Exception as e:
            logger.exception("Prediction error")
            flash(f"Prediction failed: {e}", "error")
            return redirect(url_for("predict"))

    # ── Forecast ──────────────────────────────────────────────────────────────
    @app.route("/forecast", methods=["GET", "POST"])
    def forecast():
        if request.method == "GET":
            return render_template("forecast.html")
        try:
            from src.prediction import predict_sales, rolling_forecast, generate_insights
            input_data = _parse_input(request.form)
            n_weeks    = int(request.form.get("n_weeks", 12))
            result     = predict_sales(input_data)
            fc         = rolling_forecast(input_data, n_weeks=n_weeks)
            insights   = generate_insights(result["prediction"], input_data, fc)
            return render_template("forecast.html",
                                   result=result, input_data=input_data,
                                   forecast=fc, insights=insights,
                                   n_weeks=n_weeks)
        except RuntimeError as e:
            flash(str(e), "error")
            return redirect(url_for("forecast"))
        except Exception as e:
            logger.exception("Forecast error")
            flash(f"Forecast failed: {e}", "error")
            return redirect(url_for("forecast"))

    # ── Upload (batch predict) ────────────────────────────────────────────────
    @app.route("/upload", methods=["GET", "POST"])
    def upload():
        if request.method == "GET":
            return render_template("upload.html")

        if "file" not in request.files:
            flash("No file selected.", "error")
            return redirect(url_for("upload"))

        file = request.files["file"]
        if file.filename == "" or not _allowed_file(file.filename):
            flash("Please upload a valid CSV file.", "error")
            return redirect(url_for("upload"))

        try:
            from src.prediction import predict_batch

            df = pd.read_csv(file)
            if df.empty:
                flash("Uploaded CSV is empty.", "error")
                return redirect(url_for("upload"))

            predictions = predict_batch(df.to_dict(orient="records"))
            df["Predicted_Sales"] = [p["prediction"]   for p in predictions]
            df["Low_Estimate"]    = [p["low_estimate"]  for p in predictions]
            df["High_Estimate"]   = [p["high_estimate"] for p in predictions]

            # Save to disk for download
            os.makedirs(_RESULTS_DIR, exist_ok=True)
            result_path = os.path.join(_RESULTS_DIR, "demand_predictions.csv")
            df.to_csv(result_path, index=False)

            # Preview: key columns, first 100 rows
            preview_cols = [c for c in ["Store", "Dept", "Date", "IsHoliday",
                                        "Predicted_Sales", "Low_Estimate", "High_Estimate"]
                            if c in df.columns]
            preview = df[preview_cols].head(100).to_dict(orient="records")

            # Summary stats
            summary = {
                "total_rows":  len(df),
                "avg_pred":    round(df["Predicted_Sales"].mean(), 2),
                "max_pred":    round(df["Predicted_Sales"].max(), 2),
                "min_pred":    round(df["Predicted_Sales"].min(), 2),
                "total_pred":  round(df["Predicted_Sales"].sum(), 2),
            }

            return render_template("upload.html",
                                   preview=preview,
                                   preview_cols=preview_cols,
                                   summary=summary,
                                   filename=file.filename)

        except RuntimeError as e:
            flash(str(e), "error")
            return redirect(url_for("upload"))
        except Exception as e:
            logger.exception("Batch prediction error")
            flash(f"Batch prediction failed: {e}", "error")
            return redirect(url_for("upload"))

    # ── Download batch results ────────────────────────────────────────────────
    @app.route("/download-results")
    def download_results():
        result_path = os.path.join(_RESULTS_DIR, "demand_predictions.csv")
        if not os.path.exists(result_path):
            flash("No results file found. Please upload a CSV first.", "error")
            return redirect(url_for("upload"))
        return send_file(result_path, mimetype="text/csv",
                         as_attachment=True, download_name="demand_predictions.csv")

    # ── API: single predict ───────────────────────────────────────────────────
    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        try:
            from src.prediction import predict_sales
            result = predict_sales(request.get_json(force=True))
            return jsonify({"status": "ok", "result": result})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400

    # ── API: rolling forecast ─────────────────────────────────────────────────
    @app.route("/api/forecast", methods=["POST"])
    def api_forecast():
        try:
            from src.prediction import rolling_forecast
            data    = request.get_json(force=True)
            n_weeks = int(data.pop("n_weeks", 12))
            result  = rolling_forecast(data, n_weeks=n_weeks)
            return jsonify({"status": "ok", "forecast": result, "weeks": n_weeks})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400

    # ── API: model results ────────────────────────────────────────────────────
    @app.route("/api/model-results")
    def api_model_results():
        return jsonify(_load_model_results())

    # ── Models ────────────────────────────────────────────────────────────────
    @app.route("/models")
    def models():
        data = _load_model_results()
        return render_template("models.html",
                               model_results=data["models"],
                               best_model=data["best_model"])

    # ── Health ────────────────────────────────────────────────────────────────
    @app.route("/health")
    def health():
        model_ready = os.path.exists(os.path.join(_MODELS_DIR, "best_model.pkl"))
        return jsonify({"status": "ok", "model_ready": model_ready})

    # ── Serve outputs ─────────────────────────────────────────────────────────
    @app.route("/outputs/<path:filename>")
    def serve_outputs(filename):
        return send_from_directory(_OUTPUT_DIR, filename)
