# 📦 Demand Forecasting System

> ML-powered weekly retail sales forecasting with business insights, staffing recommendations, and a full-stack Flask web application.

---

## 🚀 Live Demo

> Deployed on Render → [https://demand-forecasting-system.onrender.com](https://demand-forecasting-system.onrender.com)

---

## 📌 Problem Statement

Retail businesses struggle to accurately predict future demand, leading to:
- **Overstocking** — wasted capital and storage costs
- **Understocking** — lost sales and poor customer experience
- **Poor staffing decisions** — too many or too few workers during peak/slow periods

This system solves that by predicting **weekly sales** for any store and department combination using historical data, and then generating **actionable business insights** automatically.

---

## 🎯 Features

| Feature | Description |
|---|---|
| 🔮 Single Prediction | Predict weekly sales for any store/dept with instant result |
| 📈 Multi-Week Forecast | Rolling 4–26 week future forecast with trend chart |
| 📊 Business Insights | Auto-generated insights — holiday effect, YoY trend, seasonal signals |
| 👥 Staffing Recommendations | HIRE / REDUCE / MAINTAIN / REVIEW decisions with action steps |
| ⚡ Action Items | Prioritised HIGH/MEDIUM/LOW inventory and promotion actions |
| 📂 Batch Upload | Upload CSV → get predictions for all rows → download results |
| 📉 Model Dashboard | Compare all models with R², MAE, RMSE charts |
| 🖼️ EDA Charts | 6 auto-generated exploratory analysis charts |

---

## 🧠 ML Pipeline

```
Raw Data (train.csv + features.csv + stores.csv)
        ↓
    Load & Merge
        ↓
    Data Cleaning
    (nulls, duplicates, outliers, type encoding)
        ↓
    Feature Engineering
    (lag 1/4/52 week, rolling mean/std, time features, one-hot encoding)
        ↓
    Preprocessing
    (MinMaxScaler, temporal train/test split 80/20)
        ↓
    Train 3 Models
    (Linear Regression, Random Forest, XGBoost)
        ↓
    Weighted Ensemble
    (weights proportional to R² score)
        ↓
    Best Model Selection → Save Artefacts
        ↓
    Flask Web App → Predict + Insights
```

---

## 📊 Model Results

| Model | R² Score | MAE | RMSE | Train Time |
|---|---|---|---|---|
| Linear Regression | 0.9801 | $1,219 | $2,634 | 0.3s |
| Random Forest | 0.9855 | $1,037 | $2,251 | 173s |
| **XGBoost** ⭐ | **0.9872** | **$1,003** | **$2,115** | 22s |
| Ensemble | 0.9859 | $1,018 | $2,220 | — |

> **XGBoost** selected as best model with **R² = 0.9872** on 52K test rows.

---

## 📁 Project Structure

```
demand-forecasting-system/
│
├── data/
│   ├── raw/                    # train.csv, features.csv, stores.csv
│   └── processed/              # cleaned data
│
├── models/                     # Saved .pkl model files
│   ├── best_model.pkl
│   ├── xgboost.pkl
│   ├── random_forest.pkl
│   ├── linear_regression.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── outputs/                    # EDA charts + model_results.json
│
│
├── src/                        # ML pipeline modules
│   ├── load_data.py
│   ├── data_processing.py
│   ├── eda.py
│   ├── model_training.py
│   └── prediction.py
│
├── web_app/                    # Flask web application
│   ├── __init__.py
│   ├── routes.py
│   ├── templates/
│   └── static/
│
├── main.py                     # Run full training pipeline
├── run.py                      # Run Flask development server
├── wsgi.py                     # Gunicorn entry point (Render)
├── Procfile                    # Render deployment config
└── requirements.txt
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/demand-forecasting-system.git
cd demand-forecasting-system
```

### 2. Create conda environment
```bash
conda create -n demand-forecasting python=3.11
conda activate demand-forecasting
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add raw data
Download from [Kaggle — Walmart Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data) and place in `data/raw/`:
- `train.csv`
- `features.csv`
- `stores.csv`

### 5. Train the models
```bash
python main.py
```

### 6. Run the web app
```bash
python run.py
```

Open → [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🌐 Deployment (Render)

1. Push to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect GitHub repo
4. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn wsgi:app`
5. Add environment variable: `SECRET_KEY = your-secret-key`
6. Deploy ✅

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Models | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Web Framework | Flask |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Deployment | Render (Gunicorn) |
| Version Control | Git + GitHub |

---

## 📦 Dataset

- **Source**: [Walmart Store Sales Forecasting — Kaggle](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
- **Size**: 421,570 records
- **Period**: Feb 2010 — Oct 2012
- **Stores**: 45 Walmart stores
- **Departments**: 99 departments per store
- **Features**: Store size, type, temperature, fuel price, CPI, unemployment, markdowns, holiday flag

---

## 👨‍💻 Author

**Ajay**
Final Year — Data Science Minor Degree
[MIT CSN]

---

## 📄 License

This project is for academic purposes only.
