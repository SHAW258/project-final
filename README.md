# 🌍 Air Quality Index (AQI) Prediction & Forecasting System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat-square&logo=python)](https://www.python.org/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0%2B-4479A1.svg?style=flat-square&logo=mysql)](https://www.mysql.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-152935.svg?style=flat-square)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

An enterprise-grade Machine Learning API for real-time **Air Quality Index (AQI)** prediction, 24-hour hourly forecasting, and 7-day trend analysis. Converted from Flask to **FastAPI**, integrated with **MySQL database logging**, **XGBoost machine learning**, **Nginx SSL reverse proxying**, and high-resolution chart generation.

---

## 📐 System Architecture

```
                               ┌───────────────────────────┐
                               │     Client / Browser      │
                               └─────────────┬─────────────┘
                                             │ HTTPS (8443 / 443)
                                             ▼
                               ┌───────────────────────────┐
                               │    Nginx Reverse Proxy    │
                               │   (SSL/TLS Termination)   │
                               └─────────────┬─────────────┘
                                             │ HTTP (8000)
                                             ▼
                               ┌───────────────────────────┐
                               │   FastAPI Uvicorn App     │
                               │    (backend/app.py)       │
                               └──────┬─────────────┬──────┘
                                      │             │
                    ┌─────────────────┘             └─────────────────┐
                    ▼                                                 ▼
     ┌─────────────────────────────┐                   ┌─────────────────────────────┐
     │   XGBoost ML Pipeline       │                   │    MySQL Database Engine    │
     │   (ml_model/xgb_model)      │                   │    (database/database.py)   │
     └─────────────────────────────┘                   └─────────────────────────────┘
```

---

## 🌟 Key Features

- ⚡ **High Performance FastAPI Backend**: Asynchronous OpenAPI 3.1 engine with automated interactive Swagger UI (`/docs`).
- 🤖 **Native Machine Learning Model**: Uses a trained **XGBoost Regressor** with scikit-learn preprocessing pipelines to accurately predict AQI based on 6 core pollutants (`PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `O3`).
- 🔮 **Multi-Horizon Forecasting**: Real-time AQI prediction, 24-hour hourly forecasting with trend statistics, and 7-day daily predictions.
- 📊 **Dynamic Dashboard Visualizations**: High-resolution Matplotlib Base64 chart generation embedded directly into API JSON responses.
- 🛢️ **MySQL Persistence**: Automatically logs every prediction, timestamp, pollutant value, and health recommendation into MySQL (`aqi_db.prediction_history`).
- 🔒 **Nginx & SSL Support**: Complete Nginx reverse proxy configuration (`nginx.conf`) with SSL certificate support for secure HTTPS deployments.
- 🌐 **Global Access Ready**: Integrated tunnel support (`localtunnel` / `cloudflared`) to expose the local server globally across networks.

---

## 📁 Repository Structure

```
project-final/
├── 📁 backend/                    # Core FastAPI Application & Pydantic Schemas
│   ├── app.py                     # API Routes (/predict, /24hours, /7days, /history)
│   ├── schemas.py                 # Pydantic request/response data contracts
│   ├── run_https.py               # Direct HTTPS Uvicorn launcher
│   ├── test_fastapi.py            # Automated API integration test suite
│   └── requirements.txt           # Python dependency manifests
│
├── 📁 database/                   # Database Layer & MySQL Persistence
│   ├── database.py                # SQLAlchemy MySQL connection pool & fallback logic
│   ├── models.py                  # SQLAlchemy prediction_history table model
│   ├── check_mysql_status.py      # Database connection verification tool
│   ├── test_mysql.py              # PyMySQL authentication unit test
│   ├── .env                       # Local environment credentials
│   └── .env.example               # Environment template
│
├── 📁 ml_model/                   # Machine Learning Model Assets
│   ├── xgb_model.joblib           # Trained XGBoost AQI Regressor model
│   ├── preprocessor.joblib        # Scikit-learn feature preprocessor
│   ├── feature_info.joblib        # Feature metadata
│   ├── feature_names.joblib       # Feature column names
│   ├── aqi_tensorflow_model.h5    # Keras Neural Network asset
│   └── aqi_model.tflite           # TensorFlow Lite model asset
│
├── 📁 ml_scripts/                 # Model Training, EDA & Evaluation Pipeline
│   ├── ML.py                      # Model training & serialization script
│   ├── Predict.py                 # Standalone prediction test script
│   ├── calculate_epa.py           # EPA standard AQI calculation utility
│   ├── data_visualization.py      # Plotting pipeline
│   ├── feature_importance.py      # Feature importance extraction
│   ├── EDA.ipynb                  # Exploratory Data Analysis notebook
│   ├── Output_Bucket.xlsx         # Dataset spreadsheet
│   └── *.png                      # High-resolution model evaluation graphs
│
├── 📁 nginx/                      # Production Reverse Proxy & Security
│   ├── nginx.conf                 # Nginx SSL reverse proxy configuration
│   ├── cert.pem / key.pem         # X.509 SSL Certificate & Private Key
│   └── generate_cert.py           # SSL certificate generation tool
│
├── 📁 utils/                      # Utilities & Live Data Integration
│   └── fetch_live_pollutants.py   # Live air quality API fetcher
│
├── SETUP.md                       # Comprehensive Deployment & Installation Guide
├── LICENSE                        # MIT License
└── README.md                      # Project Documentation
```

---

## ⚡ Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/SHAW258/project-final.git
cd project-final

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Configure Database Credentials

Ensure MySQL is running on `localhost:3306`, then update credentials in `database/.env`:

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=indrajit
DB_NAME=aqi_db
```

### 3. Launch Server

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Endpoint**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

For full installation options, SSL configuration, Nginx setup, and global tunneling, refer to **[SETUP.md](SETUP.md)**.

---

## 📋 API Endpoints Reference

| Method | Endpoint | Description | Request Body Example |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | System, ML Model & MySQL Connection Health | *None* |
| `GET` | `/info` | API Version & Database Engine Details | *None* |
| `POST` | `/predict` | Current AQI Prediction + Base64 Gauge Chart | `{"pollutants": {"PM2.5": 36, "PM10": 64, "NO2": 8, "SO2": 2, "CO": 0.3, "O3": 0.01}}` |
| `POST` | `/24hours` | 24-Hour Hourly Forecast + Base64 Trend Chart | `{"pollutants": {"PM2.5": 36, "PM10": 64, "NO2": 8, "SO2": 2, "CO": 0.3, "O3": 0.01}}` |
| `POST` | `/7days` | 7-Day Forecast + Base64 Weekly Comparison Chart | `{"pollutants": {"PM2.5": 36, "PM10": 64, "NO2": 8, "SO2": 2, "CO": 0.3, "O3": 0.01}}` |
| `GET` | `/history` | Query Stored MySQL Prediction Records | Query params: `?limit=20&skip=0` |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
