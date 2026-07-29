# 🌍 Air Quality Index (AQI) Prediction & Forecasting System

An end-to-end Machine Learning API for real-time Air Quality Index (AQI) prediction, 24-hour hourly forecasting, and 7-day daily forecasting. Built with **FastAPI**, **XGBoost**, **SQLAlchemy**, **MySQL**, **Nginx**, and **Docker/SSL** support.

---

## 📁 Modular Directory Architecture

The repository is organized into distinct, modular functional folders under the root directory:

```
project-final/
├── backend/                        # FastAPI Application & API Schemas
│   ├── app.py                      # Main API Routes (/predict, /24hours, /7days, /history)
│   ├── schemas.py                  # Pydantic validation schemas
│   └── requirements.txt            # Python dependencies
│
├── database/                       # MySQL Database Layer
│   ├── database.py                 # SQLAlchemy MySQL connection engine
│   ├── models.py                   # MySQL prediction_history table model
│   ├── check_mysql_status.py       # MySQL connection status checker
│   ├── test_mysql.py               # MySQL authentication test
│   ├── .env                        # Active environment credentials (root/indrajit)
│   └── .env.example                # Example environment template
│
├── ml_model/                       # Trained Machine Learning Model Assets
│   ├── xgb_model.joblib            # Native Trained XGBoost AQI Model
│   ├── preprocessor.joblib         # Scikit-learn Data Preprocessor
│   ├── feature_info.joblib         # Feature metadata
│   ├── feature_names.joblib        # Feature column names
│   ├── aqi_tensorflow_model.h5     # TensorFlow Keras Model asset
│   └── aqi_model.tflite            # TensorFlow Lite Model asset
│
├── ml_scripts/                     # Model Training, EDA & Evaluation Scripts
│   ├── ML.py                       # XGBoost training pipeline script
│   ├── Predict.py                  # Prediction logic helper script
│   ├── calculate_epa.py            # EPA AQI calculation reference script
│   ├── data_visualization.py       # Matplotlib visualization script
│   ├── feature_importance.py       # Feature importance extraction script
│   ├── EDA.ipynb                   # Exploratory Data Analysis Jupyter Notebook
│   ├── test.py                     # Native model prediction test script
│   ├── Output_Bucket.xlsx          # Dataset Excel file
│   └── *.png                       # Model evaluation charts & graphs
│
├── nginx/                          # Nginx SSL Reverse Proxy & Certificates
│   ├── nginx.conf                  # Nginx SSL Reverse Proxy configuration
│   ├── cert.pem / key.pem          # Local SSL Certificates
│   └── generate_cert.py            # SSL Certificate generator script
│
├── utils/                          # Helper Tools & Live Data Fetchers
│   └── fetch_live_pollutants.py    # Auto-fetch live pollutant API script
│
├── LICENSE                         # Project License
└── README.md                       # Documentation
```

---

## ⚙️ Prerequisites

- **Python**: 3.10+
- **MySQL Server**: 8.0+ / MySQL Server 26.7

---

## 🚀 How to Run the Application

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/SHAW258/project-final.git
cd project-final
pip install -r backend/requirements.txt
```

### 2. Configure Database Credentials

The MySQL database credentials are configured in `database/.env`:

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=indrajit
DB_NAME=aqi_db
```

### 3. Start the FastAPI Application

From the root project directory:

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc UI**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 📋 API Endpoints Reference

| Method | Endpoint | Description | Sample Payload |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Server, ML model & MySQL health check | None |
| `GET` | `/info` | API details & active database info | None |
| `POST` | `/predict` | Predict current AQI & generate dashboard chart | `{"pollutants": {"PM2.5": 36, "PM10": 64, "NO2": 8, "SO2": 2, "CO": 0.3, "O3": 0.01}}` |
| `POST` | `/24hours` | Generate 24-hour hourly AQI forecast | `{"pollutants": {"PM2.5": 36, "PM10": 64, "NO2": 8, "SO2": 2, "CO": 0.3, "O3": 0.01}}` |
| `POST` | `/7days` | Generate 7-day daily AQI forecast | `{"pollutants": {"PM2.5": 36, "PM10": 64, "NO2": 8, "SO2": 2, "CO": 0.3, "O3": 0.01}}` |
| `GET` | `/history` | Fetch stored prediction records from MySQL | Query params: `?limit=20&skip=0` |

---

## 🔒 Nginx Reverse Proxy Setup

To run Nginx reverse proxy forwarding HTTPS (port 8443) -> FastAPI (port 8000):

```bash
nginx -c C:/Users/indrajit/Desktop/project-final/nginx/nginx.conf
```
