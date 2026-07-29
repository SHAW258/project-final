# 🛠️ System Setup, Configuration & Deployment Guide

This document provides step-by-step instructions for installing, configuring, deploying, and maintaining the **Air Quality Index (AQI) Prediction & Forecasting API**.

---

## 📋 Prerequisites & Requirements

Before proceeding with installation, ensure your environment meets the following software requirements:

- **Operating System**: Windows 10/11, macOS, or Linux (Ubuntu 20.04+)
- **Python Runtime**: Python 3.10, 3.11, 3.12, 3.13, or 3.14
- **Database Engine**: MySQL Server 8.0 or higher (or MySQL 26.7)
- **Web Server / Reverse Proxy** *(Optional)*: Nginx 1.20+
- **Package Manager**: `pip` (Python) and `winget` (Windows) or `apt` (Linux)

---

## 🚀 Step 1: Environment & Dependency Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SHAW258/project-final.git
cd project-final
```

### 2. Create a Python Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies

```bash
pip install -r backend/requirements.txt
```

---

## 🛢️ Step 2: MySQL Database Setup

The API automatically creates the `aqi_db` database and `prediction_history` table upon startup if they do not exist.

### 1. Ensure MySQL Server is Running

Ensure MySQL Server is active on port `3306`:
```bash
# Windows PowerShell
Get-Service -Name MySQL*

# Linux
sudo systemctl status mysql
```

### 2. Configure Credentials in `.env`

Edit the `.env` configuration file inside `database/.env`:

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=indrajit
DB_NAME=aqi_db

DATABASE_URL=mysql+pymysql://root:indrajit@localhost:3306/aqi_db
```

> **Note**: If MySQL server connection fails or is unavailable, the system gracefully falls back to an embedded SQLite database (`database/fallback_aqi.db`) without crashing.

---

## 🏃 Step 3: Running the Development Server

### 1. Standard Uvicorn Server (HTTP)

Launch the FastAPI application with automatic reloading enabled:

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

- **Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc API Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Verification**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 2. Direct HTTPS Uvicorn Server (SSL)

To run FastAPI with direct SSL/TLS encryption:

```bash
python backend/run_https.py
```

- **HTTPS Base URL**: [https://127.0.0.1:8443](https://127.0.0.1:8443)
- **HTTPS Swagger Documentation**: [https://127.0.0.1:8443/docs](https://127.0.0.1:8443/docs)

---

## 🛡️ Step 4: Nginx Production Reverse Proxy Setup

Nginx can be used as a front-facing SSL reverse proxy that terminates TLS on port 8443/443 and forwards traffic to the internal Uvicorn server on port 8000.

### 1. Generate SSL Certificates (If Needed)

```bash
python nginx/generate_cert.py
```

This creates `nginx/cert.pem` and `nginx/key.pem`.

### 2. Launch Nginx with Custom Configuration

```bash
# Windows
nginx -c C:/Users/indrajit/Desktop/project-final/nginx/nginx.conf

# Linux
sudo nginx -c /path/to/project-final/nginx/nginx.conf
```

---

## 🌍 Step 5: Global Access (Outside Local Network)

To make your API accessible to users outside your local network without port forwarding:

### Option A: Using Localtunnel
```bash
npx localtunnel --port 8000
```
*Provides a public HTTPS URL (e.g., `https://<subdomain>.loca.lt`).*

### Option B: Using Cloudflare Tunnel
```bash
cloudflared tunnel --url http://127.0.0.1:8000
```
*Provides a direct, zero-interstitial HTTPS URL (e.g., `https://<subdomain>.trycloudflare.com`).*

---

## 🧪 Step 6: Testing & Verification

Run the automated API test suite:

```bash
python backend/test_fastapi.py
```

Expected Output:
```text
Status Code: 200
Health Check: {'status': 'healthy', 'model_loaded': True, 'database_status': 'MySQL connected'}
Prediction Result: {'aqi': 85, 'level': 'Moderate', 'color': '#FFFF00'}
✅ API Verification Passed Successfully!
```
