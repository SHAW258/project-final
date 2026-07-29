import urllib.request
import json

BASE_URL = "http://127.0.0.1:8000"

def safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='ignore').decode('ascii'))

def make_request(url, method="GET", data=None):
    req = urllib.request.Request(url, method=method)
    req.add_header('Content-Type', 'application/json')
    body = json.dumps(data).encode('utf-8') if data else None
    try:
        with urllib.request.urlopen(req, data=body) as response:
            res_body = response.read().decode('utf-8')
            return response.status, json.loads(res_body)
    except urllib.error.HTTPError as e:
        res_body = e.read().decode('utf-8')
        return e.code, json.loads(res_body)

def test_api():
    safe_print("[TEST] Testing FastAPI AQI Server...")
    
    # 1. Health check
    try:
        status, data = make_request(f"{BASE_URL}/health")
        safe_print(f"[SUCCESS] /health status: {status} -> {data}")
    except Exception as e:
        safe_print(f"[ERROR] Could not connect to /health: {e}")
        return

    # 2. API Info
    status, data = make_request(f"{BASE_URL}/info")
    safe_print(f"[SUCCESS] /info status: {status} -> {data}")

    # Sample payload
    payload = {
        "pollutants": {
            "PM2.5": 36.0,
            "PM10": 64.0,
            "NO2": 8.0,
            "SO2": 2.0,
            "CO": 0.303,
            "O3": 0.016
        }
    }

    # 3. Predict endpoint
    status, data = make_request(f"{BASE_URL}/predict", method="POST", data=payload)
    safe_print(f"[SUCCESS] /predict status: {status}")
    if status == 200:
        safe_print(f"   Predicted AQI: {data['prediction']['aqi']} ({data['prediction']['level']})")

    # 4. 24hours endpoint
    status, data = make_request(f"{BASE_URL}/24hours", method="POST", data=payload)
    safe_print(f"[SUCCESS] /24hours status: {status}")
    if status == 200:
        safe_print(f"   24h Avg AQI: {data['statistics']['average']}")

    # 5. 7days endpoint
    status, data = make_request(f"{BASE_URL}/7days", method="POST", data=payload)
    safe_print(f"[SUCCESS] /7days status: {status}")
    if status == 200:
        safe_print(f"   7d Best Day: {data['weekly_statistics']['best_day']}")

    # 6. Database History endpoint
    status, data = make_request(f"{BASE_URL}/history")
    safe_print(f"[SUCCESS] /history status: {status}")
    if status == 200:
        safe_print(f"   Total DB Records stored: {data['total']}")

if __name__ == "__main__":
    test_api()
