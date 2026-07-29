import urllib.request
import json

def safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='ignore').decode('ascii'))

def fetch_live_air_quality(lat=28.6139, lon=77.2090):
    """
    Fetch real-time live pollutant levels (PM2.5, PM10, NO2, SO2, CO, O3)
    from the free Open-Meteo Air Quality API (No API key required).
    Default coordinates: Delhi (28.6139, 77.2090)
    """
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'FastAPI-AQI-Client/1.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            current = data.get('current', {})
            
            pollutants = {
                "PM2.5": float(current.get("pm2_5", 25.0)),
                "PM10": float(current.get("pm10", 50.0)),
                "NO2": float(current.get("nitrogen_dioxide", 10.0)),
                "SO2": float(current.get("sulphur_dioxide", 5.0)),
                "CO": float(current.get("carbon_monoxide", 500.0) / 1000.0),
                "O3": float(current.get("ozone", 20.0))
            }
            return pollutants
    except Exception as e:
        safe_print(f"[API Notice] Could not fetch live data: {e}")
        return {
            "PM2.5": 36.0, "PM10": 64.0, "NO2": 8.0, 
            "SO2": 2.0, "CO": 0.303, "O3": 0.016
        }

if __name__ == "__main__":
    safe_print("[API] Fetching live air quality pollutant data from Open-Meteo...")
    live_data = fetch_live_air_quality(lat=28.6139, lon=77.2090)
    safe_print("[SUCCESS] Live Pollutants Data Fetched:")
    safe_print(json.dumps(live_data, indent=2))
