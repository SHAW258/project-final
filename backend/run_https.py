import uvicorn
from database import safe_print

if __name__ == '__main__':
    safe_print("🔒 Starting HTTPS FastAPI server on https://127.0.0.1:8443")
    safe_print("📖 HTTPS Swagger Documentation available at https://127.0.0.1:8443/docs")
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8443, 
        ssl_keyfile="key.pem", 
        ssl_certfile="cert.pem", 
        reload=True
    )
