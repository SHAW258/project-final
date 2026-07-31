import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app import app as asgi_app

try:
    from a2wsgi import ASGIMiddleware
    # WSGI compatible entry point for plain gunicorn calls (gunicorn app:app)
    app = ASGIMiddleware(asgi_app)
except Exception:
    app = asgi_app
