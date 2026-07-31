import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app import app as asgi_app

try:
    from a2wsgi import WSGIApp
    # WSGIApp converts an ASGI application (FastAPI) into a WSGI application for gunicorn
    app = WSGIApp(asgi_app)
except Exception as e:
    app = asgi_app
