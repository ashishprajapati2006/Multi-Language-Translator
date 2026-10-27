"""
OmniTranslate API - Root Entry Point
Provides backward compatibility by loading the production backend service from backend/app.py.
"""
import os
import sys

# Ensure backend folder is in Python search path
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
