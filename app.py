"""
OmniTranslate API - Root Entry Point
Delegates to backend/app.py without circular import.
"""
import os
import sys
import importlib.util

BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

spec = importlib.util.spec_from_file_location("backend_app", os.path.join(BACKEND_DIR, "app.py"))
backend_module = importlib.util.module_from_spec(spec)
sys.modules["backend_app"] = backend_module
spec.loader.exec_module(backend_module)
app = backend_module.app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
