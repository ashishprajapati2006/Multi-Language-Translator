import os
import time
import threading
from flask import Flask, request, jsonify

from models.seq2seq import load_custom_model, translate_custom
from models.mbart import translate_mbart, MBART_LANGUAGES, get_mbart_status, get_mbart_model

app = Flask(__name__)

# Configure CORS - allows frontend on Vercel or localhost to communicate seamlessly
FRONTEND_ORIGIN = os.getenv("FRONTEND_URL", "*")
try:
    from flask_cors import CORS
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
except ImportError:
    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

# Warmup flag
WARMUP_ON_START = os.getenv("WARMUP_ON_START", "false").lower() in ("true", "1", "yes")

def warmup_models():
    """Background thread to pre-load models without blocking server startup."""
    try:
        print("[Warmup] Pre-warming models in background...")
        # Pre-load custom model
        try:
            load_custom_model("en_US", "es_ES")
            print("[Warmup] Custom model pre-loaded successfully.")
        except Exception as e:
            print(f"[Warmup] Custom model warning: {e}")

        # Pre-load mBART
        try:
            get_mbart_model()
            print("[Warmup] mBART model pre-loaded successfully.")
        except Exception as e:
            print(f"[Warmup] mBART model warning: {e}")
    except Exception as e:
        print(f"[Warmup] Error during warmup: {e}")

if WARMUP_ON_START:
    warmup_thread = threading.Thread(target=warmup_models, daemon=True)
    warmup_thread.start()

# ==================== HEALTH & INFO ROUTES ====================

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "name": "Neural Multi-Language Translator API",
        "status": "online",
        "documentation": "/api/languages",
        "health": "/health",
        "version": "2.0.0"
    })

@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health():
    mbart_info = get_mbart_status()
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "models": {
            "custom_seq2seq": {
                "name": "Custom LSTM Seq2Seq",
                "pairs": ["en_US -> es_ES"],
                "status": "ready"
            },
            "mbart_50": {
                "name": "Fine-tuned mBART-50 Multilingual",
                "status": mbart_info["status"],
                "device": mbart_info["device"],
                "supported_languages": len(MBART_LANGUAGES)
            }
        }
    })

@app.route("/api/languages", methods=["GET"])
def get_languages():
    """Return dictionary of supported languages with metadata."""
    languages_list = []
    for code, meta in MBART_LANGUAGES.items():
        languages_list.append({
            "code": code,
            "name": meta["name"],
            "native": meta["native"],
            "flag": meta["flag"],
            "rtl": meta.get("rtl", False),
            "custom_model_supported": (code in ("en_US", "es_ES")),
            "mbart_supported": True
        })
    return jsonify({
        "languages": languages_list,
        "custom_pairs": [
            {"source": "en_US", "target": "es_ES"}
        ]
    })

# ==================== TRANSLATION ROUTES ====================

@app.route("/api/translate/custom", methods=["POST"])
def translate_custom_endpoint():
    """Endpoint specifically for Custom Seq2Seq model (en_US -> es_ES)."""
    start_time = time.time()
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()
        src_lang = data.get("source_lang", "en_US")
        tgt_lang = data.get("target_lang", "es_ES")

        if not text:
            return jsonify({"error": "No text provided to translate."}), 400

        if src_lang != "en_US" or tgt_lang != "es_ES":
            return jsonify({
                "error": f"Custom model only supports 'en_US' (English) to 'es_ES' (Spanish). Got {src_lang} -> {tgt_lang}."
            }), 400

        translation = translate_custom(text, src_lang, tgt_lang)
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return jsonify({
            "success": True,
            "model": "custom_seq2seq",
            "original": text,
            "translation": translation,
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "duration_ms": elapsed_ms
        })
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route("/api/translate/mbart", methods=["POST"])
def translate_mbart_endpoint():
    """Endpoint specifically for mBART multilingual model."""
    start_time = time.time()
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()
        src_lang = data.get("source_lang", "en_US")
        tgt_lang = data.get("target_lang", "es_ES")

        if not text:
            return jsonify({"error": "No text provided to translate."}), 400

        if src_lang == tgt_lang:
            return jsonify({
                "success": True,
                "model": "mbart_50",
                "original": text,
                "translation": text,
                "source_lang": src_lang,
                "target_lang": tgt_lang,
                "duration_ms": 0
            })

        translation = translate_mbart(text, src_lang, tgt_lang)
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return jsonify({
            "success": True,
            "model": "mbart_50",
            "original": text,
            "translation": translation,
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "duration_ms": elapsed_ms
        })
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route("/api/translate", methods=["POST"])
def translate_unified():
    """
    Unified translation endpoint.
    Supports model_type: 'mbart', 'custom', or 'both'.
    """
    start_time = time.time()
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()
        src_lang = data.get("source_lang", "en_US")
        tgt_lang = data.get("target_lang", "es_ES")
        model_type = data.get("model_type", "both").lower()

        if not text:
            return jsonify({"error": "No text provided to translate."}), 400

        results = {
            "success": True,
            "original": text,
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "translations": {}
        }

        # Handle custom model
        if model_type in ("custom", "both"):
            if src_lang == "en_US" and tgt_lang == "es_ES":
                try:
                    c_start = time.time()
                    c_trans = translate_custom(text, src_lang, tgt_lang)
                    results["translations"]["custom"] = {
                        "translation": c_trans,
                        "duration_ms": round((time.time() - c_start) * 1000, 2),
                        "model_name": "Custom LSTM Seq2Seq"
                    }
                except Exception as ce:
                    results["translations"]["custom"] = {
                        "error": str(ce),
                        "model_name": "Custom LSTM Seq2Seq"
                    }
            elif model_type == "custom":
                return jsonify({
                    "error": "Custom model only supports translation from English (en_US) to Spanish (es_ES)."
                }), 400

        # Handle mBART model
        if model_type in ("mbart", "both"):
            try:
                m_start = time.time()
                m_trans = translate_mbart(text, src_lang, tgt_lang)
                results["translations"]["mbart"] = {
                    "translation": m_trans,
                    "duration_ms": round((time.time() - m_start) * 1000, 2),
                    "model_name": "mBART-50 Multilingual"
                }
            except Exception as me:
                results["translations"]["mbart"] = {
                    "error": str(me),
                    "model_name": "mBART-50 Multilingual"
                }

        results["total_duration_ms"] = round((time.time() - start_time) * 1000, 2)
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
