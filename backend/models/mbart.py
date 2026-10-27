import os
import threading
import torch
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

MODEL_REPO = os.getenv("HF_MODEL_REPO", "ashishprajapati2006/translator-model")
FALLBACK_MODEL = "facebook/mbart-large-50-many-to-many-mmt"
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", None)

MBART_LANG_MAP = {
    "en_US": "en_XX",
    "de_DE": "de_DE",
    "hi_IN": "hi_IN",
    "es_ES": "es_XX",
    "fr_FR": "fr_XX",
    "it_IT": "it_IT",
    "ar_SA": "ar_AR",
    "nl_NL": "nl_XX",
    "ja_JP": "ja_XX",
    "pt_PT": "pt_XX",
}

MBART_LANGUAGES = {
    "en_US": {"name": "English", "native": "English", "flag": "🇬🇧"},
    "de_DE": {"name": "German", "native": "Deutsch", "flag": "🇩🇪"},
    "hi_IN": {"name": "Hindi", "native": "हिन्दी", "flag": "🇮🇳"},
    "es_ES": {"name": "Spanish", "native": "Español", "flag": "🇪🇸"},
    "fr_FR": {"name": "French", "native": "Français", "flag": "🇫🇷"},
    "it_IT": {"name": "Italian", "native": "Italiano", "flag": "🇮🇹"},
    "ar_SA": {"name": "Arabic", "native": "العربية", "flag": "🇸🇦", "rtl": True},
    "nl_NL": {"name": "Dutch", "native": "Nederlands", "flag": "🇳🇱"},
    "ja_JP": {"name": "Japanese", "native": "日本語", "flag": "🇯🇵"},
    "pt_PT": {"name": "Portuguese", "native": "Português", "flag": "🇵🇹"},
}

_mbart_lock = threading.Lock()
_mbart_tokenizer = None
_mbart_model = None
_mbart_status = "unloaded"  # "unloaded", "loading", "ready", "error"
_mbart_error = None

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_mbart_status():
    return {
        "status": _mbart_status,
        "error": _mbart_error,
        "device": str(get_device()),
    }

def get_mbart_model():
    """
    Lazy-load mBART model and tokenizer with thread-safety.
    Checks local directories first, then Hugging Face Hub.
    """
    global _mbart_tokenizer, _mbart_model, _mbart_status, _mbart_error

    if _mbart_status == "ready" and _mbart_model is not None and _mbart_tokenizer is not None:
        return _mbart_tokenizer, _mbart_model

    with _mbart_lock:
        if _mbart_status == "ready" and _mbart_model is not None:
            return _mbart_tokenizer, _mbart_model

        _mbart_status = "loading"
        device = get_device()

        # Check local model paths
        local_dirs = [
            os.path.join(os.path.dirname(__file__), "..", "models", "mbart_multilingual"),
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "mbart_multilingual"),
        ]
        local_mbart_dir = None
        for p in local_dirs:
            abs_p = os.path.abspath(p)
            if os.path.isdir(abs_p) and os.path.exists(os.path.join(abs_p, "config.json")):
                local_mbart_dir = abs_p
                break

        try:
            if local_mbart_dir:
                print(f"[mBART] Loading from local directory: {local_mbart_dir}")
                _mbart_tokenizer = MBart50TokenizerFast.from_pretrained(local_mbart_dir)
                _mbart_model = MBartForConditionalGeneration.from_pretrained(local_mbart_dir).to(device)
            else:
                print(f"[mBART] Downloading from Hugging Face: {MODEL_REPO}")
                try:
                    _mbart_tokenizer = MBart50TokenizerFast.from_pretrained(
                        MODEL_REPO,
                        subfolder="mbart_multilingual",
                        token=HF_TOKEN,
                    )
                    _mbart_model = MBartForConditionalGeneration.from_pretrained(
                        MODEL_REPO,
                        subfolder="mbart_multilingual",
                        token=HF_TOKEN,
                    ).to(device)
                except Exception as repo_err:
                    print(f"[mBART] HF subfolder failed ({repo_err}), falling back to {FALLBACK_MODEL}")
                    _mbart_tokenizer = MBart50TokenizerFast.from_pretrained(FALLBACK_MODEL)
                    _mbart_model = MBartForConditionalGeneration.from_pretrained(FALLBACK_MODEL).to(device)

            _mbart_model.eval()
            _mbart_status = "ready"
            _mbart_error = None
            print(f"[mBART] Successfully loaded on {device}")
            return _mbart_tokenizer, _mbart_model

        except Exception as e:
            _mbart_status = "error"
            _mbart_error = str(e)
            print(f"[mBART] Error loading model: {e}")
            raise e

def translate_mbart(text: str, src_lang: str, tgt_lang: str, max_length: int = 128) -> str:
    """
    Translate text between any two supported languages using mBART-50.
    """
    if src_lang not in MBART_LANG_MAP:
        raise ValueError(f"Source language '{src_lang}' is not supported.")
    if tgt_lang not in MBART_LANG_MAP:
        raise ValueError(f"Target language '{tgt_lang}' is not supported.")
    if src_lang == tgt_lang:
        return text

    tokenizer, model = get_mbart_model()
    device = get_device()

    src_code = MBART_LANG_MAP[src_lang]
    tgt_code = MBART_LANG_MAP[tgt_lang]

    tokenizer.src_lang = src_code
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    forced_bos = tokenizer.convert_tokens_to_ids(tgt_code)

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
        )

    translation = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return translation
