"""
Translation Models Package
Provides interfaces for custom Seq2Seq LSTM translator and fine-tuned mBART-50 multilingual translator.
"""
from .seq2seq import load_custom_model, translate_custom, Vocabulary
from .mbart import get_mbart_model, translate_mbart, MBART_LANGUAGES

__all__ = [
    "load_custom_model",
    "translate_custom",
    "Vocabulary",
    "get_mbart_model",
    "translate_mbart",
    "MBART_LANGUAGES",
]
