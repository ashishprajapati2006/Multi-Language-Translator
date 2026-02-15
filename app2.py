import os
import re
import unicodedata
import pickle

import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MBART_MODEL_DIR = os.path.join(BASE_DIR, "models", "mbart_multilingual")

# Define special tokens for custom model
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

# Language mapping
mbart_lang_map = {
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

# Readable language names
language_names = {
    "en_US": "English",
    "de_DE": "German",
    "hi_IN": "Hindi",
    "es_ES": "Spanish",
    "fr_FR": "French",
    "it_IT": "Italian",
    "ar_SA": "Arabic",
    "nl_NL": "Dutch",
    "ja_JP": "Japanese",
    "pt_PT": "Portuguese",
}


def normalize_text(text):
    """Normalize text by removing accents and converting to lowercase."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = re.sub(r"([.!?])", r" \1", text)
    text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    """Simple word tokenization."""
    return text.split()


class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
        self.stoi = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def numericalize(self, text):
        tokenized_text = tokenize(text)
        return [self.stoi.get(token, self.stoi[UNK_TOKEN]) for token in tokenized_text]


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedding)
        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1) if x.dim() == 1 else x
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedding, (hidden, cell))
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(1)
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        _, hidden, cell = self.encoder(source)
        x = target[:, 0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = torch.rand(1).item() < teacher_force_ratio
            top1 = output.argmax(1)
            x = target[:, t] if teacher_force else top1

        return outputs


_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def load_custom_model_cached(src_lang, tgt_lang):
    model_path = os.path.join(MODEL_DIR, f"translator_{src_lang}_to_{tgt_lang}.pkl")
    if not os.path.exists(model_path):
        return None

    class _ModelUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__" and name == "Vocabulary":
                return Vocabulary
            return super().find_class(module, name)

    with open(model_path, "rb") as f:
        model_data = _ModelUnpickler(f).load()

    src_vocab = model_data["src_vocab"]
    tgt_vocab = model_data["tgt_vocab"]
    params = model_data["hyperparameters"]

    encoder = Encoder(
        len(src_vocab),
        params["embedding_size"],
        params["hidden_size"],
        params["num_layers"],
        params["dropout"],
    ).to(_device)

    decoder = Decoder(
        len(tgt_vocab),
        params["embedding_size"],
        params["hidden_size"],
        len(tgt_vocab),
        params["num_layers"],
        params["dropout"],
    ).to(_device)

    model = Seq2Seq(encoder, decoder, _device).to(_device)
    model.load_state_dict(model_data["model_state_dict"])
    model.eval()

    return {
        "model": model,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
    }


@st.cache_resource(show_spinner=False)
def load_mbart_model():
    try:
        # Load base mBART model from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt", use_fast=False
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt"
        ).to(_device)
        model.eval()
        return tokenizer, model, None
    except Exception as exc:
        return None, None, str(exc)


def translate_custom(text, src_lang, tgt_lang, max_length=50):
    model_data = load_custom_model_cached(src_lang, tgt_lang)
    if model_data is None:
        return None, f"Custom model for {src_lang} to {tgt_lang} not found."

    model = model_data["model"]
    src_vocab = model_data["src_vocab"]
    tgt_vocab = model_data["tgt_vocab"]

    sentence = normalize_text(text)
    tokens = [src_vocab.stoi[SOS_TOKEN]] + src_vocab.numericalize(sentence) + [
        src_vocab.stoi[EOS_TOKEN]
    ]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(_device)

    with torch.no_grad():
        _, hidden, cell = model.encoder(src_tensor)

    outputs = [tgt_vocab.stoi[SOS_TOKEN]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(_device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()
        outputs.append(best_guess)
        if best_guess == tgt_vocab.stoi[EOS_TOKEN]:
            break

    translated_tokens = [tgt_vocab.itos[idx] for idx in outputs]
    return " ".join(translated_tokens[1:-1]), None


def translate_mbart(text, src_lang, tgt_lang):
    tokenizer, model, load_error = load_mbart_model()
    if not tokenizer or not model:
        return None, f"Error loading mBART model: {load_error}"

    src_code = mbart_lang_map[src_lang]
    tgt_code = mbart_lang_map[tgt_lang]

    tokenizer.src_lang = src_code
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(_device)

    forced_bos = tokenizer.convert_tokens_to_ids(tgt_code)

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos,
            max_length=128,
            num_beams=4,
            early_stopping=True,
        )

    translation = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return translation, None


st.set_page_config(page_title="Language Translator", page_icon="\U0001F310", layout="wide")

st.title("Language Translator")

with st.sidebar:
    st.header("Settings")
    model_type = st.selectbox("Model", ["Custom (Seq2Seq)", "mBART"], index=0)
    st.caption("Pick the translation backend.")
    
    if model_type == "Custom (Seq2Seq)":
        st.info("⚠️ Custom model only supports:\n**English → Spanish**")

if "src_lang" not in st.session_state:
    st.session_state.src_lang = "en_US"
if "tgt_lang" not in st.session_state:
    st.session_state.tgt_lang = "es_ES"

col_left, col_swap, col_right = st.columns([3, 1, 3])

with col_left:
    src_lang = st.selectbox(
        "Source language",
        options=list(language_names.keys()),
        format_func=lambda k: f"{language_names[k]} ({k})",
        index=list(language_names.keys()).index(st.session_state.src_lang),
    )

with col_swap:
    st.write("")
    st.write("")
    if st.button("Swap"):
        st.session_state.src_lang, st.session_state.tgt_lang = (
            st.session_state.tgt_lang,
            st.session_state.src_lang,
        )
        st.rerun()

with col_right:
    tgt_lang = st.selectbox(
        "Target language",
        options=list(language_names.keys()),
        format_func=lambda k: f"{language_names[k]} ({k})",
        index=list(language_names.keys()).index(st.session_state.tgt_lang),
    )

st.session_state.src_lang = src_lang
st.session_state.tgt_lang = tgt_lang

text = st.text_area("Text to translate", height=160, placeholder="Type your text here...")

translate_clicked = st.button("Translate", type="primary")

if translate_clicked:
    if not text.strip():
        st.error("Please enter some text to translate.")
    elif src_lang == tgt_lang:
        st.error("Source and target languages must be different.")
    elif model_type == "Custom (Seq2Seq)" and (src_lang != "en_US" or tgt_lang != "es_ES"):
        st.error("❌ Custom model only supports English (en_US) → Spanish (es_ES) translation. Please switch to mBART for other languages.")
    else:
        with st.spinner("Translating..."):
            if model_type == "Custom (Seq2Seq)":
                translation, error = translate_custom(text, src_lang, tgt_lang)
            else:
                translation, error = translate_mbart(text, src_lang, tgt_lang)

        if error:
            st.error(error)
        elif not translation.strip():
            st.error("Model returned an empty translation.")
        else:
            st.subheader("Translation")
            st.write(translation)
