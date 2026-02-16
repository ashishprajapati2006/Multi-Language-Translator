from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from huggingface_hub import hf_hub_download
import pickle
import os
import unicodedata
import re

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MBART_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'mbart_multilingual')
MODEL_NAME = "ashishprajapati2006/translator-model"

# Define special tokens for custom model
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'

# ==================== CUSTOM MODEL (app.py) ====================

def normalize_text(text):
    """Normalize text by removing accents and converting to lowercase"""
    text = text.lower().strip()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    text = re.sub(r"([.!?])", r" \1", text)
    text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    """Simple word tokenization"""
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
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
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
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
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
        encoder_outputs, hidden, cell = self.encoder(source)
        x = target[:, 0]
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = torch.rand(1).item() < teacher_force_ratio
            top1 = output.argmax(1)
            x = target[:, t] if teacher_force else top1
            
        return outputs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_custom_models = {}

def load_custom_model(src_lang, tgt_lang):
    """Load a custom translation model from Hugging Face pickle file."""
    model_key = f"{src_lang}_to_{tgt_lang}"

    if model_key in loaded_custom_models:
        return loaded_custom_models[model_key]

    if model_key != "en_US_to_es_ES":
        return None

    pkl_path = hf_hub_download(
        repo_id=MODEL_NAME,
        filename="translator_en_US_to_es_ES.pkl",
    )
    
    class _ModelUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__" and name == "Vocabulary":
                return Vocabulary
            return super().find_class(module, name)

    with open(pkl_path, 'rb') as f:
        model_data = _ModelUnpickler(f).load()
    
    src_vocab = model_data['src_vocab']
    tgt_vocab = model_data['tgt_vocab']
    params = model_data['hyperparameters']
    
    encoder = Encoder(
        len(src_vocab),
        params['embedding_size'],
        params['hidden_size'],
        params['num_layers'],
        params['dropout']
    ).to(device)
    
    decoder = Decoder(
        len(tgt_vocab),
        params['embedding_size'],
        params['hidden_size'],
        len(tgt_vocab),
        params['num_layers'],
        params['dropout']
    ).to(device)
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()
    
    loaded_custom_models[model_key] = {
        'model': model,
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab
    }
    
    return loaded_custom_models[model_key]

def translate_custom(model, sentence, src_vocab, tgt_vocab, device, max_length=50):
    """Translate using custom model"""
    model.eval()
    
    sentence = normalize_text(sentence)
    tokens = [src_vocab.stoi[SOS_TOKEN]] + src_vocab.numericalize(sentence) + [src_vocab.stoi[EOS_TOKEN]]
    
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
    
    outputs = [tgt_vocab.stoi[SOS_TOKEN]]
    
    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()
        
        outputs.append(best_guess)
        
        if best_guess == tgt_vocab.stoi[EOS_TOKEN]:
            break
    
    translated_tokens = [tgt_vocab.itos[idx] for idx in outputs]
    return ' '.join(translated_tokens[1:-1])

# ==================== MBART MODEL (app2.py) ====================

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

# Load mBART model and tokenizer
print(f"Loading mBART model from {MODEL_NAME}...")

try:
    mbart_tokenizer = MBart50TokenizerFast.from_pretrained(
        MODEL_NAME,
        subfolder="mbart_multilingual",
    )
    mbart_model = MBartForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        subfolder="mbart_multilingual",
    ).to(device)
    mbart_model.eval()
    print(f"mBART model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading mBART model: {e}")
    print("Falling back to base mBART model...")
    mbart_tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt",
    )
    mbart_model = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt",
    ).to(device)
    mbart_model.eval()

def translate_mbart(text, src_lang, tgt_lang):
    """Translate using mBART model"""
    try:
        src_code = mbart_lang_map[src_lang]
        tgt_code = mbart_lang_map[tgt_lang]
        
        mbart_tokenizer.src_lang = src_code
        encoded = mbart_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(device)
        
        # Get the forced BOS token ID for the target language
        forced_bos = mbart_tokenizer.convert_tokens_to_ids(tgt_code)
        
        with torch.no_grad():
            generated = mbart_model.generate(
                **encoded,
                forced_bos_token_id=forced_bos,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
        
        translation = mbart_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return translation
    except Exception as e:
        return f"Error: {str(e)}"

# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index3.html', languages=language_names)

@app.route('/translate_custom', methods=['POST'])
def translate_custom_route():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        src_lang = data.get('source_lang', 'en_US')
        tgt_lang = data.get('target_lang', 'es_ES')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if src_lang == tgt_lang:
            return jsonify({'error': 'Source and target languages must be different'}), 400
        
        # Load custom model
        model_data = load_custom_model(src_lang, tgt_lang)
        
        if model_data is None:
            return jsonify({'error': f'Custom model for {src_lang} to {tgt_lang} not found'}), 404
        
        # Translate
        translation = translate_custom(
            model_data['model'],
            text,
            model_data['src_vocab'],
            model_data['tgt_vocab'],
            device
        )
        
        if not translation.strip():
            return jsonify({'error': 'Model returned empty translation'}), 422
        
        return jsonify({
            'original': text,
            'translation': translation,
            'source_lang': language_names.get(src_lang, src_lang),
            'target_lang': language_names.get(tgt_lang, tgt_lang)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/translate_mbart', methods=['POST'])
def translate_mbart_route():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        src_lang = data.get('source_lang', 'en_US')
        tgt_lang = data.get('target_lang', 'es_ES')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if src_lang == tgt_lang:
            return jsonify({'error': 'Source and target languages must be different'}), 400
        
        translation = translate_mbart(text, src_lang, tgt_lang)
        
        return jsonify({
            'original': text,
            'translation': translation,
            'source_lang': language_names[src_lang],
            'target_lang': language_names[tgt_lang]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
