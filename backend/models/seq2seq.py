import os
import re
import unicodedata
import pickle
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

# Define special tokens
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'

MODEL_REPO = os.getenv("HF_MODEL_REPO", "ashishprajapati2006/translator-model")
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", None)

def normalize_text(text: str) -> str:
    """Normalize text by removing accents and converting to lowercase."""
    text = text.lower().strip()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    text = re.sub(r"([.!?])", r" \1", text)
    text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str):
    """Simple whitespace word tokenization."""
    return text.split()

class Vocabulary:
    """Vocabulary mapper between words and integer indices."""
    def __init__(self, freq_threshold=2):
        self.itos = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
        self.stoi = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        self.freq_threshold = freq_threshold
        
    def __len__(self):
        return len(self.itos)
    
    def numericalize(self, text: str):
        tokenized_text = tokenize(text)
        return [self.stoi.get(token, self.stoi[UNK_TOKEN]) for token in tokenized_text]

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
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
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
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

# Cache for loaded custom models
_custom_models_cache = {}

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_custom_model(src_lang="en_US", tgt_lang="es_ES"):
    """
    Load custom Seq2Seq model for specified language pair.
    Currently supports: en_US -> es_ES
    """
    model_key = f"{src_lang}_to_{tgt_lang}"
    if model_key in _custom_models_cache:
        return _custom_models_cache[model_key]

    if model_key != "en_US_to_es_ES":
        return None

    # Check local models directory first
    local_paths = [
        os.path.join(os.path.dirname(__file__), "..", "models", "translator_en_US_to_es_ES.pkl"),
        os.path.join(os.path.dirname(__file__), "..", "..", "models", "translator_en_US_to_es_ES.pkl"),
    ]
    pkl_path = None
    for path in local_paths:
        abs_p = os.path.abspath(path)
        if os.path.isfile(abs_p):
            pkl_path = abs_p
            break

    if not pkl_path:
        # Download from Hugging Face Hub
        try:
            pkl_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename="translator_en_US_to_es_ES.pkl",
                token=HF_TOKEN
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download custom model from Hugging Face Hub: {str(e)}")

    class _SafeModelUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "Vocabulary":
                return Vocabulary
            return super().find_class(module, name)

    with open(pkl_path, 'rb') as f:
        model_data = _SafeModelUnpickler(f).load()

    src_vocab = model_data['src_vocab']
    tgt_vocab = model_data['tgt_vocab']
    params = model_data['hyperparameters']
    device = get_device()

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

    _custom_models_cache[model_key] = {
        'model': model,
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
        'device': device,
        'params': params
    }

    return _custom_models_cache[model_key]

def translate_custom(sentence: str, src_lang="en_US", tgt_lang="es_ES", max_length=50) -> str:
    """
    Translate source text using the custom Seq2Seq model.
    """
    model_bundle = load_custom_model(src_lang, tgt_lang)
    if not model_bundle:
        raise ValueError(f"Custom model not found for {src_lang} to {tgt_lang}")

    model = model_bundle['model']
    src_vocab = model_bundle['src_vocab']
    tgt_vocab = model_bundle['tgt_vocab']
    device = model_bundle['device']

    model.eval()
    norm_sentence = normalize_text(sentence)
    tokens = [src_vocab.stoi[SOS_TOKEN]] + src_vocab.numericalize(norm_sentence) + [src_vocab.stoi[EOS_TOKEN]]
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

    translated_tokens = [tgt_vocab.itos.get(idx, UNK_TOKEN) for idx in outputs]
    return ' '.join(translated_tokens[1:-1])
