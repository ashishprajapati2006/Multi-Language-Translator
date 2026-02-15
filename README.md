# 🌐 Multi-Language Translator

A complete machine learning project for translating between 10 languages using both custom Seq2Seq (LSTM) and pre-trained mBART models, with Flask and Streamlit web interfaces.

## 🌍 Supported Languages

- 🇬🇧 English (en_US)
- 🇩🇪 German (de_DE)
- 🇮🇳 Hindi (hi_IN)
- 🇪🇸 Spanish (es_ES)
- 🇫🇷 French (fr_FR)
- 🇮🇹 Italian (it_IT)
- 🇸🇦 Arabic (ar_SA)
- 🇳🇱 Dutch (nl_NL)
- 🇯🇵 Japanese (ja_JP)
- 🇵🇹 Portuguese (pt_PT)

## 📁 Project Structure

```
Language translator/
├── app.py                      # Flask web app with dual models support
├── app2.py                     # Streamlit interface for translation
├── all_translate.ipynb         # Complete translation workflow notebook
├── transtate.ipynb             # Model training notebook
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore configuration
├── models/
│   └── mbart_multilingual/     # Pre-trained mBART model files
│       ├── config.json
│       ├── generation_config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       └── tokenizer.json
└── templates/
    └── index3.html             # Flask web interface
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Streamlit App (Recommended)

```bash
streamlit run app2.py
```
Access at: http://localhost:8501

### 3. Run Flask App (Alternative)

```bash
python app.py
```
Access at: http://localhost:5000

## 📊 Model Architecture

### Custom Seq2Seq Model
- **Encoder**: 2-layer LSTM with embedding and dropout
- **Decoder**: 2-layer LSTM with fully connected output layer
- **Language Support**: **English (en_US) → Spanish (es_ES) only**
- **Hyperparameters**:
  - Embedding size: 256
  - Hidden size: 512
  - Layers: 2
  - Dropout: 0.5
  - Optimizer: Adam (learning rate: 0.001)
  - Batch size: 32

### Pre-trained mBART Model
- Facebook's `mbart-large-50-many-to-many-mmt`
- Supports 50+ languages
- Beam search with num_beams=4
- Max length: 128 tokens

## 🎯 Features

### Streamlit App (app2.py)
- ✅ Clean, modern UI with language selection
- ✅ Swap languages button for convenience
- ✅ Model selection (Custom or mBART)
- ✅ Real-time translation
- ✅ Cached model loading for performance
- ✅ Error handling and validation
- ⚠️ **Custom model restricted to English → Spanish translation only**

### Flask App (app.py)
- ✅ Web interface with responsive design
- ✅ Dual endpoints: `/translate_custom` and `/translate_mbart`
- ✅ JSON API support
- ✅ Character counter
- ✅ Loading indicators

## 📖 Training Custom Models

Open `transtate.ipynb` in Jupyter Notebook or VS Code:

1. Load dataset from Hugging Face
2. Exploratory Data Analysis (EDA)
3. Text preprocessing and normalization
4. Build source and target vocabularies
5. Create and train Seq2Seq model
6. Evaluate and test translations
7. Save model as pickle file

To train for different language pairs, modify:
```python
SRC_LANG = 'en_US'  # Change source language
TGT_LANG = 'es_ES'  # Change target language
```

Trained models are saved as: `models/translator_{src_lang}_to_{tgt_lang}.pkl`

## 🔧 API Endpoints (Flask)

### POST /translate_custom
Translate using custom Seq2Seq model

**Request:**
```json
{
  "text": "Hello, how are you?",
  "source_lang": "en_US",
  "target_lang": "es_ES"
}
```

**Response:**
```json
{
  "original": "Hello, how are you?",
  "translation": "hola como estas",
  "source_lang": "English",
  "target_lang": "Spanish"
}
```

### POST /translate_mbart
Translate using mBART model

Same request/response format as above.

## 📚 Dataset

Uses `Amani27/massive_translation_dataset` from Hugging Face, containing:
- Parallel translations across multiple languages
- High-quality translation pairs
- Diverse domains and topics

## 🎨 Customization

### Adjust Model Hyperparameters
Edit in `transtate.ipynb`:
```python
EMBEDDING_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 10
```

## 📦 Dependencies

- **Deep Learning**: torch, transformers
- **Web Framework**: streamlit
- **Data Processing**: numpy, pandas, datasets, nltk, tqdm
- **Visualization**: matplotlib, seaborn

## ⚡ Performance Tips

1. Use GPU if available (automatically detected)
2. Increase `NUM_EPOCHS` for better custom model accuracy
3. Adjust `BATCH_SIZE` based on available memory
4. Use mBART for faster inference (pre-trained)
5. Use custom model for domain-specific translations

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found error | Train the model first using `transtate.ipynb` |
| Out of memory | Reduce `BATCH_SIZE` or `HIDDEN_SIZE` |
| Poor translation quality | Increase `NUM_EPOCHS` or use mBART model |
| SentencePiece missing | `pip install sentencepiece` |
| Streamlit not loading models | Clear Streamlit cache: `streamlit cache clear` |

## 👨‍💻 Built With

- **PyTorch** - Deep learning framework
- **Transformers** - Pre-trained models (mBART)
- **Streamlit** - Web interface
- **Flask** - Alternative web framework
- **Pandas & NumPy** - Data processing
- **Matplotlib & Seaborn** - Visualization

## 📞 Notes

- Custom models are trained using the Seq2Seq architecture with LSTM
- mBART provides faster inference and better multilingual support
- All models are cached after first load for improved performance
- Text is normalized before translation (lowercase, accent removal)
