# 🌐 OmniTranslate AI — Dual-Engine Neural Machine Translator

A modern, full-stack machine learning web application featuring **dual translation engines**: a custom-trained **Seq2Seq LSTM** neural network and a fine-tuned **mBART-50** multilingual transformer.

Engineered with a **decoupled architecture**:
- **Frontend** deployed on **Vercel** (Global Edge CDN, Vanilla HTML5/CSS3/ES6, Glassmorphism design system).
- **Backend API** deployed on **Render** (Flask REST API, PyTorch, Hugging Face Hub, Gunicorn WSGI).

---

## 📸 Key Features

- **Dual Translation Engines**:
  - **Custom Seq2Seq LSTM**: Custom trained PyTorch Encoder-Decoder with attention mechanism for English → Spanish.
  - **mBART-50 Multilingual**: 50-language foundation transformer supporting 10 core languages with multi-beam search.
- **Dual View Modes**:
  - **Single Focus Mode**: Clean, distraction-free translation workbench with model switcher.
  - **Dual Model Comparison**: Real-time side-by-side translation comparison with latency metrics.
- **Voice Input & Audio Playback**:
  - **Speech-to-Text**: Voice input using Web Speech API with real-time waveform recording feedback.
  - **Text-to-Speech**: Native accented audio synthesis for translated output.
- **State-of-the-Art UI/UX**:
  - Obsidian Dark Mode and Daylight Light Mode with persistent preference.
  - Ambient glowing mesh, frosted glass panels (`backdrop-filter: blur`), and micro-interactions.
  - Auto character & word counter, sample prompt chips, 1-click clipboard copy with toast notifications.
  - Translation History drawer with search and restore (stored in `localStorage`).
  - Interactive Backend URL Config modal with live ping latency testing.

---

## 📁 Repository Structure

```
Language translator/
│
├── frontend/                         # Vercel Deployment Root
│   ├── index.html                    # Modern semantic HTML5 application
│   ├── css/
│   │   └── style.css                 # Vanilla CSS design system (tokens, glassmorphism, dark/light)
│   ├── js/
│   │   ├── config.js                 # Dynamic backend URL & environment resolver
│   │   ├── api.js                    # Robust API client with timeout & retry handling
│   │   └── app.js                    # Controller: speech recognition, TTS, history, UI state
│   ├── assets/
│   │   └── favicon.svg               # Vector brand icon
│   └── vercel.json                   # Vercel security headers & static rules
│
├── backend/                          # Render Deployment Root
│   ├── app.py                        # Production Flask API (CORS, health checks, error handling)
│   ├── models/
│   │   ├── __init__.py               # Models package initialization
│   │   ├── seq2seq.py                # Custom PyTorch Encoder, Decoder, Seq2Seq & Vocabulary
│   │   └── mbart.py                  # mBART-50 loader, language mappings & inference
│   ├── requirements.txt              # Lean backend dependencies (Flask, PyTorch, transformers, gunicorn)
│   ├── Procfile                      # Process file for Render / Heroku
│   └── .env.example                  # Environment variables template
│
├── notebooks/                        # Research & Training Notebooks
│   ├── all_translate.ipynb           # End-to-end evaluation workflow
│   ├── transtate.ipynb               # Seq2Seq training notebook
│   └── translate2.ipynb              # Model exploration notebook
│
├── scripts/                          # Utility & Legacy Tools
│   ├── upload_model.py               # Hugging Face Hub upload helper
│   └── streamlit_app.py              # Standalone Streamlit demo app
│
├── render.yaml                       # 1-Click Render Blueprint configuration
├── vercel.json                       # Root Vercel routing configuration
├── .gitignore                        # Git exclusion rules (ignores heavy model weights)
└── README.md                         # Project documentation
```

---

## 🌍 Supported Languages (mBART-50)

| Flag | Language | Code | Native Name | Custom LSTM | mBART-50 |
|:---:|:---|:---:|:---|:---:|:---:|
| 🇬🇧 | English | `en_US` | English | ✅ | ✅ |
| 🇪🇸 | Spanish | `es_ES` | Español | ✅ | ✅ |
| 🇫🇷 | French | `fr_FR` | Français | — | ✅ |
| 🇩🇪 | German | `de_DE` | Deutsch | — | ✅ |
| 🇮🇳 | Hindi | `hi_IN` | हिन्दी | — | ✅ |
| 🇯🇵 | Japanese | `ja_JP` | 日本語 | — | ✅ |
| 🇮🇹 | Italian | `it_IT` | Italiano | — | ✅ |
| 🇵🇹 | Portuguese | `pt_PT` | Português | — | ✅ |
| 🇳🇱 | Dutch | `nl_NL` | Nederlands | — | ✅ |
| 🇸🇦 | Arabic | `ar_SA` | العربية | — | ✅ |

---

## 🚀 Deployment Guide

### Part 1: Deploy Backend to Render

Render hosts the Python Flask backend with PyTorch and Hugging Face models.

#### Option A: Using `render.yaml` Blueprint (Easiest)
1. Push your repository to GitHub.
2. Log in to [Render Dashboard](https://dashboard.render.com).
3. Click **New +** → **Blueprint**.
4. Select your GitHub repository.
5. Render will detect `render.yaml` and configure:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180 app:app`
   - **Health Check Path**: `/health`
6. Click **Apply**.
7. Once deployed, copy your Render web service URL (e.g., `https://multi-language-translator.onrender.com`).

#### Option B: Manual Web Service Setup
1. On Render, click **New +** → **Web Service**.
2. Connect your GitHub repository.
3. Configure the following settings:
   - **Name**: `language-translator-api`
   - **Region**: Closest to your users (e.g., Oregon or Frankfurt)
   - **Root Directory**: `backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180 app:app`
   - **Plan**: Free (or Starter for higher RAM)
4. Under **Advanced** → **Environment Variables**:
   - `PYTHON_VERSION`: `3.11.9`
   - `FRONTEND_URL`: `*` (or your Vercel URL once deployed)
   - `HF_MODEL_REPO`: `ashishprajapati2006/translator-model`
   - `WARMUP_ON_START`: `false`
5. Click **Create Web Service**.

---

### Part 2: Deploy Frontend to Vercel

Vercel provides instant global edge deployment for the frontend.

#### Option A: Via Vercel Web Dashboard (Recommended)
1. Go to [Vercel Dashboard](https://vercel.com) and click **Add New...** → **Project**.
2. Import your GitHub repository.
3. In the project configuration screen:
   - **Framework Preset**: `Other`
   - **Root Directory**: Click *Edit* and select **`frontend`** (or leave root since `vercel.json` routes to `frontend`).
4. Click **Deploy**.
5. Your frontend is live in under 15 seconds! (e.g., `https://my-translator.vercel.app`).

#### Connecting Frontend to Render Backend:
- In the deployed Vercel web app, click the **⚙️ Settings** icon in the top right.
- Enter your Render backend URL (e.g. `https://multi-language-translator.onrender.com`).
- Click **Ping & Test** to verify connectivity, then click **Save URL**.
- The setting is saved in your browser's `localStorage` and will persist across sessions.

---

## 💻 Local Development Setup

To run both frontend and backend locally on your computer:

### 1. Start the Flask Backend
```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment (optional)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run backend on port 5000
python app.py
```
Backend will be live at: `http://localhost:5000` (Health check: `http://localhost:5000/health`)

### 2. Start the Frontend
In another terminal:
```bash
# Navigate to frontend directory
cd frontend

# Run any static server (Python or Node)
python -m http.server 3000
# or npx serve .
```
Frontend will be live at: `http://localhost:3000`

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description | Sample Payload |
|:---:|:---|:---|:---|
| `GET` | `/health` | Service health status & loaded models | None |
| `GET` | `/api/languages` | List supported languages & metadata | None |
| `POST` | `/api/translate` | Unified translation endpoint (`mbart`, `custom`, or `both`) | `{"text": "Hello", "source_lang": "en_US", "target_lang": "es_ES", "model_type": "both"}` |
| `POST` | `/api/translate/custom` | Custom Seq2Seq LSTM translation (`en_US` → `es_ES`) | `{"text": "Hello world", "source_lang": "en_US", "target_lang": "es_ES"}` |
| `POST` | `/api/translate/mbart` | Fine-tuned mBART-50 multilingual translation | `{"text": "Good morning", "source_lang": "en_US", "target_lang": "hi_IN"}` |

---

## 🧠 Model Specifications

### 1. Custom Seq2Seq LSTM Model
- **Architecture**: Encoder-Decoder with stacked LSTM layers.
- **Tokenizer**: Custom word-level dictionary with special tokens (`<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`).
- **Storage**: Pickled model bundle containing vocabulary mappings, state dict, and hyperparameters.
- **Hugging Face Asset**: [`ashishprajapati2006/translator-model/translator_en_US_to_es_ES.pkl`](https://huggingface.co/ashishprajapati2006/translator-model)

### 2. Fine-tuned mBART-50 Multilingual Model
- **Base Model**: `facebook/mbart-large-50-many-to-many-mmt`
- **Tokenizer**: `MBart50TokenizerFast`
- **Beam Search**: 4 beams with early stopping for high translation fidelity.
- **Hugging Face Asset**: [`ashishprajapati2006/translator-model/mbart_multilingual`](https://huggingface.co/ashishprajapati2006/translator-model)

---

## 🛡️ License

MIT License. Open-source for educational and research purposes.
