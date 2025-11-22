# Conversational-Based-Mental-Health-Chatbot

# 🧠 Stress Router + Gemini Chatbot

This project is a Gradio web app that routes each user message through a stress classifier, then sends it to one of two Gemini 2.5 Pro chatbots:
- a compassionate, grounding assistant for stressful messages
- a positive, encouraging companion for calm messages

Each turn is logged to a CSV file for analysis.

---

## Key Files and Directories

### Main App Script

- **`stress_router_app.py`**  
  The main entry point (this script). It:
  - Mounts Google Drive in Colab and sets all paths
  - Defines a bilingual tokenizer for English and Chinese (`mixed_tokenize`)
  - Loads the TF-IDF vectorizer and calibrated SVC model
  - Routes each message as STRESS or CALM with a configurable threshold
  - Logs every turn to a CSV file with timestamp, probability, label, and text
  - Initializes two Gemini 2.5 Pro models with different system prompts
  - Builds and launches the Gradio chat interface

---

### Data and Config

All paths are rooted at:

- **`Dataset/Stress_final/`**  
  Base directory for data, models, and logs.

Stopword files for the tokenizer:

- **`english_stopwords.txt`**  
  English stopword list used by the custom tokenizer.

- **`chinese_stopwords.txt`**  
  Chinese stopword list used for character or `jieba`-based tokenization.

If these files are missing, the code falls back to a small built-in stopword list.

---

### Models

Models are loaded from:

- **`Dataset/Stress_final/models/combined_strat/best/`**

This folder must contain:

- **`tfidf_vectorizer.joblib`**  
  Scikit-learn TF-IDF vectorizer trained on the stress dataset.  
  Uses `mixed_tokenize` as its tokenizer.

- **`classifier.joblib`**  
  Calibrated SVC classifier (via `CalibratedClassifierCV`) that outputs  
  `P(stress)` for each input message.

- **`inference_threshold.txt`**  
  Optional single-number file that sets the decision threshold `thr`.  
  If missing, the default is `0.45`.  
  - `P(stress) >= thr` → label `STRESS`  
  - `P(stress) <  thr` → label `CALM`

---

### Logging

- **`logs/`** (created automatically under `Dataset/Stress_final/`)  
  Directory used for turn-level logging.

- **`logs/svc_router_logs.csv`**  
  CSV log of all chat turns. Each row contains:
  - `timestamp`  - in America/Chicago timezone  
  - `prob`       - predicted probability of stress  
  - `label`      - `STRESS` or `CALM`  
  - `text`       - user message (single line)

This file is useful for offline analysis, threshold tuning, and monitoring.

---

### Gemini and UI

- **Gemini configuration** (inside `stress_router_app.py`)  
  - Uses `GOOGLE_API_KEY` from environment variables  
  - Creates:
    - `model_stress` with `SYSTEM_PROMPT_STRESSFUL`
    - `model_calm` with `SYSTEM_PROMPT_CALM`

- **Gradio interface**  
  Defined at the bottom of the script:
  - `gr.Chatbot` to display the conversation
  - `gr.Textbox` for user input
  - `Clear Chat` button to reset state  
  - On each submit:
    - Classify and log the message
    - Call the appropriate Gemini model
    - Show the SVC decision tag, for example  
      `"[svc: STRESS; p=0.78; thr=0.45]"`

---

## Quick Run (Colab)

1. Upload this script to Colab and set your Google Drive structure to match the paths.
2. Add `GOOGLE_API_KEY` to Colab Secrets or environment.
3. Run all cells in `stress_router_app.py`.
4. Open the Gradio link to start chatting with the stress-aware Gemini assistant.
