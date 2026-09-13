# 🎯 InterviewAI — Modern AI Interview Coach

A modern, non-technical-friendly Streamlit dashboard for AI-powered interview preparation.

## Features

- Resume/CV upload: PDF, DOCX, TXT
- Job Description upload
- Resume/JD-aware retrieval using lightweight TF-IDF
- AI-generated role-specific interview questions
- Human-like interviewer styles
- Technical, behavioral, system design, coding and mixed interview modes
- Adaptive question flow based on candidate answers
- Six-dimensional answer evaluation
- Voice answers using Groq Whisper
- Interview history with SQLite
- Progress analytics
- Clean, light, modern candidate dashboard

## Project structure

```text
InterviewAI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set your Groq key:

### Windows PowerShell

```powershell
$env:GROQ_API_KEY="your_key_here"
streamlit run app.py
```

### Streamlit Cloud

1. Upload this project to GitHub.
2. Create a new Streamlit app.
3. Select `app.py` as the main file.
4. Open **Settings → Secrets**.
5. Add:

```toml
GROQ_API_KEY = "your_key_here"
```

6. Redeploy.

## AI models

- Text generation/evaluation: `llama-3.3-70b-versatile`
- Speech transcription: `whisper-large-v3-turbo`

If your Groq account exposes different model names, update `LLM_MODEL` or `WHISPER_MODEL` at the top of `app.py`.

## Important

The dashboard intentionally hides technical implementation details from the candidate. RAG, retrieval, LLM orchestration and SQLite remain behind a simple user experience.

Do not commit API keys or `.streamlit/secrets.toml` to GitHub.
