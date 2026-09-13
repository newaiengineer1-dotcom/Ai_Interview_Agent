InterviewAI — Commercial Dark AI Interview Coach

A modern Streamlit AI interview-preparation dashboard with a premium dark SaaS design.

What was fixed

Replaced the deprecated/removed llama-3.3-70b-versatile default with:
openai/gpt-oss-120b

Added configurable model support through GROQ_LLM_MODEL

Added friendly handling for API key, authentication, rate-limit and model availability errors

Removed dependency on the missing code.html cockpit

Added a self-contained commercial-style dark dashboard

Main features

Dashboard / readiness score

Resume + Job Description upload

TF-IDF evidence retrieval

Adaptive AI interview questions

Six-dimensional answer evaluation

Voice interview using Groq Whisper

Speaking/filler-word snapshot

Progress radar analytics

SQLite session history

Modern dark SaaS UI

Project structure

InterviewAI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml

Local setup

pip install -r requirements.txt
streamlit run app.py

Set your API key:

Windows PowerShell

$env:GROQ_API_KEY="your_key_here"
streamlit run app.py

macOS/Linux

export GROQ_API_KEY="your_key_here"
streamlit run app.py

Streamlit Cloud

In Manage app → Settings → Secrets, add:

GROQ_API_KEY = "your_key_here"
GROQ_LLM_MODEL = "openai/gpt-oss-120b"
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"

Never commit API keys to GitHub.

Why this UI direction?

The dashboard uses a dark-first, quiet-chrome SaaS pattern: one primary readiness metric, progressive disclosure, restrained accent color, strong typography, and AI recommendations instead of a wall of charts.

Model

Default LLM:

openai/gpt-oss-120b

You can change it without editing the application:

GROQ_LLM_MODEL = "openai/gpt-oss-120b"

