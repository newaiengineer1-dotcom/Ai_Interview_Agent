
import os
import re
import json
import sqlite3
from datetime import datetime
from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# APP CONFIG
# ============================================================
st.set_page_config(
    page_title="InterviewAI — AI Interview Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-120b")
WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
DB_PATH = "interviewai.db"


# ============================================================
# PREMIUM DARK SAAS DESIGN
# Inspired by modern AI/SaaS products: quiet chrome,
# strong hierarchy, restrained accent colors, progressive disclosure.
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');

:root {
  --bg: #070A12;
  --panel: #0D111C;
  --panel2: #111827;
  --border: #20283A;
  --text: #F4F7FB;
  --muted: #8B96A9;
  --accent: #7C6CFF;
  --accent2: #23D3EE;
  --green: #37D39A;
  --amber: #F7B955;
  --red: #FF6678;
}

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
}

.stApp {
  background:
    radial-gradient(circle at 80% -10%, rgba(124,108,255,.14), transparent 32%),
    radial-gradient(circle at 15% 0%, rgba(35,211,238,.07), transparent 28%),
    var(--bg);
  color: var(--text);
}

.block-container {
  max-width: 1500px;
  padding: 1.15rem 2rem 3rem;
}

[data-testid="stHeader"] {
  background: rgba(7,10,18,.75);
}

[data-testid="stSidebar"] {
  background: #090D16;
  border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div {
  padding-top: 1rem;
}

section[data-testid="stSidebar"] * {
  color: #DCE3EF;
}

h1, h2, h3, h4 {
  font-family: 'Manrope', sans-serif !important;
  letter-spacing: -.025em;
}

.hero {
  border: 1px solid var(--border);
  background:
    linear-gradient(135deg, rgba(124,108,255,.15), rgba(35,211,238,.045) 45%, rgba(13,17,28,.94));
  border-radius: 24px;
  padding: 28px 30px;
  margin-bottom: 18px;
  box-shadow: 0 20px 70px rgba(0,0,0,.24);
}

.hero-title {
  font-size: 31px;
  font-weight: 800;
  margin: 0;
}

.hero-sub {
  color: var(--muted);
  margin-top: 7px;
  font-size: 14px;
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  border: 1px solid rgba(124,108,255,.35);
  background: rgba(124,108,255,.10);
  color: #BDB6FF;
  border-radius: 999px;
  padding: 6px 10px;
  font-size: 11px;
  font-weight: 700;
  margin-bottom: 13px;
}

.card {
  background: linear-gradient(180deg, rgba(17,24,39,.96), rgba(13,17,28,.96));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 19px;
  box-shadow: 0 12px 40px rgba(0,0,0,.15);
}

.card-title {
  font-size: 12px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .09em;
  font-weight: 700;
}

.card-value {
  font-size: 28px;
  font-weight: 800;
  margin-top: 7px;
}

.card-caption {
  color: var(--muted);
  font-size: 12px;
  margin-top: 4px;
}

.kpi {
  min-height: 116px;
}

.section-title {
  font-size: 18px;
  font-weight: 800;
  margin: 24px 0 11px;
}

.ai-card {
  border: 1px solid rgba(124,108,255,.28);
  background: linear-gradient(135deg, rgba(124,108,255,.10), rgba(13,17,28,.94));
  border-radius: 18px;
  padding: 20px;
}

.question-card {
  border: 1px solid rgba(35,211,238,.25);
  background: linear-gradient(135deg, rgba(35,211,238,.075), rgba(13,17,28,.98));
  border-radius: 20px;
  padding: 25px;
  box-shadow: 0 18px 55px rgba(0,0,0,.18);
}

.question-number {
  color: var(--accent2);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .11em;
  font-weight: 800;
}

.question-text {
  font-family: 'Manrope', sans-serif;
  font-size: 23px;
  line-height: 1.42;
  font-weight: 700;
  margin-top: 9px;
}

.score-good { color: var(--green); }
.score-mid { color: var(--amber); }
.score-low { color: var(--red); }

div[data-testid="stMetric"] {
  background: #0D111C;
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px 16px;
}

div[data-testid="stMetricLabel"] {
  color: var(--muted);
}

div[data-testid="stMetricValue"] {
  color: #F4F7FB;
}

.stButton > button {
  border-radius: 11px;
  border: 1px solid var(--border);
  min-height: 43px;
  font-weight: 700;
  background: #111827;
  color: #F4F7FB;
}

.stButton > button:hover {
  border-color: #5B50D8;
  color: #FFFFFF;
  background: #151C2D;
}

.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #7C6CFF, #5B50D8);
  border: 0;
  color: white;
  box-shadow: 0 10px 30px rgba(92,80,216,.25);
}

.stTextInput input, .stTextArea textarea, .stSelectbox div,
.stFileUploader section {
  background: #0D111C !important;
  border-color: var(--border) !important;
  color: #F4F7FB !important;
}

.stTextArea textarea::placeholder,
.stTextInput input::placeholder {
  color: #657086 !important;
}

hr {
  border-color: var(--border);
}

div[data-testid="stExpander"] {
  background: #0D111C;
  border: 1px solid var(--border);
  border-radius: 14px;
}

.footer {
  text-align: center;
  color: #68758B;
  font-size: 11px;
  padding: 26px 0 5px;
}

.small-muted {
  color: var(--muted);
  font-size: 12px;
}

.pill {
  display: inline-block;
  border-radius: 999px;
  padding: 5px 9px;
  font-size: 10px;
  font-weight: 800;
  background: rgba(55,211,154,.09);
  color: #69E4B4;
  border: 1px solid rgba(55,211,154,.22);
}

div[data-baseweb="tab-list"] {
  gap: 7px;
}

button[data-baseweb="tab"] {
  color: #8B96A9;
  border-radius: 10px;
}

button[data-baseweb="tab"][aria-selected="true"] {
  color: #FFFFFF;
  background: #111827;
}

[data-testid="stProgressBar"] > div > div {
  background: linear-gradient(90deg, #7C6CFF, #23D3EE);
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================
DEFAULTS = {
    "page": "Dashboard",
    "candidate_name": "Alex",
    "target_role": "AI / Software Engineer",
    "experience_level": "Mid-Level",
    "interview_type": "Mixed Technical & Behavioral",
    "personality": "Professional & Structured",
    "difficulty": "Medium",
    "question_limit": 8,
    "resume_text": "",
    "resume_name": "",
    "jd_text": "",
    "jd_name": "",
    "question": "",
    "question_no": 0,
    "answers": [],
    "feedback_history": [],
    "voice_transcript": "",
    "last_error": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# DATABASE
# ============================================================
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            role TEXT,
            interview_type TEXT,
            score REAL,
            questions INTEGER
        )
    """)
    con.commit()
    con.close()


def save_session(role, interview_type, score, questions):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO sessions(timestamp, role, interview_type, score, questions) VALUES(?,?,?,?,?)",
        (datetime.now().isoformat(timespec="seconds"), role, interview_type, score, questions),
    )
    con.commit()
    con.close()


def load_sessions():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM sessions ORDER BY id DESC", con)
    con.close()
    return df


init_db()


# ============================================================
# GROQ
# ============================================================
def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GROQ_API_KEY", "")


@st.cache_resource(show_spinner=False)
def get_client(api_key):
    return Groq(api_key=api_key) if api_key else None


def ai_call(system, user, temperature=0.25, max_tokens=1600):
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it in Streamlit Cloud → Manage app → Settings → Secrets."
        )

    client = get_client(api_key)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("Groq returned an empty response.")
        return content.strip()
    except Exception as exc:
        msg = str(exc)
        low = msg.lower()

        if "401" in msg or "authentication" in low or "invalid api key" in low:
            raise RuntimeError("Groq authentication failed. Check GROQ_API_KEY.") from exc
        if "429" in msg or "rate limit" in low:
            raise RuntimeError("Groq rate limit reached. Please wait and try again.") from exc
        if "404" in msg or "not found" in low or "model" in low and "available" in low:
            raise RuntimeError(
                f"Groq model '{MODEL}' is unavailable. Set GROQ_LLM_MODEL to a currently supported model."
            ) from exc
        raise RuntimeError(f"Groq request failed: {msg}") from exc


# ============================================================
# DOCUMENT / RAG HELPERS
# ============================================================
def extract_text(uploaded):
    if not uploaded:
        return ""

    name = uploaded.name.lower()
    data = uploaded.getvalue()

    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

    return data.decode("utf-8", errors="ignore").strip()


def chunks(text, size=900, overlap=150):
    words = re.sub(r"\s+", " ", text or "").split()
    result = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        part = " ".join(words[start:end])
        if part:
            result.append(part)
        if end >= len(words):
            break
        start = max(0, end - overlap)
    return result


def retrieve_context(query, top_k=5):
    docs = []
    labels = []

    if st.session_state.resume_text:
        docs += chunks(st.session_state.resume_text)
        labels += ["Resume"] * len(chunks(st.session_state.resume_text))

    if st.session_state.jd_text:
        docs += chunks(st.session_state.jd_text)
        labels += ["Job Description"] * len(chunks(st.session_state.jd_text))

    if not docs:
        return []

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(docs)
        q = vectorizer.transform([query])
        scores = cosine_similarity(q, matrix).flatten()
        indices = scores.argsort()[::-1][:top_k]

        return [
            {
                "source": labels[i],
                "score": round(float(scores[i]), 3),
                "text": docs[i],
            }
            for i in indices if scores[i] > 0
        ]
    except Exception:
        return []


def context_text(items):
    return "\n\n".join(
        f"[{x['source']} | relevance {x['score']}]\n{x['text']}"
        for x in items
    )


# ============================================================
# AI INTERVIEW
# ============================================================
def generate_question():
    retrieved = retrieve_context(
        f"{st.session_state.target_role} {st.session_state.interview_type} "
        f"{st.session_state.difficulty}"
    )

    system = """You are InterviewAI, a professional interview coach.
Generate ONE realistic interview question at a time.
Use the resume and job-description evidence when available.
Never invent candidate experience, projects, employers, technologies, metrics,
or achievements. If evidence is absent, ask a general role-relevant question.
Keep the question concise and natural.
Do not provide the answer."""

    user = f"""
Candidate role: {st.session_state.target_role}
Experience: {st.session_state.experience_level}
Interview type: {st.session_state.interview_type}
Interviewer style: {st.session_state.personality}
Difficulty: {st.session_state.difficulty}
Question number: {st.session_state.question_no + 1}

Retrieved evidence:
{context_text(retrieved) or "No resume/JD evidence has been uploaded."}

Generate only the interview question.
"""
    return ai_call(system, user, temperature=0.35, max_tokens=500)


def evaluate_answer(question, answer):
    retrieved = retrieve_context(question, top_k=5)

    system = """You are InterviewAI's senior interview evaluator.
Evaluate the candidate answer fairly and only using information actually present
in the answer and supplied evidence. Do not assume achievements that were not stated.
Return ONLY valid JSON with:
{
 "overall": number from 0 to 10,
 "technical_accuracy": number,
 "completeness": number,
 "depth": number,
 "communication": number,
 "problem_solving": number,
 "role_relevance": number,
 "strengths": [string, string],
 "improvements": [string, string],
 "missed_points": [string],
 "model_answer": string,
 "follow_up": string
}
"""
    user = f"""
Question:
{question}

Candidate answer:
{answer}

Role:
{st.session_state.target_role}

Evidence:
{context_text(retrieved) or "No uploaded evidence."}

Be concise, practical, and constructive.
"""
    raw = ai_call(system, user, temperature=0.15, max_tokens=1700)

    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(match.group(0) if match else raw)
    except Exception:
        return {
            "overall": 0,
            "technical_accuracy": 0,
            "completeness": 0,
            "depth": 0,
            "communication": 0,
            "problem_solving": 0,
            "role_relevance": 0,
            "strengths": [],
            "improvements": ["The AI response could not be parsed. Please try again."],
            "missed_points": [],
            "model_answer": raw,
            "follow_up": "Please retry this answer.",
        }


def transcribe_audio(audio_file):
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing.")

    client = get_client(api_key)
    try:
        result = client.audio.transcriptions.create(
            file=("answer.wav", audio_file.getvalue()),
            model=WHISPER_MODEL,
            response_format="text",
        )
        return str(result)
    except Exception as exc:
        raise RuntimeError(f"Voice transcription failed: {exc}") from exc


# ============================================================
# DASHBOARD HELPERS
# ============================================================
def current_score():
    if not st.session_state.feedback_history:
        return 0.0
    return round(
        sum(float(x.get("overall", 0)) for x in st.session_state.feedback_history)
        / len(st.session_state.feedback_history), 1
    )


def readiness_score():
    base = 58
    if st.session_state.resume_text:
        base += 9
    if st.session_state.jd_text:
        base += 9
    if st.session_state.feedback_history:
        base += min(18, current_score() * 2)
    return min(98, int(base))


def score_class(score):
    if score >= 8:
        return "score-good"
    if score >= 6:
        return "score-mid"
    return "score-low"


def reset_interview():
    st.session_state.question = ""
    st.session_state.question_no = 0
    st.session_state.answers = []
    st.session_state.feedback_history = []
    st.session_state.voice_transcript = ""
    st.session_state.last_error = ""


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        '<div style="font-family:Manrope;font-size:22px;font-weight:800;">🎯 InterviewAI</div>'
        '<div class="small-muted" style="margin:4px 0 20px;">AI Interview Coach</div>',
        unsafe_allow_html=True,
    )

    pages = [
        "Dashboard",
        "Mock Interview",
        "Resume & Job",
        "Voice Coach",
        "My Progress",
    ]
    st.session_state.page = st.radio(
        "WORKSPACE",
        pages,
        index=pages.index(st.session_state.page),
        label_visibility="visible",
    )

    st.markdown("---")
    st.markdown("**Interview profile**")

    st.session_state.target_role = st.text_input(
        "Target role", st.session_state.target_role
    )
    st.session_state.experience_level = st.selectbox(
        "Experience",
        ["Entry-Level", "Junior", "Mid-Level", "Senior", "Lead"],
        index=["Entry-Level", "Junior", "Mid-Level", "Senior", "Lead"].index(
            st.session_state.experience_level
        ),
    )
    st.session_state.interview_type = st.selectbox(
        "Interview type",
        [
            "Mixed Technical & Behavioral",
            "Technical & System Design",
            "Behavioral (STAR)",
            "Coding & Algorithms",
            "System Design",
            "HR & Culture Fit",
        ],
    )
    st.session_state.personality = st.selectbox(
        "Interviewer",
        [
            "Professional & Structured",
            "Friendly & Encouraging",
            "Strict & Demanding",
            "FAANG-Style",
            "Startup CTO",
            "HR Manager",
        ],
    )
    st.session_state.difficulty = st.select_slider(
        "Difficulty",
        options=["Easy", "Medium", "Hard", "Expert"],
        value=st.session_state.difficulty,
    )
    st.session_state.question_limit = st.slider(
        "Questions", 3, 15, st.session_state.question_limit
    )

    st.markdown("---")
    api_ok = bool(get_api_key())
    st.markdown(
        f'<span class="pill">● {"AI ONLINE" if api_ok else "API KEY REQUIRED"}</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"LLM: {MODEL}")
    st.caption(f"Voice: {WHISPER_MODEL}")

    if st.button("↻ Reset Interview", use_container_width=True):
        reset_interview()
        st.rerun()


# ============================================================
# TOP BAR
# ============================================================
st.markdown(
    f"""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
      <div>
        <div style="color:#8B96A9;font-size:12px;">AI CAREER INTELLIGENCE / {st.session_state.page.upper()}</div>
        <div style="font-family:Manrope;font-size:26px;font-weight:800;margin-top:3px;">InterviewAI</div>
      </div>
      <div class="pill">● AI SYSTEM READY</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD
# ============================================================
def dashboard():
    score = current_score()
    ready = readiness_score()

    st.markdown(
        f"""
        <div class="hero">
          <div class="badge">✦ PERSONALIZED AI INTERVIEW COACH</div>
          <div class="hero-title">Practice smarter. Walk into interviews ready.</div>
          <div class="hero-sub">
            Resume-aware questions, adaptive difficulty, real-time feedback and voice coaching
            — in one focused workspace.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Interview readiness", f"{ready}%")
    c2.metric("Current score", f"{score}/10" if score else "—")
    c3.metric("Sessions", len(load_sessions()))
    c4.metric("Questions answered", len(st.session_state.feedback_history))

    st.markdown('<div class="section-title">Your next best action</div>', unsafe_allow_html=True)

    a, b = st.columns([1.55, 1])
    with a:
        st.markdown(
            """
            <div class="ai-card">
              <div class="card-title">AI COACH RECOMMENDATION</div>
              <h3 style="margin:7px 0 6px;">Start a focused mock interview</h3>
              <div class="small-muted">
                Your strongest improvement loop is: answer → evaluate → follow-up → repeat.
                Upload your resume and JD first for deeper personalization.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🎯 Start Mock Interview", type="primary", use_container_width=True):
            st.session_state.page = "Mock Interview"
            st.rerun()

    with b:
        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">PROFILE STATUS</div>
              <div style="font-size:16px;font-weight:800;margin-top:8px;">
                {st.session_state.target_role}
              </div>
              <div class="small-muted" style="margin-top:4px;">
                {st.session_state.experience_level} · {st.session_state.interview_type}
              </div>
              <div style="margin-top:15px;">
                <span class="pill">{'Resume ready' if st.session_state.resume_text else 'Resume missing'}</span>
                &nbsp;
                <span class="pill">{'JD ready' if st.session_state.jd_text else 'JD missing'}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">AI capability map</div>', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            '<div class="card kpi"><div class="card-title">SMART INTERVIEWER</div>'
            '<div class="card-value">Adaptive</div><div class="card-caption">Difficulty changes from your performance.</div></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            '<div class="card kpi"><div class="card-title">GROUNDING</div>'
            '<div class="card-value">RAG Ready</div><div class="card-caption">Questions can use your resume and target JD.</div></div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            '<div class="card kpi"><div class="card-title">VOICE COACH</div>'
            '<div class="card-value">Real-time</div><div class="card-caption">Speech-to-text plus speaking metrics.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Recent performance</div>', unsafe_allow_html=True)
    sessions = load_sessions()
    if sessions.empty:
        st.markdown(
            '<div class="card"><div class="small-muted">No completed sessions yet. '
            'Complete your first mock interview to unlock trend analytics.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.dataframe(
            sessions.head(6)[["timestamp", "role", "interview_type", "score", "questions"]],
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# MOCK INTERVIEW
# ============================================================
def mock_interview():
    st.markdown(
        """
        <div class="hero">
          <div class="badge">● LIVE INTERVIEW MODE</div>
          <div class="hero-title">Your AI interviewer is ready.</div>
          <div class="hero-sub">One question at a time. Answer naturally. Get evidence-based feedback.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    progress = min(
        len(st.session_state.feedback_history) / max(st.session_state.question_limit, 1),
        1.0,
    )
    st.progress(progress, text=f"{len(st.session_state.feedback_history)} / {st.session_state.question_limit} answered")

    if st.session_state.question_no == 0:
        if st.button("✨ Generate First Question", type="primary"):
            try:
                st.session_state.question = generate_question()
                st.session_state.question_no = 1
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        st.info("Tip: Upload your resume and job description for stronger grounding.")
        return

    if len(st.session_state.feedback_history) >= st.session_state.question_limit:
        st.success("Interview complete. Open My Progress to review your performance.")
        if st.button("Start another interview"):
            reset_interview()
            st.rerun()
        return

    st.markdown(
        f"""
        <div class="question-card">
          <div class="question-number">QUESTION {st.session_state.question_no} · {st.session_state.difficulty.upper()}</div>
          <div class="question-text">{st.session_state.question}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Response mode",
        ["✍️ Type", "🎙️ Voice"],
        horizontal=True,
        label_visibility="collapsed",
    )

    answer = ""
    if mode == "✍️ Type":
        answer = st.text_area(
            "Your answer",
            height=210,
            placeholder="Answer as if you are speaking to a real interviewer...",
            key=f"answer_{st.session_state.question_no}",
        )
    else:
        st.markdown(
            '<div class="card"><b>🎙️ Voice answer</b><br>'
            '<span class="small-muted">Record your answer. Whisper will transcribe it for review before evaluation.</span></div>',
            unsafe_allow_html=True,
        )
        audio = st.audio_input(
            "Record answer",
            key=f"audio_{st.session_state.question_no}",
        )
        if audio:
            try:
                with st.spinner("Transcribing your answer..."):
                    st.session_state.voice_transcript = transcribe_audio(audio)
                st.text_area(
                    "Transcript — edit before submitting",
                    value=st.session_state.voice_transcript,
                    height=180,
                    key=f"transcript_{st.session_state.question_no}",
                )
                answer = st.session_state.get(
                    f"transcript_{st.session_state.question_no}",
                    st.session_state.voice_transcript,
                )
            except Exception as exc:
                st.error(str(exc))

    if st.button("Evaluate answer →", type="primary", use_container_width=True):
        if not answer.strip():
            st.warning("Please provide an answer first.")
            return

        with st.spinner("AI Coach is evaluating your answer..."):
            try:
                feedback = evaluate_answer(st.session_state.question, answer)
                st.session_state.answers.append(answer)
                st.session_state.feedback_history.append(feedback)

                score = float(feedback.get("overall", 0))
                if score >= 8.0:
                    st.session_state.difficulty = (
                        "Expert" if st.session_state.difficulty == "Hard" else "Hard"
                    )
                elif score < 6.0:
                    st.session_state.difficulty = "Medium"

                st.session_state.question = generate_question()
                st.session_state.question_no += 1
                st.session_state.voice_transcript = ""
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.feedback_history:
        latest = st.session_state.feedback_history[-1]
        st.markdown('<div class="section-title">Latest AI feedback</div>', unsafe_allow_html=True)

        cols = st.columns(6)
        dims = [
            ("Overall", "overall"),
            ("Technical", "technical_accuracy"),
            ("Complete", "completeness"),
            ("Depth", "depth"),
            ("Communication", "communication"),
            ("Relevance", "role_relevance"),
        ]
        for col, (label, key) in zip(cols, dims):
            value = float(latest.get(key, 0))
            col.metric(label, f"{value:.1f}")

        left, right = st.columns(2)
        with left:
            st.markdown("**Strengths**")
            for x in latest.get("strengths", []):
                st.write("• " + x)
            st.markdown("**Improve next**")
            for x in latest.get("improvements", []):
                st.write("• " + x)
        with right:
            st.markdown("**Missed points**")
            for x in latest.get("missed_points", []):
                st.write("• " + x)
            with st.expander("View model answer"):
                st.write(latest.get("model_answer", ""))


# ============================================================
# RESUME & JOB
# ============================================================
def resume_job():
    st.markdown(
        """
        <div class="hero">
          <div class="badge">✦ PERSONALIZATION ENGINE</div>
          <div class="hero-title">Resume + Job Description intelligence</div>
          <div class="hero-sub">Give InterviewAI the evidence it needs to ask better questions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b = st.columns(2)

    with a:
        st.markdown("### Resume / CV")
        resume = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            key="resume_upload",
        )
        if resume:
            text = extract_text(resume)
            if text:
                st.session_state.resume_text = text
                st.session_state.resume_name = resume.name
                st.success(f"Resume loaded · {len(text.split())} words")
        if st.session_state.resume_name:
            st.caption("Current: " + st.session_state.resume_name)

    with b:
        st.markdown("### Job Description")
        jd = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            key="jd_upload",
        )
        if jd:
            text = extract_text(jd)
            if text:
                st.session_state.jd_text = text
                st.session_state.jd_name = jd.name
                st.success(f"JD loaded · {len(text.split())} words")
        if st.session_state.jd_name:
            st.caption("Current: " + st.session_state.jd_name)

    if st.session_state.resume_text and st.session_state.jd_text:
        st.markdown('<div class="section-title">AI match snapshot</div>', unsafe_allow_html=True)
        retrieved = retrieve_context(st.session_state.target_role, top_k=8)
        avg = round(sum(x["score"] for x in retrieved) / max(len(retrieved), 1) * 100, 0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Evidence coverage", f"{min(avg, 99):.0f}%")
        c2.metric("Resume status", "Ready")
        c3.metric("JD status", "Ready")

        with st.expander("Show retrieved evidence"):
            for i, item in enumerate(retrieved, 1):
                st.markdown(
                    f"**{i}. {item['source']} · relevance {item['score']:.3f}**\n\n"
                    f"{item['text'][:900]}..."
                )


# ============================================================
# VOICE COACH
# ============================================================
def voice_coach():
    st.markdown(
        """
        <div class="hero">
          <div class="badge">🎙 REAL-TIME SPEAKING COACH</div>
          <div class="hero-title">Improve how you sound, not just what you say.</div>
          <div class="hero-sub">Record an answer and review pace, filler words and transcript quality.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    audio = st.audio_input("Record a practice answer")
    if not audio:
        st.markdown(
            '<div class="card"><b>Recommended exercise</b><br>'
            '<span class="small-muted">Answer: “Tell me about a challenging project you worked on and what you learned.” '
            'Aim for 60–90 seconds.</span></div>',
            unsafe_allow_html=True,
        )
        return

    try:
        with st.spinner("Transcribing with Groq Whisper..."):
            transcript = transcribe_audio(audio)

        words = re.findall(r"\b[\w'-]+\b", transcript)
        fillers = re.findall(
            r"\b(um|uh|like|you know|basically|actually|so)\b",
            transcript.lower(),
        )
        word_count = len(words)

        st.markdown('<div class="section-title">Speaking snapshot</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Words", word_count)
        c2.metric("Filler words", len(fillers))
        c3.metric("Estimated clarity", "Strong" if len(fillers) < 5 else "Improve")

        st.markdown("### Transcript")
        st.text_area("Transcript", transcript, height=220)

        if fillers:
            st.warning("Filler words detected: " + ", ".join(sorted(set(fillers))))
        else:
            st.success("No common filler words detected.")
    except Exception as exc:
        st.error(str(exc))


# ============================================================
# PROGRESS
# ============================================================
def progress_page():
    st.markdown(
        """
        <div class="hero">
          <div class="badge">📈 PERFORMANCE INTELLIGENCE</div>
          <div class="hero-title">Know exactly what to improve next.</div>
          <div class="hero-sub">Your interview performance becomes a practical improvement loop.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = st.session_state.feedback_history
    if not history:
        st.info("Complete at least one interview question to unlock your personal analytics.")
    else:
        df = pd.DataFrame(history)
        dims = [
            "technical_accuracy",
            "completeness",
            "depth",
            "communication",
            "problem_solving",
            "role_relevance",
        ]
        labels = [
            "Technical",
            "Completeness",
            "Depth",
            "Communication",
            "Problem solving",
            "Role relevance",
        ]
        values = [float(df.get(x, pd.Series([0])).mean()) for x in dims]

        c1, c2, c3 = st.columns(3)
        c1.metric("Overall", f"{current_score()}/10")
        c2.metric("Strongest", labels[int(max(range(len(values)), key=lambda i: values[i]))])
        c3.metric("Focus next", labels[int(min(range(len(values)), key=lambda i: values[i]))])

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            line=dict(color="#7C6CFF", width=2),
            fillcolor="rgba(124,108,255,.16)",
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(range=[0, 10], gridcolor="#20283A", color="#8B96A9"),
                angularaxis=dict(gridcolor="#20283A", color="#DCE3EF"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#DCE3EF"),
            margin=dict(l=30, r=30, t=25, b=25),
            height=430,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Question-by-question")
        table = df.copy()
        table.insert(0, "Question", [f"Q{i}" for i in range(1, len(table) + 1)])
        st.dataframe(
            table[["Question", "overall"] + dims].rename(
                columns={
                    "overall": "Overall",
                    "technical_accuracy": "Technical",
                    "completeness": "Complete",
                    "depth": "Depth",
                    "communication": "Communication",
                    "problem_solving": "Problem solving",
                    "role_relevance": "Relevance",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    sessions = load_sessions()
    if not sessions.empty:
        st.markdown("### Completed sessions")
        st.dataframe(sessions.head(10), use_container_width=True, hide_index=True)


# ============================================================
# ROUTER
# ============================================================
if st.session_state.page == "Dashboard":
    dashboard()
elif st.session_state.page == "Mock Interview":
    mock_interview()
elif st.session_state.page == "Resume & Job":
    resume_job()
elif st.session_state.page == "Voice Coach":
    voice_coach()
elif st.session_state.page == "My Progress":
    progress_page()

st.markdown(
    '<div class="footer">InterviewAI · AI Interview Coach · Groq + GPT-OSS · Built for focused interview preparation</div>',
    unsafe_allow_html=True,
)
