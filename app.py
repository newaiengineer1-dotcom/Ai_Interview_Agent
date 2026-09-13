import os
import io
import re
import sqlite3
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# InterviewAI — Modern Candidate Dashboard
# ============================================================

st.set_page_config(
    page_title="InterviewAI — AI Interview Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DB_PATH = "interviewai.db"
LLM_MODEL = "llama-3.3-70b-versatile"
WHISPER_MODEL = "whisper-large-v3-turbo"

# ----------------------------- Styling -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');

:root {
  --bg:#f6f8fc; --card:#ffffff; --ink:#172033; --muted:#687386;
  --primary:#5b5ce2; --primary2:#7778f5; --success:#18a673;
  --warning:#e9a52b; --border:#e6eaf1;
}
.stApp { background:var(--bg); color:var(--ink); font-family:Inter,sans-serif; }
.block-container { max-width:1240px; padding:1.25rem 2rem 3rem; }
h1,h2,h3,h4 { font-family:Outfit,sans-serif !important; color:var(--ink) !important; }
h1 { font-size:2.15rem !important; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { display:none; }
div[data-testid="stMetric"] {
  background:var(--card); border:1px solid var(--border); border-radius:16px;
  padding:14px 16px; box-shadow:0 4px 18px rgba(30,42,70,.04);
}
.hero {
  background:linear-gradient(135deg,#ffffff 0%,#f0f1ff 100%);
  border:1px solid #e3e5ff; border-radius:24px; padding:28px 30px;
  margin-bottom:20px; box-shadow:0 12px 35px rgba(60,70,160,.07);
}
.hero-kicker { color:var(--primary); font-size:.82rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
.hero h1 { margin:.25rem 0 .35rem; }
.hero p { color:var(--muted); margin:0; font-size:1rem; }
.card {
  background:var(--card); border:1px solid var(--border); border-radius:18px;
  padding:20px; box-shadow:0 5px 22px rgba(30,42,70,.045); height:100%;
}
.card-title { font-family:Outfit,sans-serif; font-size:1.05rem; font-weight:700; margin-bottom:4px; }
.muted { color:var(--muted); font-size:.9rem; }
.badge { display:inline-block; padding:5px 9px; border-radius:999px; font-size:.75rem; font-weight:700; }
.badge-green { background:#e8f8f1; color:#13835d; }
.badge-purple { background:#eeeeff; color:#5556cf; }
.badge-amber { background:#fff5df; color:#a86c09; }
.score {
  font-family:Outfit,sans-serif; font-size:3rem; font-weight:800; color:var(--primary);
  line-height:1;
}
.section { margin-top:26px; margin-bottom:12px; }
.tip { background:#f1f5ff; border:1px solid #dfe5ff; border-radius:14px; padding:13px 15px; color:#404a63; }
.answer-box { background:#fbfcff; border:1px solid var(--border); border-radius:14px; padding:16px; }
hr { border:none; border-top:1px solid var(--border); margin:24px 0; }
.stButton > button {
  border-radius:12px !important; min-height:42px !important; font-weight:700 !important;
}
button[kind="primary"] {
  background:linear-gradient(135deg,#5b5ce2,#7778f5) !important;
  border:none !important;
}
[data-testid="stFileUploader"] {
  border:1px dashed #cbd2e2; border-radius:14px; background:#fbfcff;
}
.small { font-size:.82rem; color:var(--muted); }
</style>
""", unsafe_allow_html=True)

# ----------------------------- Database -----------------------------
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS sessions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT, interview_type TEXT, score REAL, created_at TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS answers(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER, question TEXT, answer TEXT, score REAL,
            feedback TEXT, created_at TEXT
        )""")

init_db()

# ----------------------------- State -----------------------------
defaults = {
    "started": False, "session_id": None, "candidate": "",
    "role": "Software Engineer", "level": "Mid-Level",
    "interview_type": "Mixed Technical & Behavioral",
    "personality": "Professional",
    "difficulty": "Medium", "question_limit": 6,
    "resume_text": "", "resume_name": "",
    "jd_text": "", "jd_name": "",
    "question": "", "question_no": 0, "answers": [],
    "feedback": [], "last_eval": None, "voice_transcript": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------------- Helpers -----------------------------
def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")

def client():
    key = get_api_key()
    return Groq(api_key=key) if key else None

def extract_text(upload):
    if not upload:
        return ""
    data = upload.getvalue()
    name = upload.name.lower()
    try:
        if name.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(data))
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        if name.endswith(".docx"):
            doc = Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        return data.decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Could not read {upload.name}: {e}")
        return ""

def chunks(text, size=900, overlap=120):
    words = re.findall(r"\S+", text or "")
    out = []
    start = 0
    while start < len(words):
        out.append(" ".join(words[start:start+size]))
        start += max(1, size-overlap)
    return out

def retrieve_context(query, documents, top_k=5):
    docs = []
    for label, text in documents:
        for i, ch in enumerate(chunks(text)):
            docs.append((label, i+1, ch))
    if not docs:
        return ""
    corpus = [query] + [d[2] for d in docs]
    try:
        mat = TfidfVectorizer(stop_words="english").fit_transform(corpus)
        sims = cosine_similarity(mat[0:1], mat[1:]).ravel()
        idxs = sims.argsort()[::-1][:top_k]
        return "\n\n".join(
            f"[{docs[i][0]} • chunk {docs[i][1]}]\n{docs[i][2]}"
            for i in idxs if sims[i] > 0
        )
    except Exception:
        return "\n\n".join(f"[{d[0]}]\n{d[2]}" for d in docs[:top_k])

def ask_ai(system, user):
    c = client()
    if not c:
        return ""
    r = c.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.25,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
    )
    return r.choices[0].message.content.strip()

def generate_question():
    context = retrieve_context(
        f"{st.session_state.role} {st.session_state.interview_type} {st.session_state.difficulty}",
        [("Resume", st.session_state.resume_text), ("Job Description", st.session_state.jd_text)],
    )
    system = """You are InterviewAI, a professional human-like interviewer.
Generate exactly ONE practical interview question. Ground it in the supplied
resume/job context whenever available. Never invent candidate experience.
The question must match the target role, interview type, experience level and
difficulty. Do not include the answer or multiple questions."""
    user = f"""Role: {st.session_state.role}
Level: {st.session_state.level}
Type: {st.session_state.interview_type}
Difficulty: {st.session_state.difficulty}
Question number: {st.session_state.question_no + 1}
Context:
{context or "No documents uploaded. Ask a role-appropriate general question."}"""
    q = ask_ai(system, user)
    if not q:
        q = f"Tell me about a project or experience that best demonstrates your ability as a {st.session_state.role}."
    return q

def evaluate(answer):
    context = retrieve_context(
        answer, [("Resume", st.session_state.resume_text), ("Job Description", st.session_state.jd_text)]
    )
    system = """You are an expert interview coach. Evaluate the candidate answer
using only what is supported by the answer and provided context. Do not invent
facts about the candidate. Return valid JSON only with keys:
technical_accuracy, completeness, depth, communication, problem_solving,
role_relevance, overall, strengths, improvements, model_answer.
All six dimension scores and overall are 1-10 numbers. strengths and
improvements are arrays of short strings."""
    user = f"""Role: {st.session_state.role}
Question: {st.session_state.question}
Candidate answer: {answer}
Relevant context:
{context or "No resume/JD context available."}"""
    raw = ask_ai(system, user)
    try:
        raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.I)
        return json.loads(raw)
    except Exception:
        return {
            "technical_accuracy": 7, "completeness": 7, "depth": 6,
            "communication": 7, "problem_solving": 7, "role_relevance": 7,
            "overall": 7, "strengths": ["You provided a direct response."],
            "improvements": ["Add a concrete example and explain your reasoning."],
            "model_answer": "Structure the answer with context, action, reasoning, and result."
        }

def save_current_session():
    if not st.session_state.answers:
        return
    scores = [a["evaluation"]["overall"] for a in st.session_state.answers]
    score = round(sum(scores)/len(scores)*10, 1)
    with db() as c:
        cur = c.execute(
            "INSERT INTO sessions(role,interview_type,score,created_at) VALUES(?,?,?,?)",
            (st.session_state.role, st.session_state.interview_type, score, datetime.now().isoformat(timespec="seconds"))
        )
        sid = cur.lastrowid
        for a in st.session_state.answers:
            c.execute(
                "INSERT INTO answers(session_id,question,answer,score,feedback,created_at) VALUES(?,?,?,?,?,?)",
                (sid, a["question"], a["answer"], a["evaluation"]["overall"],
                 json.dumps(a["evaluation"]), datetime.now().isoformat(timespec="seconds"))
            )

def history():
    with db() as c:
        return pd.read_sql_query(
            "SELECT role, interview_type, score, created_at FROM sessions ORDER BY id DESC LIMIT 10", c
        )

# ----------------------------- Header -----------------------------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">🎯 InterviewAI</div>
  <h1>Your Personal AI Interview Coach</h1>
  <p>Practice smarter with resume-aware questions, adaptive feedback, voice answers and clear improvement guidance.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------- Navigation -----------------------------
pages = ["🏠 Dashboard", "🎯 Mock Interview", "📄 Resume & Job", "🎤 Voice Coach", "📈 My Progress"]
page = st.radio("", pages, horizontal=True, label_visibility="collapsed")

# ============================================================
# RESUME & JOB
# ============================================================
if page == "📄 Resume & Job":
    st.markdown('<div class="section"><h2>Prepare your interview</h2></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><div class="card-title">📄 Your Resume</div><div class="muted">Upload PDF, DOCX or TXT</div>', unsafe_allow_html=True)
        f = st.file_uploader("Resume", type=["pdf","docx","txt"], key="resume", label_visibility="collapsed")
        if f:
            st.session_state.resume_text = extract_text(f)
            st.session_state.resume_name = f.name
        if st.session_state.resume_name:
            st.success(f"✓ Ready: {st.session_state.resume_name}")
        else:
            st.info("Upload your CV/resume to personalize questions.")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="card-title">💼 Target Job</div><div class="muted">Upload the job description</div>', unsafe_allow_html=True)
        f = st.file_uploader("Job Description", type=["pdf","docx","txt"], key="jd", label_visibility="collapsed")
        if f:
            st.session_state.jd_text = extract_text(f)
            st.session_state.jd_name = f.name
        if st.session_state.jd_name:
            st.success(f"✓ Ready: {st.session_state.jd_name}")
        else:
            st.info("Add a JD so InterviewAI can focus on the role.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section"><h2>Interview setup</h2></div>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    with a:
        st.session_state.candidate = st.text_input("Your name", st.session_state.candidate, placeholder="e.g. Alex")
        st.session_state.role = st.text_input("Target role", st.session_state.role)
    with b:
        st.session_state.level = st.selectbox("Experience level", ["Junior","Mid-Level","Senior","Lead"], index=["Junior","Mid-Level","Senior","Lead"].index(st.session_state.level))
        st.session_state.interview_type = st.selectbox("Interview focus", ["Mixed Technical & Behavioral","Technical","Behavioral (STAR)","System Design","Coding & Algorithms","HR & Culture"])
    with c:
        st.session_state.personality = st.selectbox("Interviewer style", ["Professional","Friendly","Challenging","FAANG-Style","Startup CTO","HR Manager"])
        st.session_state.difficulty = st.select_slider("Difficulty", ["Easy","Medium","Hard","Expert"], value=st.session_state.difficulty)
    st.session_state.question_limit = st.slider("Questions per session", 3, 12, st.session_state.question_limit)

# ============================================================
# DASHBOARD
# ============================================================
elif page == "🏠 Dashboard":
    h = history()
    current = round(sum(a["evaluation"]["overall"] for a in st.session_state.answers)/len(st.session_state.answers)*10) if st.session_state.answers else (int(h.iloc[0]["score"]) if not h.empty else 0)
    previous = int(h.iloc[1]["score"]) if len(h) > 1 else current
    improvement = current - previous

    st.markdown(f"""
    <div style="margin:18px 0 8px">
      <div class="small">WELCOME BACK 👋</div>
      <h2 style="margin:2px 0">{st.session_state.candidate or "Candidate"}</h2>
      <div class="muted">Ready to improve your interview performance?</div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class="card"><div class="card-title">📄 Your Resume</div>
        <div class="muted">{st.session_state.resume_name or "No resume uploaded yet"}</div>
        <br><span class="badge {'badge-green' if st.session_state.resume_text else 'badge-amber'}">
        {'✓ Analyzed' if st.session_state.resume_text else 'Upload to personalize'}</span></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="card"><div class="card-title">💼 Target Job</div>
        <div class="muted">{st.session_state.jd_name or st.session_state.role}</div>
        <br><span class="badge {'badge-green' if st.session_state.jd_text else 'badge-purple'}">
        {'✓ Job understood' if st.session_state.jd_text else 'Add a job description'}</span></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section"><h2>Interview readiness</h2></div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Readiness", f"{current}%")
    c2.metric("Sessions", len(h))
    c3.metric("Change", f"{improvement:+d}%" if h.shape[0] > 1 else "—")
    c4.metric("Focus areas", len(st.session_state.feedback[-1]["evaluation"].get("improvements", [])) if st.session_state.feedback else "—")

    st.markdown('<div class="section"><h2>What to improve next</h2></div>', unsafe_allow_html=True)
    if st.session_state.feedback:
        ev = st.session_state.feedback[-1]["evaluation"]
        for x in ev.get("improvements", [])[:3]:
            st.markdown(f'<div class="tip">⚠️ {x}</div><br>', unsafe_allow_html=True)
    else:
        st.info("Complete a mock interview and your personal improvement areas will appear here.")

    st.markdown('<div class="section"><h2>Quick practice</h2></div>', unsafe_allow_html=True)
    q1,q2,q3 = st.columns(3)
    with q1:
        if st.button("🎯 Start Mock Interview", use_container_width=True, type="primary"):
            st.session_state.started = True
            st.session_state.question_no = 0
            st.session_state.answers = []
            st.session_state.feedback = []
            st.session_state.question = ""
            st.rerun()
    with q2:
        if st.button("📄 Prepare Resume & Job", use_container_width=True):
            st.session_state._nav_hint = "Resume"
            st.info("Open “Resume & Job” above to upload your documents.")
    with q3:
        if st.button("📈 View My Progress", use_container_width=True):
            st.info("Open “My Progress” above.")

# ============================================================
# MOCK INTERVIEW
# ============================================================
elif page == "🎯 Mock Interview":
    if not st.session_state.started:
        st.markdown('<div class="card"><h2>Start your mock interview</h2><p class="muted">InterviewAI will ask one question at a time and adapt based on your answers.</p></div>', unsafe_allow_html=True)
        a,b = st.columns(2)
        with a:
            st.session_state.role = st.text_input("Target role", st.session_state.role)
            st.session_state.interview_type = st.selectbox("Interview focus", ["Mixed Technical & Behavioral","Technical","Behavioral (STAR)","System Design","Coding & Algorithms","HR & Culture"], key="start_type")
        with b:
            st.session_state.level = st.selectbox("Experience level", ["Junior","Mid-Level","Senior","Lead"], key="start_level")
            st.session_state.personality = st.selectbox("Interviewer style", ["Professional","Friendly","Challenging","FAANG-Style","Startup CTO","HR Manager"], key="start_personality")
        if st.button("🚀 Start Interview", type="primary", use_container_width=True):
            st.session_state.started = True
            st.session_state.question_no = 0
            st.session_state.answers = []
            st.session_state.feedback = []
            st.session_state.question = ""
            st.rerun()
    else:
        completed = len(st.session_state.answers)
        progress = completed / st.session_state.question_limit
        st.progress(progress, text=f"Question {min(completed+1, st.session_state.question_limit)} of {st.session_state.question_limit}")

        if completed >= st.session_state.question_limit:
            st.success("🎉 Interview complete! Open My Progress to review your performance.")
            if st.button("Start New Interview", type="primary"):
                save_current_session()
                st.session_state.started = False
                st.session_state.question = ""
                st.session_state.answers = []
                st.session_state.feedback = []
                st.rerun()
        else:
            if not st.session_state.question:
                with st.spinner("Your AI interviewer is preparing a personalized question..."):
                    st.session_state.question_no = completed
                    st.session_state.question = generate_question()

            st.markdown(f"""
            <div class="card">
              <span class="badge badge-purple">🎙️ AI INTERVIEWER</span>
              <h2 style="margin-top:12px">Question {completed+1}</h2>
              <p style="font-size:1.12rem;line-height:1.65">“{st.session_state.question}”</p>
              <span class="badge badge-amber">Difficulty: {st.session_state.difficulty}</span>
            </div>
            """, unsafe_allow_html=True)

            mode = st.radio("How would you like to answer?", ["⌨️ Type", "🎤 Voice"], horizontal=True)
            answer = ""
            if mode == "⌨️ Type":
                answer = st.text_area("Your answer", height=180, placeholder="Explain your thinking naturally...")
            else:
                st.info("Speak naturally. InterviewAI will transcribe your answer before evaluation.")
                if hasattr(st, "audio_input"):
                    audio = st.audio_input("🎤 Record your answer")
                    if audio:
                        with st.spinner("Transcribing your answer..."):
                            c = client()
                            if c:
                                tr = c.audio.transcriptions.create(
                                    file=("answer.wav", audio.getvalue()),
                                    model=WHISPER_MODEL,
                                    response_format="text",
                                )
                                answer = str(tr)
                                st.session_state.voice_transcript = answer
                        if answer:
                            st.text_area("Transcript — review before submitting", answer, height=160, key="voice_review")
                else:
                    st.warning("Voice recording requires a recent Streamlit version. You can still use text mode.")

            if st.button("Submit Answer →", type="primary", use_container_width=True, disabled=not bool(answer.strip())):
                with st.spinner("AI Coach is evaluating your answer..."):
                    ev = evaluate(answer.strip())
                item = {"question": st.session_state.question, "answer": answer.strip(), "evaluation": ev}
                st.session_state.answers.append(item)
                st.session_state.feedback.append(item)
                st.session_state.last_eval = ev
                st.session_state.question = ""
                st.rerun()

            if st.session_state.last_eval:
                ev = st.session_state.last_eval
                st.markdown('<div class="section"><h2>Latest answer review</h2></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card"><div class="score">{ev.get("overall",0):.1f}/10</div><div class="muted">Overall answer score</div></div>', unsafe_allow_html=True)
                cols = st.columns(6)
                labels = [
                    ("Accuracy","technical_accuracy"),("Completeness","completeness"),
                    ("Depth","depth"),("Communication","communication"),
                    ("Reasoning","problem_solving"),("Role fit","role_relevance")
                ]
                for col,(label,key) in zip(cols,labels):
                    col.metric(label, f'{ev.get(key,0)}/10')
                if ev.get("strengths"):
                    st.markdown("**💚 What you did well**")
                    for x in ev["strengths"]:
                        st.write("✓", x)
                if ev.get("improvements"):
                    st.markdown("**⚠️ What to improve**")
                    for x in ev["improvements"]:
                        st.write("•", x)
                with st.expander("💡 See a stronger model answer"):
                    st.write(ev.get("model_answer",""))

# ============================================================
# VOICE COACH
# ============================================================
elif page == "🎤 Voice Coach":
    st.markdown('<div class="card"><h2>🎤 Voice Coach</h2><p class="muted">Practice speaking clearly and confidently. Record a short answer and receive transcription plus simple communication feedback.</p></div>', unsafe_allow_html=True)
    if hasattr(st, "audio_input"):
        audio = st.audio_input("Record a practice answer")
        if audio:
            c = client()
            if c:
                with st.spinner("Transcribing..."):
                    tr = c.audio.transcriptions.create(
                        file=("practice.wav", audio.getvalue()),
                        model=WHISPER_MODEL,
                        response_format="text",
                    )
                text = str(tr)
                words = re.findall(r"\b[\w'-]+\b", text)
                fillers = re.findall(r"\b(um|uh|like|actually|basically|you know)\b", text.lower())
                st.markdown("### Your transcript")
                st.write(text)
                a,b,c2 = st.columns(3)
                a.metric("Words", len(words))
                c2.metric("Fillers", len(fillers))
                b.metric("Estimated pace", "—")
                st.info("Tip: aim for concise answers with a clear situation, action, reasoning and result.")
            else:
                st.error("Add GROQ_API_KEY in Streamlit Secrets or as an environment variable.")
    else:
        st.warning("Your Streamlit version does not expose the microphone widget. Update Streamlit and redeploy.")

# ============================================================
# PROGRESS
# ============================================================
elif page == "📈 My Progress":
    st.markdown('<div class="card"><h2>📈 My Progress</h2><p class="muted">See how your interview performance changes across sessions.</p></div>', unsafe_allow_html=True)
    h = history()
    if h.empty:
        st.info("No completed sessions yet. Start a mock interview to build your progress history.")
    else:
        st.line_chart(h.sort_values("created_at").set_index("created_at")["score"])
        st.dataframe(h, use_container_width=True, hide_index=True)
        latest = st.session_state.last_eval
        if latest:
            st.markdown("### Latest competency snapshot")
            d = pd.DataFrame({
                "Skill":["Accuracy","Completeness","Depth","Communication","Reasoning","Role Fit"],
                "Score":[latest.get("technical_accuracy",0),latest.get("completeness",0),latest.get("depth",0),latest.get("communication",0),latest.get("problem_solving",0),latest.get("role_relevance",0)]
            }).set_index("Skill")
            st.bar_chart(d)

# ----------------------------- Footer -----------------------------
st.markdown("<hr><div class='small'>InterviewAI • AI-powered interview preparation • Keep candidate information grounded in uploaded documents.</div>", unsafe_allow_html=True)
