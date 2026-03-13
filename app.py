import html
import re
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from streamlit_mic_recorder import mic_recorder
from openai import OpenAI
from agent import run_agent
import tempfile

client = OpenAI()


def _format_assistant_output(text: str) -> str:
    """Convert agent markdown (headings, bullets, bold) to safe HTML for display."""
    if not text:
        return "<p></p>"
    escaped = html.escape(text)
    # Headings: ### Title -> <h4>
    escaped = re.sub(r"^### (.+)$", r"<h4 class=\"out-h4\">\1</h4>", escaped, flags=re.MULTILINE)
    escaped = re.sub(r"^## (.+)$", r"<h4 class=\"out-h4\">\1</h4>", escaped, flags=re.MULTILINE)
    # Bold
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    # Bullet lines: "- item" or "* item" -> <li>
    lines = escaped.split("\n")
    out = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if (stripped.startswith("- ") or stripped.startswith("* ")) and len(stripped) > 2:
            if not in_list:
                out.append("<ul class=\"out-ul\">")
                in_list = True
            out.append("<li>" + stripped[2:].strip() + "</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            if stripped.startswith("<h4"):
                out.append(line)
            elif stripped:
                out.append("<p>" + line + "</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out) if out else "<p>" + escaped + "</p>"


st.set_page_config(
    page_title="AI Business Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Refined UI: clear hierarchy, input card, better chat and sidebar
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stMarkdown { font-family: 'Inter', system-ui, sans-serif !important; }
    
    .main { background: #f0f0f2; }
    .main .block-container { padding: 2rem 2rem 3rem; max-width: 720px; }
    
    /* Header */
    .main h1, .main [data-testid="stMarkdown"] h1 {
        font-size: 1.625rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
        margin: 0 0 0.35rem 0 !important;
        letter-spacing: -0.025em;
        line-height: 1.25 !important;
    }
    .main .stMarkdown p:first-of-type { color: #6b7280 !important; font-size: 0.9rem !important; margin-top: 0 !important; }
    
    /* Section headings */
    .main h2, .main h3 {
        color: #111827 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        margin-top: 1.25rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Chat */
    .msg-row { margin-bottom: 1.5rem; }
    .msg-label {
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #9ca3af;
        margin-bottom: 0.4rem;
    }
    .msg-user {
        background: #111827;
        color: #f9fafb;
        padding: 0.9rem 1.2rem;
        border-radius: 16px 16px 4px 16px;
        max-width: 82%;
        margin-left: auto;
        line-height: 1.55;
        font-size: 0.9rem;
        box-shadow: 0 2px 8px rgba(17, 24, 39, 0.15);
    }
    .msg-assistant {
        background: #ffffff;
        color: #111827;
        padding: 1rem 1.25rem;
        border-radius: 16px 16px 16px 4px;
        max-width: 88%;
        border: 1px solid #e5e7eb;
        line-height: 1.6;
        font-size: 0.9rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .msg-assistant p { margin: 0 0 0.65rem 0 !important; }
    .msg-assistant p:last-child { margin-bottom: 0 !important; }
    .msg-assistant strong { color: #111827; font-weight: 600; }
    .msg-assistant .out-h4 { font-size: 0.8rem; font-weight: 600; color: #374151; margin: 1rem 0 0.4rem 0; }
    .msg-assistant .out-h4:first-child { margin-top: 0; }
    .msg-assistant .out-ul { margin: 0.5rem 0; padding-left: 1.25rem; }
    .msg-assistant .out-ul li { margin: 0.3rem 0; }
    
    .convo-inner { padding: 0; }
    .convo-empty {
        color: #6b7280;
        text-align: center;
        padding: 3rem 1.5rem;
        font-size: 0.9rem;
        line-height: 1.55;
        background: #f9fafb;
        border-radius: 12px;
        border: 1px dashed #e5e7eb;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #f9fafb !important;
        border-right: 1px solid #e5e7eb !important;
    }
    [data-testid="stSidebar"] .stMarkdown { color: #4b5563; font-size: 0.8rem; line-height: 1.55; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        color: #111827 !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.4rem !important;
    }
    [data-testid="stSidebar"] hr { margin: 1rem 0; border-color: #e5e7eb; }
    [data-testid="stSidebar"] button {
        background: #fff !important;
        color: #111827 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
    }
    [data-testid="stSidebar"] button:hover {
        background: #f3f4f6 !important;
        border-color: #d1d5db !important;
    }
    
    /* Form */
    .stTextInput input {
        border-radius: 10px !important;
        border: 1px solid #e5e7eb !important;
        font-size: 0.9rem !important;
    }
    .stTextInput input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }
    .main button {
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stVerticalBlock"] > div { padding: 0.35rem 0; }
    .streamlit-expanderHeader { font-weight: 600; border-radius: 10px; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Header — use native Streamlit so it's always visible
st.title("AI Business Assistant")
st.caption("Ask in voice or text. Answers are grounded in your documents.")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown(
        "Answer questions over your own documents. Ask by **voice** (transcribed with Whisper) or **text**. "
        "Answers are generated from the PDFs in your `data/` folder using a multi-step RAG pipeline."
    )
    st.divider()

    st.subheader("How to use")
    st.markdown("**Voice** — Click Start, speak your question, then Stop. Your speech is sent to Whisper and then to the assistant.")
    st.markdown("**Text** — Type in the box and press Enter, or click one of the example questions to run it.")
    st.markdown("**Behind the Scenes** — After each reply, expand this section to see intent, sub-queries, source documents, and retrieved chunks.")
    st.divider()

    st.subheader("Pipeline")
    st.markdown("1. **Classify** — Detect query intent and context (e.g. sector, quarter).  \n2. **Decompose** — Split complex questions into sub-queries.  \n3. **Retrieve** — Fetch relevant chunks from the document index (FAISS).  \n4. **Answer** — Generate a response from the retrieved context (GPT-4).")
    st.divider()

    st.subheader("Actions")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []
if "suggested_query" not in st.session_state:
    st.session_state.suggested_query = None

SUGGESTED = [
    "Give me an overview of Q2 sales.",
    "Compare SaaS vs FMCG performance.",
    "What are the main insights from the documents?",
]

# Input section
st.subheader("Ask something")
st.caption("Voice or text — use one. Example questions run with one click.")

col1, col2 = st.columns([1, 2])
with col1:
    audio = mic_recorder(start_prompt="Start", stop_prompt="Stop", key="recorder")
with col2:
    user_query = st.text_input(
        "Query",
        placeholder="e.g. Give me an overview of Q2 sales in SaaS",
        label_visibility="collapsed",
        key="text_query",
    )

st.caption("Example questions — click to run")
sug_cols = st.columns(3)
for i, prompt in enumerate(SUGGESTED):
    with sug_cols[i]:
        if st.button(prompt, key=f"suggest_{i}", use_container_width=True):
            st.session_state.suggested_query = prompt
            st.rerun()

# Resolve input
query_text = None
if st.session_state.suggested_query:
    query_text = st.session_state.suggested_query
    st.session_state.suggested_query = None
elif audio and audio.get("bytes"):
    with st.status("Transcribing…", state="running") as status:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio["bytes"])
            tmp.flush()
            with open(tmp.name, "rb") as f:
                transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
        query_text = transcript.text
        status.update(label="Done", state="complete")
elif user_query and user_query.strip():
    query_text = user_query.strip()

# Run agent
if query_text:
    st.session_state.history.append({"role": "user", "content": query_text})
    with st.spinner("Searching documents and generating answer…"):
        result = run_agent(query_text)
    st.session_state.history.append({"role": "assistant", "content": result["answer"], "details": result})

    if st.session_state.history[-1].get("details"):
        d = st.session_state.history[-1]["details"]
        with st.expander("Behind the Scenes (this reply)"):
            tab1, tab2, tab3 = st.tabs(["Intent", "Sub-queries & docs", "Chunks"])
            with tab1:
                st.json(d.get("intent_info", {}))
            with tab2:
                st.markdown("**Sub-queries**")
                st.write(d.get("sub_queries", []))
                st.markdown("**Documents used**")
                st.write(", ".join(d.get("retrieved_docs", [])) or "—")
            with tab3:
                for i, chunk in enumerate(d.get("retrieved_chunks", []), 1):
                    with st.expander(f"Chunk {i}"):
                        st.markdown(chunk)

# Conversation — one markdown block for entire content (no split divs = no white blocks)
st.divider()
st.subheader("Conversation")

if not st.session_state.history:
    st.markdown(
        '<div class="convo-empty">No messages yet. Use voice or text above, or click an example question.</div>',
        unsafe_allow_html=True,
    )
else:
    # Build one HTML string so one block only
    parts = []
    for msg in st.session_state.history:
        if msg["role"] == "user":
            safe = html.escape(msg["content"]).replace("\n", "<br>")
            parts.append(f'<div class="msg-row"><div class="msg-label">You</div><div class="msg-user">{safe}</div></div>')
        else:
            content = _format_assistant_output(msg["content"])
            parts.append(f'<div class="msg-row"><div class="msg-label">Assistant</div><div class="msg-assistant">{content}</div></div>')
    st.markdown('<div class="convo-inner">' + "".join(parts) + "</div>", unsafe_allow_html=True)
