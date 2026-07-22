import streamlit as st
from src.loader import load_pdf
from src.splitter import split_documents
from src.vectorstore import create_vectorstore
from src.rag_chain import ask_question

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS — the visual upgrade
# -----------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(118, 75, 162, 0.25);
    }
    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: white;
        margin: 0;
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin-top: 0.4rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #f0f0f5 !important;
    }
    section[data-testid="stSidebar"] .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        transition: transform 0.15s ease;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        transform: scale(1.03);
    }

    /* Status badges */
    .status-badge {
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .status-ready {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
    }
    .status-empty {
        background: rgba(255,255,255,0.08);
        color: #aaa;
        border: 1px solid rgba(255,255,255,0.15);
    }

    /* Chat bubbles */
    .stChatMessage {
        border-radius: 16px;
        padding: 0.3rem;
        animation: fadeIn 0.35s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Source expander */
    .stExpander {
        border-radius: 10px !important;
        border: 1px solid rgba(118, 75, 162, 0.2) !important;
    }

    /* Empty state card */
    .empty-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.08), rgba(118,75,162,0.08));
        border: 1px dashed rgba(118,75,162,0.35);
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .empty-card h3 {
        font-family: 'Poppins', sans-serif;
        color: #764ba2;
    }

    /* Metric pills in sidebar */
    .pill-row {
        display: flex;
        gap: 8px;
        margin-top: 10px;
        flex-wrap: wrap;
    }
    .pill {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 10px;
        padding: 6px 10px;
        font-size: 0.78rem;
        text-align: center;
        flex: 1;
    }
    .pill-num {
        font-size: 1.1rem;
        font-weight: 700;
        display: block;
        background: linear-gradient(135deg, #667eea, #38ef7d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Hero Header
# -----------------------------
st.markdown("""
    <div class="hero">
        <p class="hero-title">🧠 DocMind AI</p>
        <p class="hero-subtitle">Upload any document and have a real conversation with it — grounded answers, real citations, zero guesswork.</p>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# Session State
# -----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("### 📂 Your Document")

    uploaded_file = st.file_uploader(
        "Drop a PDF here",
        type="pdf",
        label_visibility="collapsed"
    )

    if uploaded_file and uploaded_file.name != st.session_state.processed_file:
        with open(f"docs/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.read())

        progress = st.progress(0, text="📖 Reading document...")
        docs = load_pdf(f"docs/{uploaded_file.name}")
        progress.progress(35, text="✂️ Splitting into chunks...")

        chunks = split_documents(docs)
        progress.progress(70, text="🧠 Creating embeddings...")

        vectorstore = create_vectorstore(chunks)
        progress.progress(100, text="✅ Ready!")

        st.session_state.vectorstore = vectorstore
        st.session_state.processed_file = uploaded_file.name
        st.session_state.chunk_count = len(chunks)
        st.balloons()

    st.markdown("---")

    if st.session_state.vectorstore is not None:
        st.markdown(
            f'<span class="status-badge status-ready">🟢 {st.session_state.processed_file}</span>',
            unsafe_allow_html=True
        )
        st.markdown(f"""
            <div class="pill-row">
                <div class="pill"><span class="pill-num">{st.session_state.chunk_count}</span>chunks</div>
                <div class="pill"><span class="pill-num">{len(st.session_state.messages)}</span>messages</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<span class="status-badge status-empty">⚪ No document loaded</span>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.processed_file = None
            st.session_state.messages = []
            st.session_state.chunk_count = 0
            st.rerun()

    with st.expander("ℹ️ How it works"):
        st.caption(
            "Your document is split into chunks and converted into embeddings. "
            "When you ask a question, the most relevant chunks are retrieved "
            "and passed to the AI to generate a grounded answer with page citations."
        )

# -----------------------------
# Main Chat Area
# -----------------------------
if not st.session_state.messages:
    if st.session_state.vectorstore:
        st.markdown("""
            <div class="empty-card">
                <h3>💬 Ready when you are</h3>
                <p style="color:#888;">Your document is indexed — ask anything below.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="empty-card">
                <h3>📤 Upload a PDF to get started</h3>
                <p style="color:#888;">Use the sidebar to add your document — then ask it anything.</p>
            </div>
        """, unsafe_allow_html=True)

for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🧠"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 View Sources"):
                st.caption(message["sources"])

# -----------------------------
# Chat Input
# -----------------------------
question = st.chat_input("Ask something about your document...")

if question:
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please upload a PDF first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Thinking..."):
            answer, docs = ask_question(st.session_state.vectorstore, question)
            pages = sorted(set(doc.metadata["page"] + 1 for doc in docs))
            sources = ", ".join(f"Page {p}" for p in pages)

        st.markdown(answer)
        with st.expander("📄 View Sources"):
            st.caption(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })