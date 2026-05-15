"""
Research Synthesis & Comparison Engine
A RAG-powered application for cross-document analysis of research papers.
"""

# Streamlit Cloud SQLite workaround for ChromaDB
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import streamlit as st # Triggering reload
from dotenv import load_dotenv
from pdf_processor import extract_text_from_pdf, create_chunks_with_metadata, get_paper_label
from vector_store import VectorStoreManager
from embeddings import EmbeddingManager
from synthesis_engine import SynthesisEngine

load_dotenv()

# ─── Page Config ───
st.set_page_config(
    page_title="Research Synthesis Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #6C63FF 0%, #3B82F6 50%, #06B6D4 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.4rem; font-weight: 700; text-align: center; margin-bottom: 0.2rem;
}
.sub-header {
    text-align: center; color: #9CA3AF; font-size: 1rem; margin-bottom: 2rem;
}
.paper-card {
    background: linear-gradient(135deg, #1E1E2E 0%, #2D2D44 100%);
    border: 1px solid #3B3B5C; border-radius: 12px; padding: 1.2rem;
    margin: 0.5rem 0; transition: transform 0.2s, box-shadow 0.2s;
}
.paper-card:hover {
    transform: translateY(-2px); box-shadow: 0 8px 25px rgba(108,99,255,0.2);
}
.paper-label {
    display: inline-block; padding: 0.2rem 0.7rem; border-radius: 20px;
    font-weight: 600; font-size: 0.8rem; margin-bottom: 0.5rem;
}
.label-a { background: #6C63FF33; color: #6C63FF; }
.label-b { background: #3B82F633; color: #3B82F6; }
.label-c { background: #06B6D433; color: #06B6D4; }
.label-d { background: #F59E0B33; color: #F59E0B; }
.label-e { background: #EF444433; color: #EF4444; }
.stat-box {
    background: #1A1D2E; border: 1px solid #2D2D44; border-radius: 10px;
    padding: 1rem; text-align: center;
}
.stat-num { font-size: 1.8rem; font-weight: 700; color: #6C63FF; }
.stat-label { font-size: 0.8rem; color: #9CA3AF; }
.mode-btn {
    background: #1E1E2E; border: 1px solid #3B3B5C; border-radius: 10px;
    padding: 1rem; text-align: center; cursor: pointer; transition: all 0.2s;
}
.mode-btn:hover { border-color: #6C63FF; background: #6C63FF11; }
.citation-tag {
    display: inline-block; padding: 0.15rem 0.5rem; border-radius: 12px;
    font-size: 0.7rem; font-weight: 600; margin: 0.1rem;
    background: #6C63FF22; color: #6C63FF; border: 1px solid #6C63FF44;
}
div[data-testid="stChatMessage"] { background: #1A1D2E; border-radius: 12px; margin: 0.5rem 0; }
.stChatInput > div { border: 1px solid #3B3B5C !important; border-radius: 12px !important; }
section[data-testid="stSidebar"] { background: #0D0D1A; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "api_key": os.getenv("GOOGLE_API_KEY", ""),
        "papers": {},          # {paper_id: {metadata, text, pages_text}}
        "paper_counter": 0,
        "vector_store": None,
        "embedding_manager": None,
        "synthesis_engine": None,
        "chat_history": [],
        "current_mode": "chat",
        "initialized": False,
        "processing": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def initialize_engines(api_key: str):
    """Initialize the embedding manager, vector store, and synthesis engine."""
    if not api_key:
        st.error("⚠️ Please provide a Google Gemini API key.")
        return False
    try:
        st.session_state.embedding_manager = EmbeddingManager(api_key)
        st.session_state.vector_store = VectorStoreManager("./chroma_db")
        st.session_state.synthesis_engine = SynthesisEngine(api_key)
        st.session_state.initialized = True
        return True
    except Exception as e:
        st.error(f"❌ Initialization failed: {e}")
        return False


def process_uploaded_pdf(pdf_file):
    """Process a single uploaded PDF file."""
    paper_id = get_paper_label(st.session_state.paper_counter)
    st.session_state.paper_counter += 1

    with st.spinner(f"📄 Extracting text from {pdf_file.name}..."):
        full_text, metadata, pages_text = extract_text_from_pdf(pdf_file)

    with st.spinner(f"✂️ Chunking {paper_id}..."):
        chunks = create_chunks_with_metadata(full_text, pages_text, metadata, paper_id)

    with st.spinner(f"🧮 Generating embeddings for {paper_id} ({len(chunks)} chunks)..."):
        texts = [c["text"] for c in chunks]
        embeddings = st.session_state.embedding_manager.embed_texts(texts)

    with st.spinner(f"💾 Storing {paper_id} in vector database..."):
        st.session_state.vector_store.add_chunks(chunks, embeddings)

    st.session_state.papers[paper_id] = {
        "metadata": metadata,
        "text": full_text,
        "pages_text": pages_text,
        "chunk_count": len(chunks),
    }
    return paper_id


def get_label_class(idx: int) -> str:
    classes = ["label-a", "label-b", "label-c", "label-d", "label-e"]
    return classes[idx % len(classes)]


def render_sidebar():
    """Render the sidebar with config and paper management."""
    with st.sidebar:
        st.markdown('<p class="main-header" style="font-size:1.5rem;">🔬 Research Engine</p>', unsafe_allow_html=True)
        st.markdown("---")

        # Auto-initialize if API key is present
        if not st.session_state.initialized and st.session_state.api_key:
            initialize_engines(st.session_state.api_key)

        if st.session_state.initialized:
            st.success("✅ Engine Ready")
        else:
            st.error("⚠️ GOOGLE_API_KEY not found. Please add it to your environment variables.")

        st.markdown("---")
        st.markdown("### 📚 Upload Papers")
        uploaded_files = st.file_uploader(
            "Upload PDFs (up to 5)", type=["pdf"],
            accept_multiple_files=True, key="pdf_uploader",
            help="Upload 3-5 research papers on a related topic",
        )

        if uploaded_files and st.session_state.initialized:
            new_files = [f for f in uploaded_files
                         if f.name not in [p["metadata"]["filename"] for p in st.session_state.papers.values()]]
            if new_files and st.button(f"📥 Process {len(new_files)} New Paper(s)", use_container_width=True):
                for f in new_files:
                    if len(st.session_state.papers) >= 5:
                        st.warning("Maximum 5 papers allowed.")
                        break
                    pid = process_uploaded_pdf(f)
                    st.success(f"✅ {pid}: {f.name}")
                st.rerun()

        # Show loaded papers
        if st.session_state.papers:
            st.markdown("---")
            st.markdown("### 📋 Loaded Papers")
            for idx, (pid, info) in enumerate(st.session_state.papers.items()):
                lbl = get_label_class(idx)
                st.markdown(f"""<div class="paper-card">
                    <span class="paper-label {lbl}">{pid}</span>
                    <div style="font-weight:600;font-size:0.9rem;">{info['metadata']['title']}</div>
                    <div style="color:#9CA3AF;font-size:0.75rem;">
                        👤 {info['metadata']['author']} · 📄 {info['metadata']['num_pages']} pages · 🧩 {info['chunk_count']} chunks
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            if st.button("🗑️ Clear All Papers", use_container_width=True, type="secondary"):
                st.session_state.vector_store.clear_collection()
                st.session_state.papers = {}
                st.session_state.paper_counter = 0
                st.session_state.chat_history = []
                st.rerun()


def render_home():
    """Render the home / landing view."""
    st.markdown('<p class="main-header">🔬 Research Synthesis & Comparison Engine</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload research papers and unlock cross-document insights powered by Gemini AI</p>', unsafe_allow_html=True)

    # Stats row
    num_papers = len(st.session_state.papers)
    num_chunks = sum(p["chunk_count"] for p in st.session_state.papers.values()) if st.session_state.papers else 0
    c1, c2, c3 = st.columns(3)
    for col, num, label in [(c1, num_papers, "Papers Loaded"), (c2, num_chunks, "Total Chunks"), (c3, len(st.session_state.chat_history) // 2, "Conversations")]:
        with col:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{num}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if num_papers == 0:
        if not st.session_state.api_key:
            st.error("⚠️ Please configure your GOOGLE_API_KEY in the environment variables to get started.")
        else:
            st.info("👈 **Get started**: Upload 2–5 research papers using the sidebar.")
        return

    # Mode selection
    st.markdown("### 🎯 Choose Analysis Mode")
    m1, m2, m3, m4 = st.columns(4)
    modes = [
        (m1, "chat", "💬", "Research Q&A", "Ask questions with paper-wise citations"),
        (m2, "summary", "📊", "Cross-Doc Summary", "Unified summary across all papers"),
        (m3, "compare", "⚖️", "Compare Findings", "Compare specific aspects between papers"),
        (m4, "methodology", "🔬", "Methodology Synthesis", "Deep-dive into technical methods"),
    ]
    for col, mode, icon, title, desc in modes:
        with col:
            if st.button(f"{icon} {title}", key=f"mode_{mode}", use_container_width=True,
                         help=desc, type="primary" if st.session_state.current_mode == mode else "secondary"):
                st.session_state.current_mode = mode
                st.rerun()


def render_chat_mode():
    """Render the interactive Q&A chat mode."""
    st.markdown("### 💬 Research Q&A with Citations")
    st.caption("Ask any question — the engine will retrieve relevant passages and cite each paper.")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧑‍🔬" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your research papers..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🔬"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 Searching across papers & synthesizing..."):
                # Get query embedding
                query_emb = st.session_state.embedding_manager.embed_query(prompt)
                paper_ids = list(st.session_state.papers.keys())
                grouped = st.session_state.vector_store.query_multiple_papers(query_emb, paper_ids, n_per_paper=3)
                paper_info = [{"paper_id": pid, **info["metadata"]} for pid, info in st.session_state.papers.items()]
                response = st.session_state.synthesis_engine.answer_with_citations(
                    prompt, grouped, paper_info, st.session_state.chat_history
                )
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})


def render_summary_mode():
    """Render the cross-document summary mode."""
    st.markdown("### 📊 Cross-Document Summary")
    st.caption("Generate a unified synthesis that covers core themes of all uploaded papers.")

    if st.button("🚀 Generate Cross-Document Summary", use_container_width=True, type="primary"):
        with st.spinner("📝 Analyzing all papers and synthesizing... (this may take a moment)"):
            all_contexts = {pid: info["text"] for pid, info in st.session_state.papers.items()}
            paper_info = [{"paper_id": pid, **info["metadata"]} for pid, info in st.session_state.papers.items()]
            summary = st.session_state.synthesis_engine.generate_cross_document_summary(all_contexts, paper_info)
        st.markdown(summary)

    # Individual paper summaries
    st.markdown("---")
    st.markdown("### 📄 Individual Paper Summaries")
    for pid, info in st.session_state.papers.items():
        with st.expander(f"📄 {pid}: {info['metadata']['title']}"):
            if st.button(f"Generate Summary for {pid}", key=f"sum_{pid}"):
                with st.spinner(f"Summarizing {pid}..."):
                    summary = st.session_state.synthesis_engine.generate_paper_summary(
                        info["text"], {"paper_id": pid, **info["metadata"]}
                    )
                st.markdown(summary)


def render_compare_mode():
    """Render the findings comparison mode."""
    st.markdown("### ⚖️ Compare Findings Across Papers")
    st.caption("Enter an aspect or topic to compare how different papers address it.")

    aspect = st.text_input("🔍 What aspect do you want to compare?",
                           placeholder="e.g., model accuracy, energy efficiency, training cost...")

    if aspect and st.button("⚖️ Compare Now", type="primary", use_container_width=True):
        with st.spinner("🔍 Retrieving and comparing..."):
            query_emb = st.session_state.embedding_manager.embed_query(aspect)
            paper_ids = list(st.session_state.papers.keys())
            grouped = st.session_state.vector_store.query_multiple_papers(query_emb, paper_ids, n_per_paper=4)
            paper_info = [{"paper_id": pid, **info["metadata"]} for pid, info in st.session_state.papers.items()]
            comparison = st.session_state.synthesis_engine.compare_findings(grouped, paper_info, aspect)
        st.markdown(comparison)


def render_methodology_mode():
    """Render the methodology synthesis mode."""
    st.markdown("### 🔬 Technical Methodology Synthesis")
    st.caption("Deep-dive into the technical methods used across your papers.")

    topic = st.text_input("🔍 Methodology topic to synthesize",
                          placeholder="e.g., transformer architecture, data augmentation, evaluation metrics...")

    if topic and st.button("🔬 Synthesize Methodologies", type="primary", use_container_width=True):
        with st.spinner("🔍 Extracting methodology details across papers..."):
            query = f"methodology approach method technique {topic}"
            query_emb = st.session_state.embedding_manager.embed_query(query)
            paper_ids = list(st.session_state.papers.keys())
            grouped = st.session_state.vector_store.query_multiple_papers(query_emb, paper_ids, n_per_paper=4)
            paper_info = [{"paper_id": pid, **info["metadata"]} for pid, info in st.session_state.papers.items()]
            synthesis = st.session_state.synthesis_engine.synthesize_methodology(grouped, paper_info, topic)
        st.markdown(synthesis)


# ─── MAIN ───
def main():
    init_session_state()
    render_sidebar()
    render_home()

    if not st.session_state.papers:
        return

    st.markdown("---")

    mode = st.session_state.current_mode
    if mode == "chat":
        render_chat_mode()
    elif mode == "summary":
        render_summary_mode()
    elif mode == "compare":
        render_compare_mode()
    elif mode == "methodology":
        render_methodology_mode()


if __name__ == "__main__":
    main()
