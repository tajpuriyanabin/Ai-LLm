import os
import json
import time
import hashlib
import streamlit as st
import pymupdf as fitz

import config
from rag_pipeline import (
    get_embedding_function,
    extract_documents,
    chunk_documents,
    create_session_vector_store,
    load_session_vector_store,
    delete_session_vector_store,
    build_hybrid_retriever,
    LocalHybridRetriever,
    rerank_documents,
    contextualize_query,
    format_docs,
    generate_study_tool,
    RAG_PROMPT_TEMPLATE,
)

# ---------------------------------------------------------------------------
# 1. Page Configuration & Custom Styling (Hidden Input Instructions)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Offline RAG & Study Hub", page_icon="🎓", layout="wide")

st.markdown(
    """
    <style>
    .stApp { 
        background-color: #0e1117; 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; 
    }

    /* 1. HIDE 'Press Enter to apply' helper instructions */
    [data-testid="InputInstructions"] {
        display: none !important;
    }
    .st-emotion-cache-121p9b2 {
        display: none !important;
    }

    /* 2. Centered Main Page Container */
    .main .block-container {
        max-width: 1050px;
        margin: 0 auto;
        padding-top: 1.2rem;
        padding-bottom: 4rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    /* 3. Centered Hero Header */
    .hero-wrapper {
        text-align: center;
        padding: 0.5rem 0 0.8rem 0;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        font-size: 2.2rem; 
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        text-align: center;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.98rem;
        max-width: 720px;
        margin: 0 auto 1.2rem auto;
        line-height: 1.5;
        text-align: center;
    }

    /* 4. Feature Cards */
    .feature-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.1rem 0.9rem;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        min-height: 135px;
    }
    .feature-icon { font-size: 1.5rem; margin-bottom: 0.3rem; }
    .feature-heading { font-weight: 700; color: #f1f5f9; font-size: 0.95rem; margin-bottom: 0.2rem; }
    .feature-desc { color: #94a3b8; font-size: 0.8rem; line-height: 1.4; }

    /* 5. Centered Tabs */
    .stTabs { width: 100% !important; }
    .stTabs [data-baseweb="tab-list"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: fit-content !important;
        margin: 0 auto 1.8rem auto !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
        gap: 1.2rem !important;
        padding: 0.3rem 0.5rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.3rem !important;
        border-radius: 8px !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.15) !important;
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* 6. Citations & Badges */
    .citation-badge {
        display: inline-block; 
        padding: 0.2rem 0.55rem; 
        border-radius: 9999px;
        font-size: 0.75rem; 
        font-weight: 600; 
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8; 
        border: 1px solid rgba(56, 189, 248, 0.3); 
        margin-bottom: 0.3rem;
    }
    .source-card {
        background-color: #1e293b; 
        border-radius: 6px; 
        padding: 0.75rem;
        border-left: 3px solid #38bdf8; 
        font-size: 0.85rem; 
        color: #cbd5e1;
        margin-top: 0.3rem; 
        margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 2. Authentication & User Database (Hashing & Reset Support)
# ---------------------------------------------------------------------------
AUTH_FILE = "users_auth.json"
BASE_USER_DIR = "./user_data"

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def load_auth_db() -> dict:
    if os.path.exists(AUTH_FILE):
        try:
            with open(AUTH_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_auth_db(users: dict):
    try:
        with open(AUTH_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def register_user(username: str, password: str, confirm_pass: str, display_name: str) -> tuple:
    clean_user = username.strip().lower()
    if not clean_user or not password:
        return False, "Username and password cannot be empty."
    if len(password) < 4:
        return False, "Password must be at least 4 characters long."
    if password != confirm_pass:
        return False, "Passwords do not match. Please re-enter."
    
    users = load_auth_db()
    if clean_user in users:
        return False, "Username already exists. Please choose a different username."
    
    users[clean_user] = {
        "display_name": display_name.strip() if display_name.strip() else clean_user,
        "password_hash": hash_password(password),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_auth_db(users)
    return True, "Account created successfully! You can now sign in."

def authenticate_user(username: str, password: str) -> tuple:
    users = load_auth_db()
    clean_user = username.strip().lower()
    if clean_user not in users:
        return False, "User does not exist."
    if users[clean_user]["password_hash"] == hash_password(password):
        return True, users[clean_user].get("display_name", clean_user)
    return False, "Incorrect password. Please try again."

def reset_password(username: str, new_pass: str, confirm_pass: str) -> tuple:
    clean_user = username.strip().lower()
    if not clean_user or not new_pass:
        return False, "Username and password cannot be empty."
    if len(new_pass) < 4:
        return False, "New password must be at least 4 characters long."
    if new_pass != confirm_pass:
        return False, "Passwords do not match."
    
    users = load_auth_db()
    if clean_user not in users:
        return False, "Username not found."
    
    users[clean_user]["password_hash"] = hash_password(new_pass)
    save_auth_db(users)
    return True, "Password has been successfully updated! You can now sign in."

# State Initialization
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "display_name" not in st.session_state:
    st.session_state.display_name = ""

# ---------------------------------------------------------------------------
# 3. Login / Registration / Forgot Password Screen
# ---------------------------------------------------------------------------
if not st.session_state.authenticated:
    st.markdown(
        """
        <div class="hero-wrapper" style="margin-top: 1.5rem;">
            <div class="hero-title">🎓 Offline RAG & Interactive Study Hub</div>
            <div class="hero-subtitle">100% private, local document intelligence with personal user workspaces.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    auth_tab1, auth_tab2, auth_tab3, auth_tab4 = st.tabs(["🔐 Sign In", "✨ Create Account", "🔑 Forgot Password?", "👤 Guest Mode"])

    # 1. Sign In Form
    with auth_tab1:
        st.markdown("### Sign In to Your Workspace")
        with st.form("signin_form", clear_on_submit=False):
            login_user = st.text_input("Username:")
            login_pass = st.text_input("Password:", type="password")
            submit_login = st.form_submit_button("🚀 Sign In", use_container_width=True, type="primary")
            
            if submit_login:
                success, msg = authenticate_user(login_user, login_pass)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.username = login_user.strip().lower()
                    st.session_state.display_name = msg
                    st.success(f"Welcome back, {msg}!")
                    time.sleep(0.3)
                    st.rerun()
                else:
                    st.error(msg)

    # 2. Create Account with Confirm Password
    with auth_tab2:
        st.markdown("### Create an Account")
        with st.form("signup_form", clear_on_submit=False):
            reg_name = st.text_input("Full Name:")
            reg_user = st.text_input("Choose Username:")
            reg_pass = st.text_input("Password (min 4 chars):", type="password")
            reg_confirm = st.text_input("Confirm Password:", type="password")
            submit_reg = st.form_submit_button("✨ Register Account", use_container_width=True, type="primary")
            
            if submit_reg:
                success, msg = register_user(reg_user, reg_pass, reg_confirm, reg_name)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

    # 3. Forgot Password
    with auth_tab3:
        st.markdown("### Reset Your Password")
        with st.form("reset_pass_form", clear_on_submit=False):
            reset_user = st.text_input("Registered Username:")
            new_pass = st.text_input("New Password:", type="password")
            confirm_new_pass = st.text_input("Confirm New Password:", type="password")
            submit_reset = st.form_submit_button("🔄 Update Password", use_container_width=True, type="primary")
            
            if submit_reset:
                success, msg = reset_password(reset_user, new_pass, confirm_new_pass)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

    # 4. Guest Mode
    with auth_tab4:
        st.markdown("### Quick Guest Access")
        st.caption("Access the application immediately as a temporary guest without registering.")
        if st.button("👤 Enter as Guest", use_container_width=True):
            st.session_state.authenticated = True
            st.session_state.username = "guest"
            st.session_state.display_name = "Guest User"
            st.rerun()

    st.stop()

# ---------------------------------------------------------------------------
# 4. Authenticated Application Logic (Per-User Storage)
# ---------------------------------------------------------------------------
current_user = st.session_state.username
current_display = st.session_state.display_name

def get_user_session_file(username: str) -> str:
    safe_user = "".join([c if c.isalnum() else "_" for c in str(username)]).lower()
    user_folder = os.path.join(BASE_USER_DIR, safe_user)
    os.makedirs(user_folder, exist_ok=True)
    return os.path.join(user_folder, "chat_sessions.json")

def load_user_sessions(username: str) -> dict:
    file_path = get_user_session_file(username)
    default_structure = {
        "session_1": {
            "title": "General Chat",
            "messages": [],
            "indexed_files_info": None,
        }
    }
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and data:
                    migrated = {}
                    for k, v in data.items():
                        if isinstance(v, list):
                            migrated[k] = {"title": k, "messages": v, "indexed_files_info": None}
                        elif isinstance(v, dict):
                            migrated[k] = {
                                "title": v.get("title", k),
                                "messages": v.get("messages", []),
                                "indexed_files_info": v.get("indexed_files_info", None),
                            }
                    if migrated:
                        return migrated
        except Exception:
            pass
    return default_structure

def save_user_sessions(username: str, sessions: dict):
    try:
        file_path = get_user_session_file(username)
        clean_data = {}
        for s_id, s_obj in sessions.items():
            if not isinstance(s_obj, dict):
                s_obj = {"title": str(s_id), "messages": s_obj if isinstance(s_obj, list) else [], "indexed_files_info": None}
            
            clean_msgs = []
            for m in s_obj.get("messages", []):
                clean_srcs = []
                for src in m.get("sources", []):
                    if isinstance(src, dict):
                        clean_srcs.append({
                            "source": str(src.get("source", "Document")),
                            "page": str(src.get("page", "N/A")),
                            "content": str(src.get("content", "")),
                        })
                clean_msgs.append({
                    "role": str(m["role"]),
                    "content": str(m["content"]),
                    "sources": clean_srcs,
                })
            clean_data[s_id] = {
                "title": str(s_obj.get("title", "Chat")),
                "messages": clean_msgs,
                "indexed_files_info": s_obj.get("indexed_files_info", None),
            }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(clean_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("Error saving sessions:", e)

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return get_embedding_function()

if "user_sessions_cache" not in st.session_state or st.session_state.get("cached_user") != current_user:
    st.session_state.user_sessions_cache = load_user_sessions(current_user)
    st.session_state.cached_user = current_user
    st.session_state.active_session_id = list(st.session_state.user_sessions_cache.keys())[0]

current_sessions = st.session_state.user_sessions_cache

if "active_session_id" not in st.session_state or st.session_state.active_session_id not in current_sessions:
    st.session_state.active_session_id = list(current_sessions.keys())[0]

active_id = st.session_state.active_session_id
active_session_data = current_sessions.get(active_id, {"title": "General Chat", "messages": [], "indexed_files_info": None})

embeddings = load_embeddings()
try:
    session_vector_store = load_session_vector_store(active_id, embeddings, user_id=current_user)
except Exception:
    session_vector_store = load_session_vector_store(active_id, embeddings)

# ---------------------------------------------------------------------------
# 5. Sidebar UI (User Account Header & Workspaces)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 👤 {current_display}")
    st.caption(f"Signed in as `{current_user}`")
    
    if st.button("🚪 Log Out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.display_name = ""
        st.session_state.pop("user_sessions_cache", None)
        st.rerun()

    st.markdown("---")
    st.markdown("### 💬 Chat Workspaces")
    
    if st.button("➕ New Chat Workspace", use_container_width=True, type="primary"):
        new_id = f"session_{int(time.time())}"
        current_sessions[new_id] = {
            "title": f"Chat {len(current_sessions) + 1}",
            "messages": [],
            "indexed_files_info": None,
        }
        st.session_state.active_session_id = new_id
        save_user_sessions(current_user, current_sessions)
        st.rerun()

    for s_id, s_obj in list(current_sessions.items()):
        is_active = (s_id == active_id)
        s_title = s_obj.get("title", s_id) if isinstance(s_obj, dict) else str(s_id)
        btn_label = f"👉 {s_title}" if is_active else f"💬 {s_title}"
        
        col_title, col_trash = st.columns(2)
        with col_title:
            if st.button(btn_label, key=f"btn_{current_user}_{s_id}", use_container_width=True):
                st.session_state.active_session_id = s_id
                st.rerun()
        with col_trash:
            if st.button("🗑️", key=f"del_{current_user}_{s_id}", help=f"Delete {s_title}"):
                try:
                    delete_session_vector_store(s_id, user_id=current_user, embeddings=embeddings)
                except Exception:
                    delete_session_vector_store(s_id, embeddings=embeddings)
                del current_sessions[s_id]
                if not current_sessions:
                    first_id = f"session_{int(time.time())}"
                    current_sessions[first_id] = {
                        "title": "General Chat",
                        "messages": [],
                        "indexed_files_info": None,
                    }
                st.session_state.active_session_id = list(current_sessions.keys())[0]
                save_user_sessions(current_user, current_sessions)
                st.rerun()

    st.markdown("---")
    st.markdown(f"### 📂 Documents for `{active_session_data.get('title', 'Chat')}`")

    uploaded_files = st.file_uploader(
        f"Upload Files for {current_display}",
        type=["pdf", "docx", "pptx", "txt", "md", "csv", "xlsx", "py", "cpp"],
        accept_multiple_files=True,
        key=f"uploader_{current_user}_{active_id}",
    )

    start_page, end_page, enable_ocr = 1, None, True
    if uploaded_files:
        pdf_files = [f for f in uploaded_files if f.name.lower().endswith(".pdf")]
        if pdf_files:
            try:
                pdf_files[0].seek(0)
                temp_doc = fitz.open(stream=pdf_files[0].read(), filetype="pdf")
                total_pdf_pages = len(temp_doc)
                temp_doc.close()
                pdf_files[0].seek(0)
                st.caption(f"📄 Detected: **{total_pdf_pages} pages**")

                with st.expander("⚙️ Speed & Page Range Controls", expanded=True):
                    enable_ocr = st.checkbox("Enable OCR for Scanned Pages", value=True, key=f"ocr_{current_user}_{active_id}")
                    process_all = st.checkbox("Process Entire Document", value=False, key=f"all_{current_user}_{active_id}")
                    if not process_all:
                        max_r = min(20, total_pdf_pages)
                        page_range = st.slider("Select Page Range:", 1, total_pdf_pages, (1, max_r), key=f"range_{current_user}_{active_id}")
                        start_page, end_page = page_range
                    else:
                        start_page, end_page = 1, total_pdf_pages
            except Exception:
                pass

    if st.button("🚀 Process & Index for this Chat", use_container_width=True, type="primary", key=f"idx_btn_{current_user}_{active_id}"):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            progress_bar = st.progress(0, text="Starting Multi-Format Ingestion (0%)...")

            def update_progress(msg: str, progress: float):
                clamped = min(max(progress, 0.0), 1.0)
                progress_bar.progress(clamped, text=f"📊 {int(clamped * 100)}% — {msg}")

            try:
                raw_docs = extract_documents(
                    uploaded_files,
                    start_page=start_page,
                    end_page=end_page,
                    enable_ocr=enable_ocr,
                    progress_callback=update_progress,
                )

                if not raw_docs:
                    progress_bar.empty()
                    st.error("No extractable text found in uploaded files.")
                else:
                    update_progress("Splitting text into semantic chunks...", 0.82)
                    chunks = chunk_documents(raw_docs)
                    
                    update_progress("Generating embeddings & ChromaDB index...", 0.90)
                    emb = load_embeddings()
                    try:
                        session_vector_store = create_session_vector_store(active_id, chunks, emb, user_id=current_user)
                    except Exception:
                        session_vector_store = create_session_vector_store(active_id, chunks, emb)

                    active_session_data["indexed_files_info"] = {
                        "file_count": len(uploaded_files),
                        "chunk_count": len(chunks),
                        "page_range": f"Pages {start_page}–{end_page}" if end_page else "All content",
                        "file_names": [f.name for f in uploaded_files],
                    }
                    save_user_sessions(current_user, current_sessions)

                    progress_bar.progress(1.0, text="✅ Indexing Complete!")
                    st.success(f"Indexed {len(chunks)} chunks for '{current_display}' successfully!")
                    time.sleep(0.5)
                    st.rerun()
            except Exception as e:
                progress_bar.empty()
                st.error(f"Indexing Error: {str(e)}")

    if active_session_data.get("indexed_files_info"):
        info = active_session_data["indexed_files_info"]
        st.markdown("---")
        st.markdown("### 📊 Workspace Index")
        st.info(
            f"**Files:** {info['file_count']}\n\n"
            f"**Scope:** {info.get('page_range', 'All')}\n\n"
            f"**Total Chunks:** {info['chunk_count']}\n\n"
            + "\n".join([f"- `{n}`" for n in info['file_names']])
        )

    st.markdown("---")
    col_clear, col_exp = st.columns(2)
    with col_clear:
        if st.button("🧹 Clear Chat", use_container_width=True, help="Clear messages in this chat"):
            active_session_data["messages"] = []
            save_user_sessions(current_user, current_sessions)
            st.rerun()
    with col_exp:
        if active_session_data.get("messages"):
            chat_md = "\n\n".join([f"### {m['role'].capitalize()}:\n{m['content']}" for m in active_session_data["messages"]])
            st.download_button("📥 Export", data=chat_md, file_name=f"{active_session_data.get('title', 'chat')}.md", mime="text/markdown", use_container_width=True)

# ---------------------------------------------------------------------------
# 6. Main Page (Centered Hero Section & Feature Cards)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-wrapper">
        <div class="hero-title">🎓 Offline RAG & Interactive Study Hub</div>
        <div class="hero-subtitle">
            100% private, local document intelligence powered by <b>Ollama (llama3.2)</b>, 
            Hybrid BM25 + Vector Search, and FlashRank Reranking.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

f_col1, f_col2, f_col3 = st.columns(3)
with f_col1:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">💬</div>
            <div class="feature-heading">Private User Storage</div>
            <div class="feature-desc">All files, chats, and vector indexes are completely isolated to your account.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with f_col2:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">📝</div>
            <div class="feature-heading">Exam & Study Hub</div>
            <div class="feature-desc">Instantly generate Multiple-Choice Quizzes, Formula Flashcards, and Summaries.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with f_col3:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon">📂</div>
            <div class="feature-heading">Multi-Format & OCR</div>
            <div class="feature-desc">Ingests scanned PDFs with page slicing & RapidOCR, plus DOCX, PPTX, CSV, & code.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 7. Main Multi-Tab Interface
# ---------------------------------------------------------------------------
tab_chat, tab_study, tab_inspector = st.tabs(["💬 Document Chat", "📝 Study & Exam Hub", "🔍 Document Inspector"])

current_msgs = active_session_data.get("messages", [])

# --- TAB 1: Conversational Chat ---
with tab_chat:
    for message in current_msgs:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander(f"🔍 Reranked Citations ({len(message['sources'])})"):
                    for idx, doc in enumerate(message["sources"], 1):
                        st.markdown(f"<span class='citation-badge'>Passage {idx} • {doc.get('source', 'Doc')} (Page {doc.get('page', 'N/A')})</span>", unsafe_allow_html=True)
                        st.markdown(f"<div class='source-card'>{doc.get('content', '')}</div>", unsafe_allow_html=True)

    user_query = st.chat_input(f"Ask a question as '{current_display}' in '{active_session_data.get('title', 'Chat')}'...")

    if user_query:
        if session_vector_store is None:
            st.warning("Please upload and index a document for this workspace in the sidebar first.")
        else:
            s_title = active_session_data.get("title", "")
            if (s_title.startswith("Chat ") or s_title == "General Chat" or s_title.startswith("Session ")) and not current_msgs:
                active_session_data["title"] = (user_query[:26] + "..") if len(user_query) > 26 else user_query

            current_msgs.append({"role": "user", "content": user_query, "sources": []})
            active_session_data["messages"] = current_msgs
            save_user_sessions(current_user, current_sessions)

            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving via Hybrid Search & Reranking..."):
                    try:
                        standalone_q = contextualize_query(user_query, current_msgs)
                        candidate_docs = session_vector_store.similarity_search(standalone_q, k=config.RETRIEVAL_CANDIDATES)
                        top_docs = rerank_documents(standalone_q, candidate_docs, top_k=config.TOP_K)
                        formatted_context = format_docs(top_docs)

                        from langchain_ollama import ChatOllama
                        from langchain_core.prompts import ChatPromptTemplate

                        llm = ChatOllama(model=config.OLLAMA_MODEL, temperature=config.TEMPERATURE)
                        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
                        chain_input = prompt.format_messages(context=formatted_context, question=user_query)

                        def stream_resp():
                            for chunk in llm.stream(chain_input):
                                yield chunk.content

                        response_placeholder = st.empty()
                        full_resp = response_placeholder.write_stream(stream_resp())

                        serialized_sources = [
                            {"content": d.page_content, "source": d.metadata.get("source", "Doc"), "page": d.metadata.get("page", "N/A")}
                            for d in top_docs
                        ]

                        if serialized_sources:
                            with st.expander(f"🔍 Reranked Citations ({len(serialized_sources)})"):
                                for idx, doc in enumerate(serialized_sources, 1):
                                    st.markdown(f"<span class='citation-badge'>Passage {idx} • {doc['source']} (Page {doc['page']})</span>", unsafe_allow_html=True)
                                    st.markdown(f"<div class='source-card'>{doc['content']}</div>", unsafe_allow_html=True)

                        current_msgs.append({
                            "role": "assistant",
                            "content": full_resp,
                            "sources": serialized_sources,
                        })
                        active_session_data["messages"] = current_msgs
                        save_user_sessions(current_user, current_sessions)

                    except Exception as e:
                        st.error(f"Generation Error: {str(e)}")

# --- TAB 2: Study & Exam Hub ---
with tab_study:
    st.markdown("### 🎓 Interactive Study Toolkits")
    st.caption(f"Generating study materials for **{current_display} / {active_session_data.get('title', 'Workspace')}**.")

    if session_vector_store is None:
        st.info("👈 Upload and index a document in the sidebar first to unlock study tools.")
    else:
        st_col1, st_col2, st_col3 = st.columns(3)
        with st_col1:
            quiz_btn = st.button("📝 Generate Practice Quiz (MCQs)", use_container_width=True, type="primary", key=f"q_{current_user}_{active_id}")
        with st_col2:
            flashcard_btn = st.button("🎴 Generate Study Flashcards", use_container_width=True, key=f"f_{current_user}_{active_id}")
        with st_col3:
            summary_btn = st.button("📑 One-Click Executive Summary", use_container_width=True, key=f"s_{current_user}_{active_id}")

        if quiz_btn:
            with st.spinner("Synthesizing Multiple Choice Practice Quiz..."):
                sample_docs = session_vector_store.similarity_search("core principles concepts overview", k=15)
                quiz_output = generate_study_tool("quiz", sample_docs, extra_param=5)
                st.markdown(quiz_output)
                st.download_button("📥 Download Quiz (.md)", data=quiz_output, file_name="practice_quiz.md", mime="text/markdown", key=f"dq_{current_user}_{active_id}")

        if flashcard_btn:
            with st.spinner("Extracting Technical Terms & Core Flashcards..."):
                sample_docs = session_vector_store.similarity_search("definitions formulas technical terms", k=15)
                flashcards_output = generate_study_tool("flashcards", sample_docs, extra_param=6)
                st.markdown(flashcards_output)
                st.download_button("📥 Download Flashcards (.md)", data=flashcards_output, file_name="study_flashcards.md", mime="text/markdown", key=f"df_{current_user}_{active_id}")

        if summary_btn:
            with st.spinner("Generating Structured Document Summary..."):
                sample_docs = session_vector_store.similarity_search("main topics summary conclusion", k=15)
                summary_output = generate_study_tool("summary", sample_docs)
                st.markdown(summary_output)
                st.download_button("📥 Download Summary (.md)", data=summary_output, file_name="executive_summary.md", mime="text/markdown", key=f"ds_{current_user}_{active_id}")

# --- TAB 3: Document Inspector ---
with tab_inspector:
    st.markdown("### 🔍 Document Inspector")
    if session_vector_store is None:
        st.info("Upload and index a document in this workspace to inspect its vector collection.")
    else:
        doc_count = session_vector_store._collection.count()
        st.write(f"Total Vectors in `{current_display} / {active_session_data.get('title', 'Workspace')}`: **{doc_count}**")
        sample_results = session_vector_store.similarity_search("overview", k=min(10, max(1, doc_count)))
        for i, d in enumerate(sample_results, 1):
            with st.expander(f"Chunk {i} • {d.metadata.get('source')} (Page {d.metadata.get('page')})"):
                st.text(d.page_content)
