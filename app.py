import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

import RAG
from router import route_query
from tools import get_llm
from memory import init_memory, add_message, clear_memory, get_history

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

APP_TITLE = "Thesis Research Assistant"
DEFAULT_DATA_DIR = "data"
DEFAULT_INDEX_DIR = "faiss_index"
DEFAULT_LLM_MODEL = "mistral-small-latest"

ROUTE_LABELS = {
    "chat": "Discussion",
    "rag": "Réponse sur documents",
    "doc_search": "Recherche documentaire",
    "tool": "Outil",
}

TOOL_LABELS = {
    "calcul": "Calcul",
    "web": "Recherche web",
    "meteo": "Météo",
}

# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# STYLE
# =========================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1450px;
            padding-top: 1.1rem;
            padding-bottom: 1.2rem;
        }
        .hero {
            border: 1px solid rgba(120,120,120,0.20);
            border-radius: 24px;
            padding: 1.3rem 1.35rem;
            background: linear-gradient(135deg, rgba(120,120,120,0.10), rgba(120,120,120,0.03));
            margin-bottom: 1rem;
        }
        .soft-card {
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 18px;
            padding: 1rem;
            background: rgba(120,120,120,0.035);
        }
        .result-card {
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 18px;
            padding: 1rem;
            background: rgba(120,120,120,0.03);
            margin-bottom: 0.85rem;
        }
        .source-card {
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 14px;
            padding: 0.9rem;
            background: rgba(120,120,120,0.035);
            margin-bottom: 0.7rem;
        }
        .pill {
            display: inline-block;
            padding: 0.22rem 0.6rem;
            border-radius: 999px;
            background: rgba(120,120,120,0.12);
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
            font-size: 0.82rem;
        }
        .tiny {
            color: #7d7d7d;
            font-size: 0.9rem;
        }
        .kpi {
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 16px;
            padding: 0.8rem 0.9rem;
            background: rgba(120,120,120,0.03);
            text-align: center;
        }
        .route-tag {
            display: inline-block;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 0.16rem 0.55rem;
            border-radius: 999px;
            background: rgba(120,120,120,0.12);
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# MEMORY
# =========================================================

init_memory(st.session_state)

# =========================================================
# SESSION STATE
# =========================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

if "corpus_stats" not in st.session_state:
    st.session_state.corpus_stats = None

# =========================================================
# CACHE
# =========================================================

@st.cache_resource(show_spinner=False)
def cached_llm(model_name: str):
    return get_llm(model_name)

# =========================================================
# HELPERS UI
# =========================================================

def render_sources(docs):
    if not docs:
        return

    with st.expander("Sources utilisées", expanded=False):
        for i, doc in enumerate(docs, start=1):
            source = Path(doc.metadata.get("source", "Source inconnue")).name
            page = doc.metadata.get("page", "N/A")
            excerpt = doc.page_content[:700].replace("\n", " ").strip()

            st.markdown(
                f"""
                <div class="source-card">
                    <strong>Source {i}</strong><br>
                    <span class="tiny">{source} — page {page}</span><br><br>
                    {excerpt}
                </div>
                """,
                unsafe_allow_html=True,
            )

def render_doc_search_results(results):
    if not results:
        st.info("Aucun document pertinent trouvé.")
        return

    for r in results:
        st.markdown(
            f"""
            <div class='result-card'>
                <strong>{Path(r['source']).name}</strong><br>
                <span class='tiny'>Page {r['page']}</span><br><br>
                {r['excerpt']}
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_web_results(results):
    if not results:
        return

    with st.expander("Sources web", expanded=False):
        for r in results:
            st.markdown(
                f"""
                <div class='result-card'>
                    <strong>{r.get('title', 'Sans titre')}</strong><br>
                    <span class='tiny'>{r.get('link', '')}</span><br><br>
                    {r.get('snippet', '')}
                </div>
                """,
                unsafe_allow_html=True,
            )

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.title("⚙️ Réglages")

    folder_path = st.text_input("Dossier du corpus", value=DEFAULT_DATA_DIR)
    index_dir = st.text_input("Dossier de l'index", value=DEFAULT_INDEX_DIR)

    llm_model = st.selectbox(
        "Modèle Mistral",
        ["mistral-small-latest", "mistral-large-latest", "mistral-medium-latest"],
        index=0,
    )

    chunk_size = st.slider("Chunk size", 400, 1800, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap", 50, 400, 200, 25)
    k_docs = st.slider("Top-k passages", 2, 8, 4)

    st.markdown("---")
    build_btn = st.button("🔄 Reconstruire l'index", use_container_width=True)
    clear_btn = st.button("🧹 Vider l'historique", use_container_width=True)

    st.markdown("---")
    files = RAG.list_supported_files(folder_path)
    st.subheader("Corpus détecté")
    st.caption(f"{len(files)} fichier(s)")

    for p in files[:8]:
        st.markdown(f"<span class='pill'>{p.name}</span>", unsafe_allow_html=True)

    if len(files) > 8:
        st.caption(f"+ {len(files) - 8} autres")

# =========================================================
# ACTIONS SIDEBAR
# =========================================================

if clear_btn:
    clear_memory(st.session_state)
    st.rerun()

if build_btn:
    with st.spinner("Construction de l'index..."):
        try:
            vectorstore, documents, chunks = RAG.rebuild_index(
                folder_path=folder_path,
                index_dir=index_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            st.session_state.vectorstore = vectorstore
            st.session_state.index_ready = True
            st.session_state.corpus_stats = {
                "documents": len(documents),
                "chunks": len(chunks),
            }
            st.success("Index prêt.")
        except Exception as e:
            st.error(f"Impossible de construire l'index : {e}")

if st.session_state.vectorstore is None and Path(index_dir).exists():
    try:
        st.session_state.vectorstore = RAG.load_vectorstore(index_dir)
        st.session_state.index_ready = True
    except Exception:
        pass

# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h2 style="margin-bottom:0.35rem;">🧠 Thesis Research Assistant</h2>
        <div style="margin-bottom:0.7rem;">
            Un assistant pour explorer tes articles, notes, bibliographie et résultats textuels.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"<div class='kpi'><div class='tiny'>LLM</div><strong>{llm_model}</strong></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='kpi'><div class='tiny'>Index</div><strong>{'Prêt' if st.session_state.index_ready else 'Non prêt'}</strong></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='kpi'><div class='tiny'>Docs détectés</div><strong>{len(files)}</strong></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='kpi'><div class='tiny'>Top-k</div><strong>{k_docs}</strong></div>", unsafe_allow_html=True)

if st.session_state.corpus_stats:
    st.caption(
        f"Documents chargés : {st.session_state.corpus_stats['documents']} • "
        f"Chunks : {st.session_state.corpus_stats['chunks']}"
    )

st.markdown("---")

left, right = st.columns([2.2, 1])

with left:
    query = st.text_area("Requête", height=110)

with right:
    st.markdown(
        """
        <div class='soft-card'>
            <strong>Conseil</strong><br>
            <span class='tiny'>
                Le routeur choisit automatiquement entre outil, documents ou discussion.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    run_btn = st.button("▶️ Lancer la requête", use_container_width=True)

st.markdown("---")

tab_result, tab_history, tab_corpus = st.tabs(["Résultat", "Historique", "Corpus"])

# =========================================================
# RUN QUERY
# =========================================================

llm = cached_llm(llm_model)

if run_btn and query.strip():
    add_message(st.session_state, "user", query, route="user")
    history = get_history(st.session_state, limit=6)

    with st.spinner("Analyse de la requête..."):
        try:
            routed = route_query(
                question=query,
                vectorstore=st.session_state.vectorstore,
                llm=llm,
                history=history,
                k_docs=k_docs,
                folder_path=folder_path,
            )

            route_name = routed.get("route", "chat")
            tool_name = routed.get("tool")
            result = routed.get("result", "")
            docs = routed.get("docs", [])
            extra = routed.get("extra", [])

            add_message(
                st.session_state,
                "assistant",
                result,
                route=route_name,
                tool=tool_name,
                docs=docs,
                extra=extra,
            )
        except Exception as e:
            st.error(f"Erreur pendant la requête : {e}")

# =========================================================
# TAB RESULT
# =========================================================

with tab_result:
    if not st.session_state.messages:
        st.info("Aucun résultat pour le moment.")
    else:
        last_assistant = None
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant":
                last_assistant = msg
                break

        if last_assistant is None:
            st.info("Aucune réponse assistant disponible.")
        else:
            if last_assistant["route"] == "tool" and last_assistant.get("tool"):
                label = TOOL_LABELS.get(last_assistant["tool"], last_assistant["tool"].capitalize())
            else:
                label = ROUTE_LABELS.get(last_assistant["route"], last_assistant["route"])
            st.markdown(f"<div class='route-tag'>{label}</div>", unsafe_allow_html=True)

            if last_assistant["route"] == "doc_search" and isinstance(last_assistant["content"], list):
                render_doc_search_results(last_assistant["content"])
            else:
                st.markdown(last_assistant["content"])

            if last_assistant.get("docs"):
                render_sources(last_assistant["docs"])

            if last_assistant.get("tool") == "web" and last_assistant.get("extra"):
                render_web_results(last_assistant["extra"])

# =========================================================
# TAB HISTORY
# =========================================================

with tab_history:
    if not st.session_state.messages:
        st.info("Aucun échange pour l'instant.")
    else:
        for msg in st.session_state.messages:
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                if msg["role"] == "assistant":
                    if msg["route"] == "tool" and msg.get("tool"):
                        label = TOOL_LABELS.get(msg["tool"], msg["tool"].capitalize())
                    else:
                        label = ROUTE_LABELS.get(msg["route"], msg["route"])
                    st.markdown(f"<div class='route-tag'>{label}</div>", unsafe_allow_html=True)

                if msg["route"] == "doc_search" and isinstance(msg["content"], list):
                    render_doc_search_results(msg["content"])
                else:
                    st.markdown(msg["content"])

                if msg.get("docs"):
                    render_sources(msg["docs"])

                if msg.get("tool") == "web" and msg.get("extra"):
                    render_web_results(msg["extra"])

# =========================================================
# TAB CORPUS
# =========================================================

with tab_corpus:
    if not files:
        st.info("Aucun fichier PDF, DOCX ou TXT détecté dans le dossier du corpus.")
    else:
        search_name = st.text_input("Filtrer les documents par nom", value="")
        shown = files

        if search_name.strip():
            shown = [p for p in files if search_name.lower() in p.name.lower()]

        st.caption(f"{len(shown)} document(s) affiché(s)")

        for p in shown:
            st.markdown(
                f"""
                <div class='result-card'>
                    <strong>{p.name}</strong><br>
                    <span class='tiny'>{p.parent}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("---")
st.caption("Astuce : commence par un corpus restreint et homogène, puis élargis progressivement.")