import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from router import route_query
import RAG
from tools import get_llm
from memory import init_memory, add_message, clear_memory, get_history, format_history_for_prompt

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

APP_TITLE = "GPCR Research Assistant"
DEFAULT_DATA_DIR = "data"
DEFAULT_INDEX_DIR = "faiss_index"
DEFAULT_LLM_MODEL = "mistral-small-latest"


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
        .mode-tag {
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

if "active_mode" not in st.session_state:
    st.session_state.active_mode = "Auto"

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
        ["mistral-small-latest", "mistral-large-latest", "open-mistral-7b"],
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
    query = st.text_area(
        "Requête",
        height=110,
    )

with right:
    st.markdown(
        """
        <div class='soft-card'>
            <strong>Conseil</strong><br>
            <span class='tiny'>
                Utilise des formulations explicites. En recherche documentaire, précise le thème ou le mot-clé.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    run_btn = st.button("▶️ Lancer la requête", use_container_width=True)

st.markdown("---")


# =========================================================
# TABS
# =========================================================

tab_result, tab_history, tab_corpus = st.tabs(["Résultat", "Historique", "Corpus"])

with tab_result:
    if run_btn:
        llm = cached_llm(llm_model)
        selected_mode_key = MODE_CONFIG[st.session_state.active_mode]["key"]

        add_message(st.session_state, "user", query, mode=selected_mode_key)

        with st.spinner("Traitement en cours..."):
            try:
                history = get_history(st.session_state)

                if selected_mode_key == "auto":
                    response = route_query(
                        question=query,
                        vectorstore=st.session_state.vectorstore,
                        llm=llm,
                        history=history,
                        k_docs=k_docs,
                    )
                    detected_mode = response["mode"]
                    detected_label = next(
                        (label for label, cfg in MODE_CONFIG.items() if cfg["key"] == detected_mode),
                        detected_mode,
                    )
                    st.info(f"Mode détecté : {detected_label}")
                else:
                    response = route_query(
                        question=query,
                        vectorstore=st.session_state.vectorstore,
                        llm=llm,
                        history=history,
                        k_docs=k_docs,
                    )
                    response["mode"] = selected_mode_key

                mode = response["mode"]
                pretty_label = next(
                    (label for label, cfg in MODE_CONFIG.items() if cfg["key"] == mode),
                    mode,
                )

                st.markdown(f"<div class='mode-tag'>{pretty_label}</div>", unsafe_allow_html=True)

                if mode == "doc_search":
                    render_doc_search_results(response["result"])
                    add_message(
                        st.session_state,
                        "assistant",
                        str(response["result"]),
                        mode=mode,
                    )
                else:
                    st.markdown(response["result"])

                    if response.get("docs"):
                        render_sources(response["docs"])

                    if response.get("extra"):
                        render_web_results(response["extra"])

                    add_message(
                        st.session_state,
                        "assistant",
                        response["result"],
                        mode=mode,
                        docs=response.get("docs", []),
                        extra=response.get("extra", []),
                    )

            except Exception as e:
                st.error(f"Erreur : {e}")
                add_message(
                    st.session_state,
                    "assistant",
                    f"Erreur : {e}",
                    mode="chat",
                )
    else:
        st.markdown(
            "<div class='soft-card'>Lance une requête pour voir le résultat ici.</div>",
            unsafe_allow_html=True,
        )

with tab_history:
    if not st.session_state.messages:
        st.info("Aucun échange pour l'instant.")
    else:
        for msg in st.session_state.messages:
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                if msg["role"] == "assistant":
                    pretty = next(
                        (label for label, cfg in MODE_CONFIG.items() if cfg["key"] == msg["mode"]),
                        msg["mode"],
                    )
                    st.markdown(f"<div class='mode-tag'>{pretty}</div>", unsafe_allow_html=True)

                st.markdown(msg["content"])

                if msg.get("docs"):
                    render_sources(msg["docs"])

                if msg.get("extra"):
                    render_web_results(msg["extra"])

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