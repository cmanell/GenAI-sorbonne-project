import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI

try:
    from ddgs import DDGS
    HAS_DDG = True
except Exception:
    HAS_DDG = False



# CONFIG

load_dotenv()

APP_TITLE = "GPCR Research Assistant"
DEFAULT_DATA_DIR = "data"
DEFAULT_INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "mistral-small-latest"

MODE_CONFIG = {
    "Auto": {
        "key": "auto",
        "emoji": "🤖",
        "desc": "Détection automatique du mode via le LLM.",
    },
    "Question-réponse": {
        "key": "rag",
        "emoji": "🧠",
        "desc": "Répondre à une question à partir du corpus.",
    },
    "Recherche de documents": {
        "key": "doc_search",
        "emoji": "📁",
        "desc": "Retrouver les fichiers et passages pertinents.",
    },
    "Résumé": {
        "key": "summary",
        "emoji": "📝",
        "desc": "Résumer un thème ou un ensemble de documents.",
    },
    "Quiz": {
        "key": "quiz",
        "emoji": "❓",
        "desc": "Générer des questions de révision.",
    },
    "Recherche web": {
        "key": "web",
        "emoji": "🌐",
        "desc": "Compléter le corpus avec une recherche internet.",
    },
    "Calculatrice": {
        "key": "calcul",
        "emoji": "🧮",
        "desc": "Effectuer un calcul arithmétique.",
    },
    "Conversation": {
        "key": "chat",
        "emoji": "💬",
        "desc": "Discussion libre sans RAG ni outils.",
    },
}



# PAGE + STYLE

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
        .mode-card {
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 18px;
            padding: 0.95rem;
            background: rgba(120,120,120,0.03);
            min-height: 120px;
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



# BACKEND

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str):
    return ChatMistralAI(model=model_name, temperature=0, api_key=os.getenv("MISTRAL_API_KEY"))


def list_supported_files(folder_path: str) -> List[Path]:
    folder = Path(folder_path)
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".pdf", ".docx"}]


def load_documents(folder_path: str):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Le dossier '{folder_path}' n'existe pas.")

    documents = []
    for file_path in folder.iterdir():
        if file_path.suffix.lower() == ".pdf":
            documents.extend(PyPDFLoader(str(file_path)).load())
        elif file_path.suffix.lower() == ".docx":
            documents.extend(Docx2txtLoader(str(file_path)).load())

    if not documents:
        raise ValueError("Aucun PDF ou DOCX trouvé dans le dossier indiqué.")
    return documents


def split_documents(documents, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def build_or_load_vectorstore(chunks, index_dir: str):
    embeddings = get_embeddings()
    index_path = Path(index_dir)
    if index_path.exists():
        return FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(index_path))
    return vectorstore


def rebuild_index(folder_path: str, index_dir: str, chunk_size: int, chunk_overlap: int):
    index_path = Path(index_dir)
    if index_path.exists():
        import shutil
        shutil.rmtree(index_path)

    documents = load_documents(folder_path)
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    vectorstore = build_or_load_vectorstore(chunks, index_dir)
    return vectorstore, documents, chunks


def retrieve_documents(vectorstore, question: str, k: int = 4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(question)


def generate_answer(question: str, docs, llm, history: List[Dict[str, str]] = None) -> str:
    if not docs:
        return "Je ne trouve pas d'information pertinente dans les documents."

    context = "\n\n".join(
        [
            f"Source: {Path(doc.metadata.get('source', 'Source inconnue')).name} | Page: {doc.metadata.get('page', 'N/A')}\n{doc.page_content}"
            for doc in docs
        ]
    )

    history_text = ""
    if history:
        lines = []
        for msg in history[-6:]:  # 3 derniers échanges max
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role} : {msg['content']}")
        history_text = "\n".join(lines)

    prompt = f"""
Tu es un assistant de recherche scientifique spécialisé dans les récepteurs FFA/GPCR.
Réponds uniquement à partir du contexte fourni.
Si l'information n'est pas dans le contexte, dis clairement que tu ne la trouves pas dans les documents.

Contraintes :
- réponse en français
- style clair, académique et structuré
- ne rien inventer
- chaque affirmation doit être suivie d'une citation inline au format [nom_du_fichier, p.X]
  Exemple : "FFA4 est impliqué dans la régulation de l'inflammation [etude_gpcr.pdf, p.3]."
{f"- tiens compte de l'historique de conversation ci-dessous pour contextualiser ta réponse" if history_text else ""}

{f"Historique :{chr(10)}{history_text}{chr(10)}" if history_text else ""}
Contexte :
{context}

Question :
{question}

Réponse :
"""
    return llm.invoke(prompt).content


def search_documents(vectorstore, query: str, k: int = 6) -> List[Dict[str, Any]]:
    docs = retrieve_documents(vectorstore, query, k=k)
    out = []
    for doc in docs:
        out.append(
            {
                "source": doc.metadata.get("source", "Source inconnue"),
                "page": doc.metadata.get("page", "N/A"),
                "excerpt": doc.page_content[:450].replace("\n", " ").strip(),
            }
        )
    return out


def summarize_from_docs(query: str, vectorstore, llm, k: int = 4) -> Tuple[str, List[Any]]:
    docs = retrieve_documents(vectorstore, query, k=k)
    if not docs:
        return "Je ne trouve pas de contenu pertinent à résumer.", []
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
Résume en français le contenu suivant.
Structure le résultat en 3 parties :
1. idée générale
2. points clés
3. à retenir

Contenu :
{context}

Résumé :
"""
    return llm.invoke(prompt).content, docs


def generate_quiz(query: str, vectorstore, llm, k: int = 4) -> Tuple[str, List[Any]]:
    docs = retrieve_documents(vectorstore, query, k=k)
    if not docs:
        return "Je ne trouve pas assez de contenu pertinent pour générer un quiz.", []
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
À partir du contenu ci-dessous, génère un quiz de 5 questions en français.
Ajoute un corrigé bref à la fin.

Contenu :
{context}

Quiz :
"""
    return llm.invoke(prompt).content, docs


def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    if not HAS_DDG:
        return [{"title": "Module manquant", "link": "", "snippet": "Installe duckduckgo-search pour activer la recherche web."}]

    out: List[Dict[str, str]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            out.append(
                {
                    "title": r.get("title", "Sans titre"),
                    "link": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
            )
    return out


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


def add_message(role: str, content: str, mode: str = "rag", docs=None, extra=None):
    st.session_state.messages.append(
        {"role": role, "content": content, "mode": mode, "docs": docs or [], "extra": extra}
    )


def dispatch_mode(mode_key: str, query: str, llm, vectorstore, k_docs: int, mode_label: str):
    needs_index = mode_key in {"rag", "doc_search", "summary", "quiz"}
    if needs_index and vectorstore is None:
        st.warning("L'index n'est pas prêt. Reconstruis d'abord l'index dans la barre latérale.")
        return

    if mode_key == "rag":
        docs = retrieve_documents(vectorstore, query, k=k_docs)
        answer = generate_answer(query, docs, llm, history=st.session_state.messages)
        st.markdown(f"<div class='mode-tag'>{mode_label}</div>", unsafe_allow_html=True)
        st.markdown(answer)
        render_sources(docs)
        add_message("assistant", answer, mode=mode_key, docs=docs)

    elif mode_key == "doc_search":
        results = search_documents(vectorstore, query, k=k_docs)
        st.markdown(f"<div class='mode-tag'>{mode_label}</div>", unsafe_allow_html=True)
        if not results:
            st.info("Aucun document pertinent trouvé.")
            add_message("assistant", "Aucun document pertinent trouvé.", mode=mode_key)
        else:
            blocks = []
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
                blocks.append(f"{Path(r['source']).name} — page {r['page']}\n{r['excerpt']}")
            add_message("assistant", "\n\n".join(blocks), mode=mode_key)

    elif mode_key == "summary":
        summary, docs = summarize_from_docs(query, vectorstore, llm, k=k_docs)
        st.markdown(f"<div class='mode-tag'>{mode_label}</div>", unsafe_allow_html=True)
        st.markdown(summary)
        render_sources(docs)
        add_message("assistant", summary, mode=mode_key, docs=docs)

    elif mode_key == "quiz":
        quiz, docs = generate_quiz(query, vectorstore, llm, k=k_docs)
        st.markdown(f"<div class='mode-tag'>{mode_label}</div>", unsafe_allow_html=True)
        st.markdown(quiz)
        render_sources(docs)
        add_message("assistant", quiz, mode=mode_key, docs=docs)

    elif mode_key == "calcul":
        from tools import calculate
        import re
        match = re.search(r"[\d+\-*/().^, ]+", query)
        expression = match.group().strip() if match else query
        result = calculate(expression)
        st.markdown(f"<div class='mode-tag'>{mode_label}</div>", unsafe_allow_html=True)
        st.markdown(f"**{result}**")
        add_message("assistant", result, mode=mode_key)

    elif mode_key == "chat":
        history_text = ""
        for msg in st.session_state.messages[-6:]:
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            history_text += f"{role} : {msg['content']}\n"
        chat_prompt = f"{history_text}Utilisateur : {query}\nAssistant :"
        answer = llm.invoke(chat_prompt).content
        st.markdown(f"<div class='mode-tag'>{mode_label}</div>", unsafe_allow_html=True)
        st.markdown(answer)
        add_message("assistant", answer, mode=mode_key)

    elif mode_key == "web":
        web_results = search_web(query)
        st.markdown(f"<div class='mode-tag'>{mode_label}</div>", unsafe_allow_html=True)
        if not web_results:
            st.info("Aucun résultat web trouvé.")
            add_message("assistant", "Aucun résultat web trouvé.", mode=mode_key)
        else:
            snippets = "\n\n".join([r.get("snippet", "") for r in web_results if r.get("snippet")])
            synthesis_prompt = f"""Tu es un assistant de recherche. Synthétise ces informations issues du web en une réponse claire et structurée en français.

Informations :
{snippets}

Question : {query}

Réponse :"""
            answer = llm.invoke(synthesis_prompt).content
            st.markdown(f"<div class='mode-tag'>{mode_label}</div>", unsafe_allow_html=True)
            st.markdown(answer)
            with st.expander("Sources web", expanded=False):
                for r in web_results:
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
            add_message("assistant", answer, mode=mode_key, extra=web_results)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("⚙️ Réglages")
    folder_path = st.text_input("Dossier du corpus", value=DEFAULT_DATA_DIR)
    index_dir = st.text_input("Dossier de l'index", value=DEFAULT_INDEX_DIR)
    llm_model = st.selectbox("Modèle Mistral", ["mistral-small-latest", "mistral-large-latest", "open-mistral-7b"], index=0)
    chunk_size = st.slider("Chunk size", 400, 1800, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap", 50, 400, 200, 25)
    k_docs = st.slider("Top-k passages", 2, 8, 4)

    st.markdown("---")
    build_btn = st.button("🔄 Reconstruire l'index", use_container_width=True)
    clear_btn = st.button("🧹 Vider l'historique", use_container_width=True)

    st.markdown("---")
    st.subheader("Ajouter un document")
    uploaded_file = st.file_uploader("PDF ou DOCX", type=["pdf", "docx"], label_visibility="collapsed")
    if uploaded_file is not None:
        dest = Path(folder_path) / uploaded_file.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"**{uploaded_file.name}** sauvegardé dans `{folder_path}/`")
        if st.button("➕ Indexer ce document", use_container_width=True):
            with st.spinner("Indexation en cours..."):
                try:
                    if dest.suffix.lower() == ".pdf":
                        new_docs = PyPDFLoader(str(dest)).load()
                    else:
                        new_docs = Docx2txtLoader(str(dest)).load()
                    new_chunks = split_documents(new_docs, chunk_size, chunk_overlap)
                    if st.session_state.vectorstore is not None:
                        st.session_state.vectorstore.add_documents(new_chunks)
                        st.session_state.vectorstore.save_local(index_dir)
                    else:
                        st.session_state.vectorstore = FAISS.from_documents(new_chunks, get_embeddings())
                        st.session_state.vectorstore.save_local(index_dir)
                        st.session_state.index_ready = True
                    st.success(f"{len(new_chunks)} chunks ajoutés à l'index.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur lors de l'indexation : {e}")

    st.markdown("---")
    files = list_supported_files(folder_path)
    st.subheader("Corpus détecté")
    st.caption(f"{len(files)} fichier(s)")
    for p in files[:8]:
        st.markdown(f"<span class='pill'>{p.name}</span>", unsafe_allow_html=True)
    if len(files) > 8:
        st.caption(f"+ {len(files) - 8} autres")



# SESSION STATE

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "active_mode" not in st.session_state:
    st.session_state.active_mode = "Auto"

if clear_btn:
    st.session_state.messages = []
    st.rerun()

if build_btn:
    with st.spinner("Construction de l'index..."):
        try:
            vectorstore, documents, chunks = rebuild_index(folder_path, index_dir, chunk_size, chunk_overlap)
            st.session_state.vectorstore = vectorstore
            st.session_state.index_ready = True
            st.session_state.corpus_stats = {"documents": len(documents), "chunks": len(chunks)}
            st.success("Index prêt.")
        except Exception as e:
            st.error(f"Impossible de construire l'index : {e}")

if st.session_state.vectorstore is None and Path(index_dir).exists():
    try:
        st.session_state.vectorstore = FAISS.load_local(index_dir, get_embeddings(), allow_dangerous_deserialization=True)
        st.session_state.index_ready = True
    except Exception:
        pass



# HEADER

st.markdown(
    """
    <div class="hero">
        <h2 style="margin-bottom:0.35rem;">🧠 Thesis Research Assistant</h2>
        <div style="margin-bottom:0.7rem;">
            Un assistant premium pour explorer tes articles, notes de thèse, bibliographie et résultats textuels.
            Choisis un mode clair, lance ta requête, puis exploite les résultats sans friction.
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
if "corpus_stats" in st.session_state:
    st.caption(f"Docs/pages chargés : {st.session_state.corpus_stats['documents']} • Chunks : {st.session_state.corpus_stats['chunks']}")

st.markdown("---")



# MODE SELECTOR

st.subheader("Choisis ton mode")
mode_cols = st.columns(len(MODE_CONFIG))
for i, (label, cfg) in enumerate(MODE_CONFIG.items()):
    with mode_cols[i]:
        st.markdown(
            f"""
            <div class="mode-card">
                <div style="font-size:1.25rem; margin-bottom:0.35rem;">{cfg['emoji']}</div>
                <strong>{label}</strong><br>
                <span class="tiny">{cfg['desc']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(f"Activer", key=f"mode_{cfg['key']}", use_container_width=True):
            st.session_state.active_mode = label
            st.rerun()

st.info(f"Mode actif : **{st.session_state.active_mode}**")

examples = {
    "Auto": "Quel est le rôle de FFA4 dans l'inflammation ?",
    "Question-réponse": "Quel est le rôle de FFA4 ?",
    "Recherche de documents": "Retrouve les documents où je parle de YASARA",
    "Résumé": "Résume les agonistes synthétiques de FFA4",
    "Quiz": "Fais-moi un quiz sur FFA4",
    "Recherche web": "Cherche sur internet FFA4 receptor inflammation",
    "Calculatrice": "Calcule 3.14 * 6.5 * 6.5",
    "Conversation": "Bonjour, que peux-tu faire pour moi ?",
}

left, right = st.columns([2.2, 1])
with left:
    query = st.text_area(
        "Requête",
        value=examples[st.session_state.active_mode],
        height=110,
        help="Adapte ou remplace cette requête.",
    )
with right:
    st.markdown("<div class='soft-card'><strong>Conseil</strong><br><span class='tiny'>Utilise des formulations explicites. En recherche de documents, précise le thème, le mot-clé ou le type de fichier recherché.</span></div>", unsafe_allow_html=True)
    run_btn = st.button("▶️ Lancer la requête", use_container_width=True)

st.markdown("---")



# TABS

tab_result, tab_history, tab_corpus = st.tabs(["Résultat", "Historique", "Corpus"])

with tab_result:
    if run_btn:
        llm = get_llm(llm_model)
        mode_key = MODE_CONFIG[st.session_state.active_mode]["key"]
        add_message("user", query, mode=mode_key)

        with st.spinner("Traitement en cours..."):
            try:
                if mode_key == "auto":
                    from router import classify_query
                    detected_key = classify_query(query, llm)
                    mode_labels = {v["key"]: k for k, v in MODE_CONFIG.items() if v["key"] != "auto"}
                    detected_label = mode_labels.get(detected_key, "Question-réponse")
                    st.info(f"Mode détecté : **{MODE_CONFIG[detected_label]['emoji']} {detected_label}**")
                    dispatch_mode(detected_key, query, llm, st.session_state.vectorstore, k_docs, detected_label)
                else:
                    dispatch_mode(mode_key, query, llm, st.session_state.vectorstore, k_docs, st.session_state.active_mode)

            except Exception as e:
                st.error(f"Erreur : {e}")
                add_message("assistant", f"Erreur : {e}", mode=mode_key)
    else:
        st.markdown("<div class='soft-card'>Lance une requête pour voir le résultat ici.</div>", unsafe_allow_html=True)

with tab_history:
    if not st.session_state.messages:
        st.info("Aucun échange pour l'instant.")
    else:
        for msg in st.session_state.messages:
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                if msg["role"] == "assistant":
                    pretty = [k for k, v in MODE_CONFIG.items() if v["key"] == msg["mode"]]
                    if pretty:
                        st.markdown(f"<div class='mode-tag'>{pretty[0]}</div>", unsafe_allow_html=True)
                st.markdown(msg["content"])
                if msg.get("docs"):
                    render_sources(msg["docs"])
                if msg.get("extra"):
                    with st.expander("Voir le détail", expanded=False):
                        for r in msg["extra"]:
                            st.markdown(f"**{r.get('title', 'Sans titre')}**")
                            if r.get("link"):
                                st.caption(r["link"])
                            st.write(r.get("snippet", ""))
                            st.markdown("---")

with tab_corpus:
    if not files:
        st.info("Aucun fichier PDF ou DOCX détecté dans le dossier du corpus.")
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
