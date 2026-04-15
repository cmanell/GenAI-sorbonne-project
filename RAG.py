from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from memory import format_history_for_prompt

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def list_supported_files(folder_path: str) -> List[Path]:
    folder = Path(folder_path)
    if not folder.exists():
        return []

    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".txt"}
    ]


def load_documents(folder_path: str) -> List[Document]:
    documents = []
    files = list_supported_files(folder_path)

    for file_path in files:
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()

        elif suffix == ".docx":
            loader = Docx2txtLoader(str(file_path))
            docs = loader.load()

        elif suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()

        else:
            continue

        for doc in docs:
            doc.metadata["source"] = str(file_path)

        documents.extend(docs)

    return documents


def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks: List[Document], index_dir: str):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_dir)
    return vectorstore


def load_vectorstore(index_dir: str):
    embeddings = get_embeddings()
    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def rebuild_index(folder_path: str, index_dir: str, chunk_size: int, chunk_overlap: int):
    index_path = Path(index_dir)
    if index_path.exists():
        import shutil
        shutil.rmtree(index_path)

    documents = load_documents(folder_path)
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vectorstore = build_vectorstore(chunks, index_dir)

    return vectorstore, documents, chunks


def retrieve_documents(vectorstore, query: str, k: int = 4):
    if vectorstore is None:
        return []
    return vectorstore.similarity_search(query, k=k)


def search_documents(vectorstore, query: str, k: int = 5):
    docs = retrieve_documents(vectorstore, query, k=k)
    results = []

    for doc in docs:
        results.append(
            {
                "source": doc.metadata.get("source", "Source inconnue"),
                "page": doc.metadata.get("page", "N/A"),
                "excerpt": doc.page_content[:400].replace("\n", " ").strip(),
            }
        )

    return results


def generate_answer(question: str, docs, llm, history=None) -> str:
    if not docs:
        return "Je ne trouve pas d'information pertinente dans les documents."

    context = "\n\n".join(doc.page_content for doc in docs)
    history_text = format_history_for_prompt(history, limit=6)

    prompt = f"""
Tu es un assistant de recherche documentaire.

Réponds uniquement à partir du contexte fourni.
Si l'information n'est pas présente dans les documents, dis clairement que tu ne sais pas.

Historique :
{history_text}

Contexte :
{context}

Question :
{question}

Réponse en français :
"""

    response = llm.invoke(prompt)
    return response.content