import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# 1) CONFIGURATION

load_dotenv()

DATA_DIR = "data"
FAISS_DIR = "faiss_index"

USE_OPENAI_FOR_ANSWER = True
LOCAL_LLM_MODEL = "mistral"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "gpt-4.1-mini"



# 2) CHARGER LES DOCUMENTS

def load_documents(folder_path: str):
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Le dossier '{folder_path}' n'existe pas.")

    documents = []

    for file_path in folder.iterdir():
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())

        elif suffix == ".docx":
            loader = Docx2txtLoader(str(file_path))
            documents.extend(loader.load())

    if not documents:
        raise ValueError(
            f"Aucun fichier PDF ou DOCX trouvé dans le dossier '{folder_path}'."
        )

    return documents



# 3) DECOUPER EN CHUNKS

def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)



# 4) EMBEDDINGS LOCAUX

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )



# 5) CREER OU CHARGER L'INDEX FAISS

def build_or_load_vectorstore(chunks):
    embeddings = get_embeddings()

    if os.path.exists(FAISS_DIR):
        print("Index FAISS existant trouvé. Chargement...")
        vectorstore = FAISS.load_local(
            FAISS_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("Aucun index trouvé. Création de l'index FAISS...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(FAISS_DIR)
        print(f"Index sauvegardé dans '{FAISS_DIR}'.")

    return vectorstore



# 6) RECHERCHE SIMPLE

def retrieve_documents(vectorstore, question: str, k: int = 4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    return docs



# 7) REPONSE LOCALE SANS OPENAI

def local_answer(question: str, docs):
    if not docs:
        return "Je ne trouve pas d'information pertinente dans les documents."

    answer = []
    answer.append(
        "Je n'utilise pas de LLM ici : voici les passages les plus pertinents retrouvés pour ta question.\n"
    )

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Source inconnue")
        page = doc.metadata.get("page", "N/A")
        excerpt = doc.page_content[:700].replace("\n", " ").strip()

        answer.append(
            f"[Passage {i}] Source: {source} | Page: {page}\n{excerpt}\n"
        )

    return "\n".join(answer)



# 8) REPONSE AVEC Mistral

def generate_answer(question: str, docs):
    if not docs:
        return "Je ne trouve pas d'information pertinente dans les documents."

    llm = ChatOllama(
        model="mistral",   # ou "llama3"
        temperature=0
    )

    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'Source inconnue')} | "
            f"Page: {doc.metadata.get('page', 'N/A')}\n{doc.page_content}"
            for doc in docs
        ]
    )

    prompt = f"""
Tu es un assistant scientifique spécialisé dans les récepteurs FFA/GPCR.

Réponds uniquement à partir du contexte fourni.
Si l'information n'est pas dans le contexte, dis clairement :
"Je ne trouve pas cette information dans les documents fournis."

La réponse doit être :
- claire, précise et structurée
- fidèle aux documents
- chaque affirmation doit être suivie d'une citation inline au format [nom_du_fichier, p.X]
  Exemple : "FFA4 est impliqué dans la régulation de l'inflammation [etude_gpcr.pdf, p.3]."

Contexte :
{context}

Question :
{question}

Réponse :
"""

    response = llm.invoke(prompt)
    return response.content



# 9) AFFICHER LES SOURCES

def print_sources(context_docs):
    print("\n" + "=" * 80)
    print("SOURCES UTILISEES")
    print("=" * 80)

    for i, doc in enumerate(context_docs, start=1):
        source = doc.metadata.get("source", "Source inconnue")
        page = doc.metadata.get("page", "N/A")

        print(f"\n--- Source {i} ---")
        print(f"Fichier : {source}")
        print(f"Page   : {page}")
        print("Extrait :")
        print(doc.page_content[:600].replace("\n", " "))
        print()



# 9b) ANSWER WITH RAG (utilisé par router.py)

def answer_with_rag(vectorstore, question: str):
    docs = retrieve_documents(vectorstore, question)
    answer = generate_answer(question, docs)
    return {"mode": "rag", "answer": answer, "docs": docs}



# 10) MAIN

def main():
    print("Chargement des documents...")
    documents = load_documents(DATA_DIR)
    print(f"{len(documents)} document(s)/page(s) chargé(s).")

    print("Découpage en chunks...")
    chunks = split_documents(documents)
    print(f"{len(chunks)} chunk(s) créés.")

    print("Création / chargement du vector store...")
    vectorstore = build_or_load_vectorstore(chunks)

    print("\nAssistant RAG prêt.")
    print("Tape ta question. Tape 'quit' pour arrêter.\n")

    while True:
        question = input("Question : ").strip()

        if question.lower() in {"quit", "exit", "q"}:
            print("Fin du programme.")
            break

        if not question:
            print("Merci d'entrer une question.\n")
            continue

        try:
            docs = retrieve_documents(vectorstore, question, k=4)

            if USE_OPENAI_FOR_ANSWER:
                answer = generate_answer(question, docs)
            else:
                answer = local_answer(question, docs)

            print("\n" + "=" * 80)
            print("REPONSE")
            print("=" * 80)
            print(answer)

            if docs:
                print_sources(docs)

            print("\n")

        except Exception as e:
            print(f"\nErreur pendant l'exécution : {e}\n")


if __name__ == "__main__":
    main()