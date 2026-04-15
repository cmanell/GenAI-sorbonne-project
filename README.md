# Assistant Intelligent MultiCompétences — GPCR Research Assistant

Projet Sorbonne — RAG + Agents  
Assistant scientifique spécialisé dans les récepteurs FFA/GPCR, combinant un pipeline RAG, un routeur LLM automatique et plusieurs outils intégrés.

---

## Architecture

```
GenAI-sorbonne-project/
├── app.py        # Interface Streamlit principale
├── RAG.py           # Pipeline RAG (ingestion, vectorisation, réponse)
├── router.py        # Routeur LLM automatique
├── tools.py         # Outils : calculatrice, recherche web, résumé, quiz, doc_search
├── data/            # Dossier du corpus (PDF, DOCX)
├── faiss_index/     # Index vectoriel FAISS (généré automatiquement)
├── .env             # Clé API Mistral (non versionné)
└── requirements.txt  # Dépendances Python
```

### Flux de données

```
Document (PDF/DOCX)
    → Chargement (PyPDFLoader / Docx2txtLoader)
    → Découpage en chunks (RecursiveCharacterTextSplitter)
    → Embeddings (HuggingFace all-MiniLM-L6-v2)
    → Index FAISS

Question utilisateur
    → Routeur LLM (classify_query via Mistral)
    → Outil sélectionné (RAG / Web / Calculatrice / Résumé / Quiz / Doc Search / Chat)
    → Réponse avec citations inline [fichier, p.X]
```

---



### Mémoire conversationnelle
Les 6 derniers messages sont injectés dans chaque prompt pour contextualiser les réponses.


---

## Installation

### Prérequis
- Python 3.10+
- Une clé API [Mistral](https://console.mistral.ai/)

### 1. Cloner le dépôt

```bash
git clone <url-du-repo>
cd GenAI-sorbonne-project
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer la clé API

Crée un fichier `.env` à la racine :

```
MISTRAL_API_KEY=ta_clé_mistral_ici
```

### 5. Déposer les documents

Place tes fichiers PDF ou DOCX dans le dossier `data/`.

---

## Lancement

### Interface Streamlit

```bash
streamlit run app.py
```

Ouvre ensuite [http://localhost:8501](http://localhost:8501), puis clique sur **"🔄 Reconstruire l'index"** dans la sidebar pour indexer le corpus.



## Dépendances principales

| Package | Rôle |
|---|---|
| `langchain-mistralai` | LLM Mistral via API |
| `langchain-community` | Loaders, FAISS, embeddings |
| `sentence-transformers` | Embeddings locaux (all-MiniLM-L6-v2) |
| `faiss-cpu` | Index vectoriel |
| `streamlit` | Interface web |
| `ddgs` | Recherche web DuckDuckGo |
| `pypdf` / `docx2txt` | Lecture des documents |

---

## Modèles disponibles

Le modèle Mistral est sélectionnable dans la sidebar de l'interface :

- `mistral-small-latest` (défaut, rapide)
- `mistral-large-latest` (plus performant)
- `open-mistral-7b` (open source)
