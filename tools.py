def search_documents(vectorstore, query, k=5):
    docs = vectorstore.similarity_search(query, k=k)
    results = []
    for doc in docs:
        results.append({
            "source": doc.metadata.get("source", "Source inconnue"),
            "page": doc.metadata.get("page", "N/A"),
            "excerpt": doc.page_content[:300]
        })
    return results

def summarize_text(llm, text):
    prompt = f"Résume ce texte en français de façon claire :\n\n{text}"
    response = llm.invoke(prompt)
    return response.content


def generate_quiz(llm, text):
    prompt = f"Génère 3 questions de révision à partir du texte suivant :\n\n{text}"
    response = llm.invoke(prompt)
    return response.content

from duckduckgo_search import DDGS

def search_web(query: str, max_results: int = 5):
    results = []

    try:
        with DDGS() as ddgs:
            found = ddgs.text(
                query,
                max_results=max_results
            )

            for r in found:
                results.append(
                    {
                        "title": r.get("title", "Sans titre"),
                        "link": r.get("href", "") or r.get("url", ""),
                        "snippet": r.get("body", "") or r.get("snippet", ""),
                    }
                )

    except Exception as e:
        results.append(
            {
                "title": "Erreur de recherche web",
                "link": "",
                "snippet": f"Impossible d'exécuter la recherche : {e}",
            }
        )

    return results

def web_summary(llm, query):
    results = search_web(query)

    text = "\n\n".join([r["snippet"] for r in results])

    prompt = f"""
Résume ces informations issues du web :

{text}

Réponse :
"""

    response = llm.invoke(prompt)
    return response.content