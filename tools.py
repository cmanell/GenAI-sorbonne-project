import math
import re

import numexpr
import requests
from ddgs import DDGS

_CONSTANTS = {"pi": math.pi, "e": math.e}


def _extract_expression(text: str) -> str:
    text = text.replace(",", ".")
    # Capture function names (letters followed by '('), numbers, and operators
    tokens = re.findall(r'[a-zA-Z_]\w*(?=\s*\()|[\d]+(?:\.\d+)?(?:[eE][+-]?\d+)?|[+\-*/()^]', text)
    if not tokens:
        return ""
    return " ".join(tokens)


def _to_numexpr(expr: str) -> str:
    """Normalise l'expression pour numexpr."""
    expr = expr.replace("^", "**")
    expr = expr.replace("π", "pi")
    expr = expr.replace("²", "**2")
    expr = expr.replace("³", "**3")
    return expr


def calculate(expression: str) -> str:
    try:
        try:
            result = numexpr.evaluate(_to_numexpr(expression.strip()), local_dict=_CONSTANTS)
        except Exception:
            cleaned = _extract_expression(expression)
            if not cleaned:
                return f"Erreur de calcul : expression non reconnue — '{expression}'"
            result = numexpr.evaluate(_to_numexpr(cleaned), local_dict=_CONSTANTS)

        value = result.item() if hasattr(result, "item") else result
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(round(value, 10))
    except Exception as e:
        return f"Erreur de calcul : {e}"


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


def summarize_document(vectorstore, llm, query, k=4):
    docs = vectorstore.similarity_search(query, k=k)
    if not docs:
        return "Je ne trouve pas de contenu pertinent à résumer."
    text = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Résume ce texte en français de façon claire :\n\n{text}"
    response = llm.invoke(prompt)
    return response.content


def make_quiz(vectorstore, llm, query, k=4):
    docs = vectorstore.similarity_search(query, k=k)
    if not docs:
        return "Je ne trouve pas assez de contenu pour générer un quiz."
    text = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Génère 3 questions de révision à partir du texte suivant :\n\n{text}"
    response = llm.invoke(prompt)
    return response.content


def weather_tool(city: str) -> str:
    try:
        resp = requests.get(f"https://wttr.in/{city}?format=3", timeout=5)
        resp.raise_for_status()
        return resp.content.decode("utf-8")
    except Exception as e:
        return f"Weather error: {e}"


def search_web(query: str, max_results: int = 5):
    results = []
    try:
        with DDGS() as ddgs:
            found = ddgs.text(query, max_results=max_results)
            for r in found:
                results.append({
                    "title": r.get("title", "Sans titre"),
                    "link": r.get("href", "") or r.get("url", ""),
                    "snippet": r.get("body", "") or r.get("snippet", ""),
                })
    except Exception as e:
        results.append({
            "title": "Erreur de recherche web",
            "link": "",
            "snippet": f"Impossible d'exécuter la recherche : {e}",
        })
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
