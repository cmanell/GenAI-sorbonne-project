from typing import Dict, Any
import re
import RAG

from memory import format_history_for_prompt
from tools import (
    extract_math_expression,
    calculate,
    search_web,
    weather_tool,
)


import re


def extract_city_from_question(question: str) -> str:
    q = question.strip()

    patterns = [
        r"(?:météo|meteo|temps)\s+(?:à|a|de|d')\s+([A-Za-zÀ-ÿ\- ]+)",
        r"(?:quel temps fait[- ]il|quelle est la météo|quelle est la meteo)\s+(?:à|a|de|d')\s+([A-Za-zÀ-ÿ\- ]+)",
        r"(?:météo|meteo)\s+([A-Za-zÀ-ÿ\- ]+)",
    ]

    city = None

    for pattern in patterns:
        match = re.search(pattern, q, re.IGNORECASE)
        if match:
            city = match.group(1).strip(" ?.!,;:")
            break

    if city is None:
        city = q.strip(" ?.!,;:")

    # Nettoyage des mots parasites
    city = re.sub(r"\b(aujourd'hui|aujourdhui|today|maintenant|ce soir|demain)\b", "", city, flags=re.IGNORECASE)
    city = re.sub(r"^(de|d'|à|a)\s+", "", city, flags=re.IGNORECASE)
    city = re.sub(r"\s+", " ", city).strip()

    return city


def classify_query(question: str, llm, has_vectorstore: bool = True) -> str:
    rag_hint = (
        "rag : si la question semble dépendre du corpus/documentation locale."
        if has_vectorstore
        else "chat : si la question nécessiterait des documents mais qu'aucun index n'est prêt."
    )

    prompt = f"""
    Tu es un routeur intelligent.

    Ta tâche est de choisir UNE SEULE route parmi :
    - tool
    - doc_search
    - rag
    - chat

    Règles :
    - tool : si la question nécessite l'usage d'un outil, comme un calcul, une météo, ou une recherche web.
    - doc_search : si l'utilisateur veut retrouver un document, une source, un passage, une page, ou chercher dans les documents sans demander une réponse rédigée.
    - rag : si l'utilisateur pose une question dont la réponse doit être trouvée dans les documents.
    - chat : pour une conversation générale, explication générale, reformulation, ou discussion qui ne nécessite ni outil ni documents.

    {rag_hint}

    Réponds avec un seul mot parmi :
    tool, doc_search, rag, chat

    Question :
    {question}

    Route :
    """

    response = llm.invoke(prompt)
    route = response.content.strip().lower().split()[0]
    valid = {"tool", "doc_search", "rag", "chat"}

    return route if route in valid else "chat"


def detect_tool_type(question: str, llm) -> str:
    prompt = f"""
    Tu dois choisir un seul outil parmi :
    - calcul
    - web
    - meteo

    Règles :
    - calcul : opération mathématique, pourcentage, addition, soustraction, division, multiplication, puissance, modulo, fonctions mathématiques
    - web : si la question nécessite une recherche sur le web
    - meteo : si la question concerne la météo d'une ville, d'une région ou d'un pays

    Réponds avec un seul mot :
    calcul
    web
    meteo

    Question :
    {question}

    Outil :
    """
    response = llm.invoke(prompt)
    tool_type = response.content.strip().lower().split()[0]

    if tool_type not in {"calcul", "web", "meteo"}:
        return "calcul"

    return tool_type


def answer_chat(question: str, llm, history=None) -> str:
    history_text = format_history_for_prompt(history, limit=6)

    prompt = f"""
    Tu es un assistant utile et naturel.

    Historique :
    {history_text}

    Question :
    {question}

    Réponse en français :
    """
    response = llm.invoke(prompt)
    return response.content


def route_query(question: str, vectorstore, llm, history=None, k_docs: int = 4) -> Dict[str, Any]:
    has_vectorstore = vectorstore is not None
    route = classify_query(question, llm, has_vectorstore=has_vectorstore)

    if route == "tool":
        tool_type = detect_tool_type(question, llm)

        if tool_type == "calcul":
            expression = extract_math_expression(question)

            if not expression:
                return {
                    "route": "tool",
                    "tool": "calcul",
                    "result": "Impossible d'extraire une expression mathématique valide.",
                    "docs": [],
                    "extra": [],
                }

            try:
                result = calculate(expression)
            except Exception as e:
                result = f"Erreur lors du calcul : {e}"

            return {
                "route": "tool",
                "tool": "calcul",
                "result": result,
                "docs": [],
                "extra": [],
            }

        if tool_type == "web":
            result = search_web(question)
            return {
                "route": "tool",
                "tool": "web",
                "result": result,
                "docs": [],
                "extra": [],
            }

        if tool_type == "meteo":
            city = extract_city_from_question(question)

            try:
                result = weather_tool(city)
            except Exception as e:
                result = f"Erreur météo : {e}"

            return {
                "route": "tool",
                "tool": "meteo",
                "result": result,
                "docs": [],
                "extra": [{"city": city}],
            }

    if route == "doc_search":
        if vectorstore is None:
            return {
                "route": "chat",
                "result": "Je n'ai pas d'index documentaire disponible pour effectuer cette recherche.",
                "docs": [],
                "extra": [],
            }

        results = RAG.search_documents(vectorstore, question, k=k_docs)
        return {
            "route": "doc_search",
            "result": results,
            "docs": [],
            "extra": [],
        }

    if route == "rag":
        if vectorstore is None:
            return {
                "route": "chat",
                "result": "Je n'ai pas d'index documentaire disponible pour répondre à partir des documents.",
                "docs": [],
                "extra": [],
            }

        docs = RAG.retrieve_documents(vectorstore, question, k=k_docs)
        answer = RAG.generate_answer(question, docs, llm, history=history)
        return {
            "route": "rag",
            "result": answer,
            "docs": docs,
            "extra": [],
        }

    answer = answer_chat(question, llm, history=history)
    return {
        "route": "chat",
        "result": answer,
        "docs": [],
        "extra": [],
    }