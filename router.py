from typing import Dict, Any
import RAG

from memory import format_history_for_prompt
from tools import (
    extract_math_expression,
    calculate,
    search_web,
    weather_tool,
    extract_city_from_question,
)


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
    - tool : si la question nécessite l'usage d'un outil, comme un calcul, une météo, une recherche web, ou une recherche factuelle sur une personne, un lieu, un événement, un pays, une date, une définition, ou une information générale.
    - doc_search : si l'utilisateur veut retrouver un document, une source, un passage, une page, ou chercher dans les documents sans demander une réponse rédigée.
    - rag : si l'utilisateur pose une question dont la réponse doit être trouvée dans les documents.
    - chat : pour une conversation générale, reformulation, aide rédactionnelle, avis, ou discussion qui ne nécessite ni outil ni documents.

    Important :
    - Une question comme "qui est Nelson Mandela", "qu'est-ce que la photosynthèse", "quelle est la capitale du Pérou", "parle-moi de Victor Hugo" doit aller vers tool.
    - Une simple conversation comme "bonjour", "merci", "peux-tu reformuler", "explique-moi simplement ce texte" doit aller vers chat.

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


def answer_tool_result(question: str, tool_name: str, tool_result: str, llm, history=None) -> str:
    history_text = format_history_for_prompt(history, limit=6)

    prompt = f"""
    Tu es un assistant utile et naturel.

    Historique :
    {history_text}

    L'utilisateur a posé cette question :
    {question}

    Un outil a été utilisé :
    {tool_name}

    Résultat brut de l'outil :
    {tool_result}

    Ta tâche :
    - reformuler ce résultat en une réponse naturelle en français
    - être clair et concis
    - ne pas inventer d'information absente du résultat
    - si le résultat contient une erreur, expliquer simplement qu'il y a eu un problème

    Réponse :
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
                tool_output = "Impossible d'extraire une expression mathématique valide."
            else:
                try:
                    tool_output = calculate(expression)
                except Exception as e:
                    tool_output = f"Erreur lors du calcul : {e}"

            final_answer = answer_tool_result(
                question=question,
                tool_name="calcul",
                tool_result=str(tool_output),
                llm=llm,
                history=history,
            )

            return {
                "route": "tool",
                "tool": "calcul",
                "result": final_answer,
                "docs": [],
                "extra": [{
                    "raw_result": str(tool_output),
                    "expression": expression if expression else ""
                }],
            }

        if tool_type == "web":
            try:
                web_results = search_web(question)
        
                if not web_results:
                    tool_output = "Aucun résultat trouvé."
                else:
                    formatted_results = []
                    for r in web_results:
                        title = r.get("title", "")
                        snippet = r.get("snippet", "")
                        link = r.get("link", "")
                        formatted_results.append(
                            f"Titre : {title}\n"
                            f"Résumé : {snippet}\n"
                            f"Lien : {link}"
                        )
        
                    tool_output = "\n\n".join(formatted_results)
        
            except Exception as e:
                tool_output = f"Erreur de recherche web : {e}"
        
            final_answer = answer_tool_result(
                question=question,
                tool_name="web",
                tool_result=str(tool_output),
                llm=llm,
                history=history,
            )

            return {
                "route": "tool",
                "tool": "web",
                "result": final_answer,
                "docs": [],
                "extra": web_results if isinstance(web_results, list) else [],
            }

        if tool_type == "meteo":
            city = extract_city_from_question(question)
            print("VILLE EXTRAITE =", city)
        
            if not city:
                tool_output = "Impossible de déterminer la ville demandée."
            else:
                try:
                    tool_output = weather_tool(city)
                except Exception as e:
                    tool_output = f"Erreur météo : {e}"
        
            final_answer = answer_tool_result(
                question=question,
                tool_name="meteo",
                tool_result=str(tool_output),
                llm=llm,
                history=history,
            )
        
            return {
                "route": "tool",
                "tool": "meteo",
                "result": final_answer,
                "docs": [],
                "extra": [
                    {"city": city},
                    {"raw_result": str(tool_output)},
                ],
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