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


def is_corpus_related(question: str, llm, corpus_files: list) -> bool:
    """Vérifie si la question attend une réponse rédigée à partir du corpus."""
    file_list = "\n".join(f"- {f}" for f in corpus_files)
    prompt = f"""Tu disposes d'un corpus de documents :
{file_list}

La question suivante attend-elle une réponse rédigée à partir du contenu de ces documents ?
Réponds NON si :
- c'est une demande de reformulation, correction ou amélioration d'un texte fourni par l'utilisateur
- c'est une demande de recherche d'un passage ou d'une source précise dans les documents
- c'est une conversation générale (salutation, remerciement, etc.)
Réponds OUI uniquement si la question appelle une réponse construite à partir du contenu des documents.

Réponds uniquement par oui ou non.

Question : {question}

Réponse :"""
    response = llm.invoke(prompt)
    return response.content.strip().lower().startswith("oui")


def classify_query(question: str, llm, has_vectorstore: bool = True, corpus_files: list = None) -> str:
    if not has_vectorstore:
        rag_hint = "Aucun index documentaire n'est disponible : n'utilise jamais rag ni doc_search."
    elif corpus_files:
        file_list = "\n".join(f"- {f}" for f in corpus_files)
        rag_hint = (
            f"Un corpus de documents est indexé. Voici les fichiers disponibles :\n{file_list}\n\n"
            "Utilise rag si la question porte sur un sujet couvert par ces documents.\n"
            "Utilise doc_search si l'utilisateur cherche un passage ou une source sans vouloir une réponse rédigée."
        )
    else:
        rag_hint = "Un corpus de documents est indexé. Utilise rag si la question semble liée aux documents chargés."

    prompt = f"""
    Tu es un routeur intelligent.

    Ta tâche est de choisir UNE SEULE route parmi :
    - tool
    - doc_search
    - rag
    - chat

    Règles :
    - tool : calcul mathématique, météo, ou recherche web sur un fait extérieur au corpus (actualité, personne publique, événement mondial, définition générale).
    - doc_search : l'utilisateur veut retrouver un passage, une source, une page dans les documents sans demander de réponse rédigée.
    - rag : la question porte sur un sujet traité dans le corpus (voir liste ci-dessous). Privilégie rag dès qu'un des fichiers du corpus semble pertinent.
    - chat : conversation générale, reformulation, aide rédactionnelle, ou sujet sans lien avec le corpus ni besoin d'outil.

    Important :
    - Si le sujet de la question correspond à un fichier du corpus, choisis rag, même si la question ne mentionne pas explicitement "les documents".
    - Une simple conversation comme "bonjour", "merci", "peux-tu reformuler" doit aller vers chat.

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


def route_query(question: str, vectorstore, llm, history=None, k_docs: int = 4, folder_path: str = "data") -> Dict[str, Any]:
    has_vectorstore = vectorstore is not None
    corpus_files = [p.name for p in RAG.list_supported_files(folder_path)] if has_vectorstore else []

    if has_vectorstore and corpus_files and is_corpus_related(question, llm, corpus_files):
        route = "rag"
    else:
        route = classify_query(question, llm, has_vectorstore=has_vectorstore, corpus_files=corpus_files)

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
            city = extract_city_from_question(question, llm=llm)
        
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