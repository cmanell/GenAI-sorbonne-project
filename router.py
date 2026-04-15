import os

from langchain_mistralai import ChatMistralAI
from tools import search_web, search_documents, summarize_document, make_quiz, calculate, weather_tool
from RAG import answer_with_rag


def get_llm(model: str = "mistral-small-latest") -> ChatMistralAI:
    return ChatMistralAI(model=model, temperature=0, api_key=os.getenv("MISTRAL_API_KEY"))


def classify_query(question: str, llm) -> str:
    prompt = f"""Tu es un routeur intelligent. Classe la question suivante dans exactement une des catégories ci-dessous.

Catégories :
- rag         : question sur le contenu des documents internes (articles, notes, corpus)
- web         : recherche d'informations sur internet
- doc_search  : retrouver un document ou un passage précis dans le corpus
- summary     : résumer un thème ou un ensemble de documents
- quiz        : générer des questions de révision
- calcul      : calcul arithmétique (addition, soustraction, multiplication, division)
- météo       : demande de météo ou de température pour une ville
- chat        : salutation ou conversation générale sans besoin de documents ni d'outils

Réponds avec un seul mot parmi : rag, web, doc_search, summary, quiz, calcul, météo, chat

Question : {question}

Catégorie :"""

    response = llm.invoke(prompt)
    category = response.content.strip().lower().split()[0]

    valid = {"rag", "web", "doc_search", "summary", "quiz", "calcul", "météo", "chat"}
    return category if category in valid else "rag"


def extract_expression(question: str, llm) -> str:
    prompt = f"""Transforme cette question en une expression mathématique Python évaluable.

Règles strictes :
- Utilise ** pour les puissances (jamais ^ ni ²)
- Utilise pi pour le nombre pi (jamais π ni 3.14...)
- Utilise sqrt(), log(), exp(), sin(), cos() pour les fonctions
- Rappel : surface d'un cercle = pi * r**2, périmètre = 2 * pi * r
- Réponds UNIQUEMENT avec l'expression, sans texte ni explication

Question : {question}

Expression :"""
    response = llm.invoke(prompt)
    return response.content.strip()


def extract_city(question: str, llm) -> str:
    prompt = f"""Extrait uniquement le nom de la ville dans cette question. Réponds avec juste le nom de la ville, sans ponctuation ni explication.

Question : {question}

Ville :"""
    response = llm.invoke(prompt)
    return response.content.strip()


def route_query(question, vectorstore, llm):
    mode = classify_query(question, llm)

    if mode == "web":
        return {"mode": "web", "result": search_web(question)}

    elif mode == "doc_search":
        return {"mode": "doc_search", "result": search_documents(vectorstore, question)}

    elif mode == "summary":
        return {"mode": "summary", "result": summarize_document(vectorstore, llm, question)}

    elif mode == "quiz":
        return {"mode": "quiz", "result": make_quiz(vectorstore, llm, question)}

    elif mode == "calcul":
        expression = extract_expression(question, llm)
        return {"mode": "calcul", "result": calculate(expression)}

    elif mode == "météo":
        city = extract_city(question, llm)
        return {"mode": "météo", "result": weather_tool(city)}

    elif mode == "chat":
        response = llm.invoke(question)
        return {"mode": "chat", "result": response.content}

    else:  # rag par défaut
        return answer_with_rag(vectorstore, question)
