from typing import Dict, Any
import RAG
from tools import (
    extract_math_expression, calculate, 
    extract_city_from_weather_question, 
    weather_tool, web_summary, summarize_document)


def classify_query(question: str, llm) -> str:
    prompt = f"""Tu es un routeur intelligent.
Classe la question suivante dans exactement une seule catégorie.

Catégories :
- rag
- web
- doc_search
- summary
- quiz
- calcul
- weather
- chat

Réponds avec un seul mot parmi :
rag, web, doc_search, summary, quiz, calcul, weather, chat

Question : {question}

Catégorie :
"""
    response = llm.invoke(prompt)
    category = response.content.strip().lower().split()[0]

    valid = {"rag", "web", "doc_search", "summary", "quiz", "calcul", "weather", "chat"}
    return category if category in valid else "rag"


def route_query(question: str, vectorstore, llm, history=None, k_docs: int = 4) -> Dict[str, Any]:
    mode = classify_query(question, llm)

    if mode == "web":
        answer, results = web_summary(llm, question)
        return {"mode": "web", "result": answer, "extra": results}

    if mode == "doc_search":
        results = RAG.search_documents(vectorstore, question, k=k_docs)
        return {"mode": "doc_search", "result": results}

    if mode == "summary":
        summary, docs = summarize_document(vectorstore, llm, question, k=k_docs)
        return {"mode": "summary", "result": summary, "docs": docs}

    if mode == "quiz":
        quiz, docs = make_quiz(vectorstore, llm, question, k=k_docs)
        return {"mode": "quiz", "result": quiz, "docs": docs}

    if mode == "calcul":
        expression = extract_math_expression(question)
        result = calculate(expression)
        return {"mode": "calcul", "result": result}

    if mode == "weather":
        city = extract_city_from_weather_question(question)
        result = weather_tool(city)
        return {"mode": "weather", "result": result}

    if mode == "chat":
        response = llm.invoke(question)
        return {"mode": "chat", "result": response.content}

    docs = RAG.retrieve_documents(vectorstore, question, k=k_docs)
    answer = RAG.generate_answer(question, docs, llm, history=history)
    return {"mode": "rag", "result": answer, "docs": docs}