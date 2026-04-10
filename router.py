def route_query(question, vectorstore, llm):
    q = question.lower()

    # outil recherche web
    if "internet" in q or "web" in q or "cherche sur internet" in q:
        from tools import search_web
        return {
            "mode": "web",
            "result": search_web(question)
        }

    # outil recherche de documents
    elif "retrouve" in q or "trouve le document" in q or "dans quel document" in q:
        from tools import search_documents
        return {
            "mode": "doc_search",
            "result": search_documents(vectorstore, question)
        }

    # outil résumé
    elif "résume" in q or "resume" in q:
        from tools import summarize_document
        return {
            "mode": "summary",
            "result": summarize_document(vectorstore, llm, question)
        }

    # outil quiz
    elif "quiz" in q or "question de révision" in q:
        from tools import make_quiz
        return {
            "mode": "quiz",
            "result": make_quiz(vectorstore, llm, question)
        }

    # sinon RAG
    else:
        from rag import answer_with_rag
        return answer_with_rag(vectorstore, question)