def classify_query(question: str, llm) -> str:
    prompt = f"""Tu es un routeur intelligent. Classe la question suivante dans exactement une des catégories ci-dessous.

Catégories :
- rag         : question sur le contenu des documents internes (articles, notes, corpus)
- web         : recherche d'informations sur internet
- doc_search  : retrouver un document ou un passage précis dans le corpus
- summary     : résumer un thème ou un ensemble de documents
- quiz        : générer des questions de révision
- calcul      : calcul arithmétique (addition, soustraction, multiplication, division)
- chat        : salutation ou conversation générale sans besoin de documents ni d'outils

Réponds avec un seul mot parmi : rag, web, doc_search, summary, quiz, calcul, chat

Question : {question}

Catégorie :"""

    response = llm.invoke(prompt)
    category = response.content.strip().lower().split()[0]

    valid = {"rag", "web", "doc_search", "summary", "quiz", "calcul", "chat"}
    return category if category in valid else "rag"


def route_query(question, vectorstore, llm):
    mode = classify_query(question, llm)

    if mode == "web":
        from tools import search_web
        return {"mode": "web", "result": search_web(question)}

    elif mode == "doc_search":
        from tools import search_documents
        return {"mode": "doc_search", "result": search_documents(vectorstore, question)}

    elif mode == "summary":
        from tools import summarize_document
        return {"mode": "summary", "result": summarize_document(vectorstore, llm, question)}

    elif mode == "quiz":
        from tools import make_quiz
        return {"mode": "quiz", "result": make_quiz(vectorstore, llm, question)}

    elif mode == "calcul":
        from tools import calculate
        import re
        match = re.search(r"[\d+\-*/()., ]+", question)
        expression = match.group().strip() if match else question
        return {"mode": "calcul", "result": calculate(expression)}

    elif mode == "chat":
        response = llm.invoke(question)
        return {"mode": "chat", "result": response.content}

    else:  # rag par défaut
        from RAG import answer_with_rag
        return answer_with_rag(vectorstore, question)
