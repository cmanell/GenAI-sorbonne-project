import ast
import operator as op
import re
import os
from langchain_mistralai import ChatMistralAI
import re
import requests
import RAG
from ddgs import DDGS



_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    ast.Mod: op.mod,
}


def get_llm(model_name: str = "mistral-small-latest"):
    return ChatMistralAI(
        model=model_name,
        temperature=0,
        api_key=os.getenv("MISTRAL_API_KEY"),
    )


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Constante non autorisée.")

    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        operator_type = type(node.op)

        if operator_type not in _ALLOWED_OPERATORS:
            raise ValueError("Opérateur non autorisé.")

        return _ALLOWED_OPERATORS[operator_type](left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        operator_type = type(node.op)

        if operator_type not in _ALLOWED_OPERATORS:
            raise ValueError("Opérateur unaire non autorisé.")

        return _ALLOWED_OPERATORS[operator_type](operand)

    raise ValueError("Expression non autorisée.")


def extract_math_expression(text: str) -> str:
    cleaned = text.replace(",", ".")
    matches = re.findall(r"[0-9\.\+\-\*\/%\(\)\s]+", cleaned)
    expression = "".join(matches).strip()
    return expression if expression else text.strip()


def calculate(expression: str) -> str:
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_eval(parsed.body)
        return str(result)
    except Exception as e:
        return f"Erreur de calcul : {e}"
    


def extract_city_from_weather_question(question: str) -> str:
    patterns = [
        r"(?:à|a|pour|sur)\s+([A-Za-zÀ-ÿ' -]+)$",
        r"météo\s+([A-Za-zÀ-ÿ' -]+)$",
    ]

    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" ?!.")

    return "Paris"


def weather_tool(city: str) -> str:
    try:
        resp = requests.get(f"https://wttr.in/{city}?format=3", timeout=5)
        resp.raise_for_status()
        return resp.text.strip()
    except Exception as e:
        return f"Erreur météo : {e}"
    


def search_web(query: str, max_results: int = 5):
    results = []

    with DDGS() as ddgs:
        found = ddgs.text(query, max_results=max_results)
        for r in found:
            results.append(
                {
                    "title": r.get("title", ""),
                    "link": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
            )

    return results


def web_summary(llm, query: str):
    results = search_web(query)
    text = "\n\n".join(r["snippet"] for r in results if r.get("snippet"))

    prompt = f"""
    Résume ces informations :

    {text}

    Question : {query}
    """
    response = llm.invoke(prompt)
    return response.content, results


def summarize_document(vectorstore, llm, query: str, k: int = 4):
    docs = RAG.retrieve_documents(vectorstore, query, k=k)

    text = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
    Résume en français :

    {text}
    """
    response = llm.invoke(prompt)
    return response.content, docs
