
import re
import os
import requests
import math

from ddgs import DDGS
from langchain_mistralai import ChatMistralAI


def get_llm(model_name: str = "mistral-small-latest"):
    return ChatMistralAI(
        model=model_name,
        temperature=0,
        api_key=os.getenv("MISTRAL_API_KEY"),
    )


SAFE_GLOBALS = {
    "__builtins__": None,
}

SAFE_LOCALS = {
    "exp": math.exp,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
    "pi": math.pi,
    "e": math.e,
}

ALLOWED_FUNCTION_NAMES = {
    "exp", "sqrt", "sin", "cos", "tan", "log", "log10", "abs", "pi", "e"
}

def extract_math_expression(text: str) -> str:
    cleaned = text.replace(",", ".").lower()

    tokens = re.findall(
        r"[a-zA-Z_][a-zA-Z0-9_]*|\d+(?:\.\d+)?|[+\-*/%(),]",
        cleaned
    )

    filtered_tokens = []
    for token in tokens:
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            filtered_tokens.append(token)
        elif token in {"+", "-", "*", "/", "%", "(", ")", ","}:
            filtered_tokens.append(token)
        elif token in ALLOWED_FUNCTION_NAMES:
            filtered_tokens.append(token)

    expression = "".join(filtered_tokens).strip()
    return expression


def calculate(expression: str) -> str:
    try:
        result = eval(expression, SAFE_GLOBALS, SAFE_LOCALS)
        return str(result)
    except Exception as e:
        return f"Erreur de calcul : {e}"



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


def extract_city_from_question(question: str) -> str:
    q = question.strip()

    patterns = [
        r"(?:météo|meteo|temps)\s+(?:à|a|de|d'|sur)\s+([A-Za-zÀ-ÿ\- ]+)",
        r"(?:quel temps fait[- ]il|quelle est la météo|quelle est la meteo)\s+(?:à|a|de|d'|sur)\s+([A-Za-zÀ-ÿ\- ]+)",
        r"(?:météo|meteo)\s+([A-Za-zÀ-ÿ\- ]+)",
        r"(?:météo|meteo).*(?:sur|à|a|de|d')\s+([A-Za-zÀ-ÿ\- ]+)",
    ]

    city = None

    for pattern in patterns:
        match = re.search(pattern, q, re.IGNORECASE)
        if match:
            city = match.group(1).strip(" ?.!,;:")
            break

    if city is None:
        city = q.strip(" ?.!,;:")

    # Supprime les mots parasites
    city = re.sub(
        r"\b(aujourd'hui|aujourdhui|today|maintenant|ce soir|demain|sur)\b",
        "",
        city,
        flags=re.IGNORECASE,
    )
    city = re.sub(r"^(de|d'|à|a|sur)\s+", "", city, flags=re.IGNORECASE)
    city = re.sub(r"\s+", " ", city).strip()

    return city



def weather_tool(city: str) -> str:
    try:
        resp = requests.get(f"https://wttr.in/{city}?format=3", timeout=5)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        return f"Weather error: {e}"
    

def web_summary(llm, query: str):
    results = search_web(query)
    text = "\n\n".join(r["snippet"] for r in results if r.get("snippet"))

    prompt = f"""
    Tu es un assistant.
    Résume les informations suivantes pour répondre à la question.

    Informations :
    {text}

    Question :
    {query}

    Réponse en français :
    """
    response = llm.invoke(prompt)
    return response.content, results



