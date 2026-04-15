
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


def extract_math_expression(text: str) -> str:
    cleaned = text.replace(",", ".").lower()

    matches = re.findall(
        r"[a-zA-Z_][a-zA-Z0-9_]*|\d+(?:\.\d+)?|[+\-*/%(),]",
        cleaned
    )
    expression = "".join(matches).strip()

    return expression if expression else text.strip()


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



