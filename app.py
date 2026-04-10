from rag import load_documents, split_documents, build_or_load_vectorstore, generate_answer
from router import route_query
from langchain_ollama import ChatOllama

def main():
    documents = load_documents("data")
    chunks = split_documents(documents)
    vectorstore = build_or_load_vectorstore(chunks)

    llm = ChatOllama(model="mistral", temperature=0)

    print("Assistant prêt. Tape 'quit' pour quitter.")

    while True:
        question = input("\nQuestion : ").strip()

        if question.lower() in {"quit", "exit", "q"}:
            break

        result = route_query(question, vectorstore, llm)

        if result["mode"] == "rag":
            print("\nRéponse :")
            print(result["answer"])

        elif result["mode"] == "doc_search":
            print("\nDocuments trouvés :")
            for r in result["result"]:
                print(f"- {r['source']} | page {r['page']}")
                print(f"  {r['excerpt']}\n")

        elif result["mode"] == "web":
            print("\nRésultats web :")
            for r in result["result"]:
                print(f"- {r['title']}")
                print(f"  {r['link']}")
                print(f"  {r['snippet']}\n")

        else:
            print("\nRésultat :")
            print(result["result"])

if __name__ == "__main__":
    main()