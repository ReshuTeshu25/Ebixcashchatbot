import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from ddgs import DDGS
from flask import Flask, request, jsonify
from flask_cors import CORS

DATA_FILE = "ebix_data.txt"

def web_search(query):
    results = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(r.get("body", ""))

    except Exception as e:
        print("Search Error:", e)
        return "No results found"

    return "\n".join(results)

def load_vectorstore():

    if not os.path.exists(DATA_FILE):
        open(DATA_FILE, "w").close()

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_text(text)

    if len(docs) == 0:
        return None

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    return FAISS.from_texts(docs, embeddings)

vectorstore = load_vectorstore()

llm = OllamaLLM(model="mistral")

chat_history = []

def ebix_rag(query):

    global vectorstore

    context = ""

    if vectorstore is not None:
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n".join(
            [doc.page_content for doc in docs]
        )

    latest_info = web_search(
        "site:ebixcash.com " + query
    )

    if (
        latest_info == "No results found"
        or len(latest_info.strip()) < 30
    ):
        latest_info = web_search(
            "Ebix Cash " + query
        )

    if (
        latest_info
        and latest_info != "No results found"
    ):

        with open(
            DATA_FILE,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                "\n\n"
                + "=" * 50 +
                "\n[AUTO UPDATED]\n"
                + f"Question: {query}\n"
                + latest_info +
                "\n"
            )

        vectorstore = load_vectorstore()

    final_context = f"""
    Existing Knowledge:
    {context}

    Latest Information:
    {latest_info}
    """

    prompt = f"""
You are an EBIX Cash AI Assistant.

RULES:
1. Always prioritize Latest Information.
2. Prefer official EbixCash information.
3. If information is unavailable, use web information.
4. Never make up facts.
5. Give concise and professional answers.

Context:
{final_context}

Question:
{query}

Answer:
"""

    return llm.invoke(prompt)

def general_ai(query):

    prompt = f"""
You are a helpful AI assistant.

Chat History:
{chat_history}

User:
{query}

Answer:
"""

    return llm.invoke(prompt)

def router(query):

    q = query.lower()

    ebix_keywords = [
        "ebix",
        "ebix cash",
        "forex",
        "remittance",
        "travel card",
        "money transfer",
        "insurance",
        "payment",
        "gift card",
        "prepaid card"
    ]

    if any(word in q for word in ebix_keywords):
        return ebix_rag(query)

    elif any(
        word in q
        for word in [
            "news",
            "today",
            "current",
            "latest",
            "price",
            "rate",
            "currency"
        ]
    ):
        return web_search(query)

    elif (
        "search online" in q
        or "search internet" in q
        or "google" in q
    ):
        return web_search(query)

    else:
        return general_ai(query)

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():

    data = request.json

    user_input = data.get(
        "message",
        ""
    )

    response = router(user_input)

    chat_history.append(
        f"User: {user_input}"
    )

    chat_history.append(
        f"Bot: {response}"
    )

    return jsonify({
        "response": response
    })

if __name__ == "__main__":

    print(
        "EbixCash AI Server Running..."
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )