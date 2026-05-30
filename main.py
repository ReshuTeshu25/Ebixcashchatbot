import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from ddgs import DDGS
from langchain_ollama import OllamaEmbeddings
from flask import Flask, request, jsonify
from flask_cors import CORS
DATA_FILE = "ebix_data.txt"

def web_search(query):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(r["body"])
    except:
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
    if vectorstore is None:
        context = ""
    else:
        docs = vectorstore.similarity_search(query, k=2)
        context = "\n".join([d.page_content for d in docs])
    if len(context.strip()) < 50:
        web_data = web_search("Ebix Cash " + query)
        with open(DATA_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n[Source: Web | Verified: No]\n{web_data}\n")
        vectorstore = load_vectorstore()
        context = web_data
    prompt = f"""
    You are an expert assistant for EBIX Cash.

    Context:
    {context}

    Question: {query}
    Answer:
    """
    return llm.invoke(prompt)

def general_ai(query):
    prompt = f"""
    You are a helpful AI assistant.

    Chat history:
    {chat_history}

    User: {query}
    Answer:
    """
    return llm.invoke(prompt)

def router(query):
    q = query.lower()
    if "search on internet" in q or "search online" in q or "google" in q:
        clean_query = query.replace("search on internet", "").replace("search online", "").replace("google", "")
        return web_search(clean_query)
    elif "ebix" in q or "ebix cash" in q:
        return ebix_rag(query)
    elif any(word in q for word in ["news", "rate", "price", "today", "currency","new","current "]):
        return web_search(query)
    else:
        return general_ai(query)
app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():

    data = request.json
    user_input = data.get("message", "")

    response = router(user_input)

    chat_history.append(f"User: {user_input}")
    chat_history.append(f"Bot: {response}")

    return jsonify({
        "response": response
    })


if __name__ == "__main__":
    print("EbixCash AI Server Running...")
    app.run(host="0.0.0.0", port=5000, debug=True)