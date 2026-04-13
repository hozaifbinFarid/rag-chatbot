import os
import requests as req
import numpy as np
from flask import Flask, request, jsonify
from supabase import create_client
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

HF_API_URL = "https://router.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_TOKEN", "")

def get_embedding(text):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = req.post(
        HF_API_URL,
        headers=headers,
        json={"inputs": text, "options": {"wait_for_model": True}}
    )
    result = response.json()
    if isinstance(result, list):
        if isinstance(result[0], float):
            return result
        if isinstance(result[0], list):
            arr = np.array(result)
            return arr.mean(axis=0).tolist()
    raise ValueError(f"Unexpected response: {result}")

def search_documents(query, top_k=6):
    query_embedding = get_embedding(query)
    result = supabase.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_count": top_k
    }).execute()
    return [row["content"] for row in result.data]

def answer_question(question):
    context_chunks = search_documents(question)
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}
Answer:"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        answer = answer_question(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return jsonify({"status": "RAG Chatbot is running!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))