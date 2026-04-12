import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from sentence_transformers import SentenceTransformer
from supabase import create_client
from groq import Groq

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")

def search_documents(query, top_k=3):
    query_embedding = model.encode(query).tolist()
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

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    print(f"Question received: {user_question}")
    answer = answer_question(user_question)
    print(f"Answer: {answer}")
    await update.message.reply_text(answer)

if __name__ == "__main__":
    print("Bot is starting...")
    app = ApplicationBuilder().token(os.getenv("TELEGRAM_TOKEN")).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot is running! Send a message on Telegram.")
    app.run_polling()