import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def ingest_pdf(pdf_path, source_name):
    print(f"Reading {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks. Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    print("Uploading to Supabase...")
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        supabase.table("documents").insert({
            "content": chunk,
            "embedding": embedding.tolist(),
            "source": source_name
        }).execute()
        print(f"  Uploaded {i+1}/{len(chunks)}")
    print("Done! Your document is now searchable.")

ingest_pdf("menu.pdf", "restaurant_menu")