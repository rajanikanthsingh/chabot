import os
import requests
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
from numpy.linalg import norm
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# ----------------------------------
# ENV
# ----------------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# ----------------------------------
# PDF LOADER 
# ----------------------------------
def load_pdf_from_local(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.strip()

# ----------------------------------
# CHUNKING
# ----------------------------------
def chunk_text(text: str, chunk_size=220, overlap=55):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

# ----------------------------------
# EMBEDDINGS
# ----------------------------------
print("🔍 Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded")

def create_embeddings(chunks):
    return embedding_model.encode(chunks, convert_to_numpy=True)

# ----------------------------------
# FAISS
# ----------------------------------
def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# ----------------------------------
# COSINE SIMILARITY
# ----------------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# ----------------------------------
# SEARCH
# ----------------------------------
def search_faiss(query, chunks, embeddings, index, top_k=3, threshold=0.1):
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(query_embedding, top_k)

    results = []
    scores = []

    for idx in indices[0]:
        score = cosine_similarity(query_embedding[0], embeddings[idx])
        if score >= threshold:
            results.append(chunks[idx])
            scores.append(score)

    return results, scores

# ----------------------------------
# INITIALIZE RAG 
# ----------------------------------
PDF_PATH = "data/LLM.pdf"

print("📄 Initializing RAG...")
text = load_pdf_from_local(PDF_PATH)

if not text:
    raise RuntimeError("No text extracted from PDF")

chunks = chunk_text(text)
embeddings = create_embeddings(chunks)
index = create_faiss_index(embeddings)

print(f"✅ RAG ready with {len(chunks)} chunks")


# ----------------------------------
# PROMPT
# ----------------------------------
def build_rag_prompt(context, question):
    return f"""
You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not present, say:
"I don't have enough information from the document."
if the answer is not present do not hallucinate with made-up answers.
do not use any prior knowledge.and rely solely on the provided context.
do not reply with tags such as <s> or </s>.,<OST>,<\\OST>.

Context:
{context}

Question:
{question}

Answer:
""".strip()

# ----------------------------------
# OPENROUTER 
# ----------------------------------
def call_openrouter_llm(prompt):
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistralai/mistral-7b-instruct:free",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
    )

    if response.status_code != 200:
        raise RuntimeError(response.text)

    data = response.json()
    return data["choices"][0]["message"]["content"]

# ----------------------------------
# FASTAPI
# ----------------------------------
app = FastAPI(title="Simple RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    if index is None:
        return {"response": "Knowledge base not ready"}

    retrieved_chunks, _ = search_faiss(
        request.query,
        chunks,
        embeddings,
        index
    )

    if not retrieved_chunks:
        return {"response": "Sorry, I can only answer based on the document."}

    context = "\n\n".join(retrieved_chunks)
    prompt = build_rag_prompt(context, request.query)
    answer = call_openrouter_llm(prompt)

    return {"response": answer}
