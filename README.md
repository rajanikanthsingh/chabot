
Make Sure To Create OpenRouter Key For Accessing The Model And Put them in .env file
And Make to sure system has all the dependencies before starting  the application 

Command To Start Up The Application

uvicorn app:app -- host 0.0.0.0 -- port 8000 -- reload


PDF → Text Extraction → Chunks → Embeddings → FAISS Index
                                                    ↓
User Question → Query Embedding → Similarity Search
                                                    ↓
                        Relevant Chunks (Context) + Question
                                                    ↓
                        Build Prompt → Mistral-7B → Answer



Loads a PDF file (LLM.pdf)
Splits it into chunks
Creates embeddings using sentence transformers
Uses FAISS for fast similarity search
Sends relevant chunks to Mistral AI via OpenRouter
Provides a FastAPI endpoint to chat

Example flow summary:
1. You ask: "What is attention mechanism?"
2. Your question → Embedding: [0.23, -0.45, ...]
3. FAISS finds 3 similar chunks from PDF
4. Build prompt with those 3 chunks as context
5. Send prompt to OpenRouter → Mistral-7B
6. Mistral reads context + answers your question
7. Answer returned to you




Detailed Flow Explanation:
One-Time Setup (When Server Starts):
1. PDF is loaded
2. PDF → Split into chunks
3. Chunks → Converted to embeddings
4. FAISS stores all CHUNK embeddings ← This happens ONCE

Server Starts → uvicorn app:app --reload
                        ↓
┌─────────────────────────────────────────────────┐
│  1. Load PDF: data/LLM.pdf                      │
│  2. Extract Text: "Large Language Models..."    │
│  3. Split into Chunks:                          │
│     - Chunk 1: "Transformers are..."            │
│     - Chunk 2: "Attention mechanism..."         │
│     - Chunk 3: "Self-attention allows..."       │
│     - ... (all chunks)                          │
│                                                  │
│  4. Convert Chunks → Embeddings:                │
│     - Chunk 1 → [0.45, 0.23, -0.12, ...] (384D) │
│     - Chunk 2 → [0.89, -0.34, 0.56, ...] (384D) │
│     - Chunk 3 → [0.41, 0.22, -0.09, ...] (384D) │
│                                                  │
│  5. Store in FAISS Index (RAM):                 │
│     ┌──────────────────────────────────┐        │
│     │  FAISS Index (In Memory)         │        │
│     │  ├─ Chunk 1 embedding             │        │
│     │  ├─ Chunk 2 embedding             │        │
│     │  ├─ Chunk 3 embedding             │        │
│     │  └─ ... all chunk embeddings      │        │
│     └──────────────────────────────────┘        │
└─────────────────────────────────────────────────┘
         ✅ Ready to answer questions!

FAISS Index contains: All the PDF content as vectors (stored in memory)
Every Time  when You Ask a Question:
1. You ask: "What is attention mechanism?"
2. Question → Converted to embedding [0.23, -0.45, ...]
3. FAISS compares question embedding with stored CHUNK embeddings
4. FAISS returns the 3 most similar chunks from PDF(The 3 chunks are NOT the final answer 
    they are the relevant information that will help answer your question.)
Example to Clarify:
The PDF contains:
Chunk 1: "Transformers are neural network architectures introduced in 2017..."
Chunk 2: "Self-attention mechanisms allow models to weigh importance of words..."
Chunk 3: "The attention mechanism computes similarity between query and key vectors..."
Chunk 4: "CNNs are used for image processing with convolutional layers..."
Chunk 5: "RNNs process sequential data one step at a time..."
You ask: "What is attention mechanism?"
FAISS Search:
FAISS compares your question with ALL chunks:

Chunk 3: Similarity = 0.92 ✅ (Directly mentions "attention mechanism")
Chunk 2: Similarity = 0.88 ✅ (Talks about "self-attention")
Chunk 1: Similarity = 0.75 ✅ (Related to transformers)
Chunk 4: Similarity = 0.23 ❌ (About CNNs, not relevant)
Chunk 5: Similarity = 0.19 ❌ (About RNNs, not relevant)

FAISS returns top 3: Chunks 3, 2, 1

5. Build prompt with those 3 chunks as context
Building a prompt = Creating a complete instruction/message that will be sent to Mistral-7B, which includes:
[System instructions (how to behave),
Context (the 3 relevant chunks from your PDF),
User question]

6. Send prompt to OpenRouter → Mistral-7B
The prompt  here is 
[System instructions (how to behave),
Context (the 3 relevant chunks from your PDF),
User question]

7. Mistral reads context + answers your question