from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from uuid import uuid4
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import os
import re
import openai  # nếu dùng OpenAI để sinh câu trả lời

# === CONFIG ===
openai.api_key = "sk-your-api-key-here"  # hoặc dùng biến môi trường
# ==============

app = FastAPI()

# Load embedding model
model = SentenceTransformer("intfloat/multilingual-e5-base")

# Load transformers pipelines
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
title_generator = pipeline("text2text-generation", model="google/flan-t5-base")

# Mongo & Qdrant
mongo_client = MongoClient("mongodb://admin:admin123@localhost:27017")
mongo_db = mongo_client["books"].chunks

qdrant = QdrantClient("localhost", port=6333)
collection_name = "book_chunks"

# Create collection if needed
if not qdrant.collection_exists(collection_name):
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )


# === TEXT UTILS ===

def split_sentences(text):
    return re.findall(r'[^.!?\n]+[.!?\n]', text)

def chunk_text(text, max_chunk_words=300, threshold=0.75):
    sentences = split_sentences(text)
    chunks = []
    current_chunk = []
    current_len = 0
    prev_embedding = None

    for sent in sentences:
        words = sent.split()
        if not words:
            continue

        current_chunk.append(sent)
        current_len += len(words)
        joined_chunk = " ".join(current_chunk)

        embedding = model.encode(joined_chunk)

        if prev_embedding is not None:
            sim = cosine_similarity([embedding], [prev_embedding])[0][0]
            if sim < threshold or current_len >= max_chunk_words:
                chunks.append(joined_chunk)
                current_chunk = []
                current_len = 0
                prev_embedding = None
                continue

        prev_embedding = embedding

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def summarize_text(text):
    try:
        return summarizer(text[:1024], max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    except Exception:
        return text[:300]

def generate_title(text):
    try:
        prompt = f"Generate a short title for this text:\n{text[:500]}"
        return title_generator(prompt, max_length=20, do_sample=False)[0]["generated_text"]
    except Exception:
        return "Untitled Chunk"

def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""Dựa trên các đoạn sau, hãy trả lời câu hỏi: "{question}"\n\n{context}\n\nTrả lời:"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return "Không thể sinh câu trả lời."

# === ROUTES ===

@app.post("/upload-book")
async def upload_book(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return JSONResponse(content={"error": "Only PDF files allowed"}, status_code=400)

    temp_path = f"temp_{uuid4()}.pdf"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        reader = PdfReader(temp_path)
        all_text = " ".join([page.extract_text() or "" for page in reader.pages])
        chunks = chunk_text(all_text)

        points = []
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid4())
            summary = summarize_text(chunk)
            title = generate_title(chunk)
            vector = model.encode(summary).tolist()

            mongo_db.insert_one({
                "_id": chunk_id,
                "chunk_index": i,
                "summary": summary,
                "title": title,
                "chunk_text": chunk,
                "source_file": file.filename
            })

            points.append(PointStruct(
                id=chunk_id,
                vector=vector,
                payload={
                    "summary": summary,
                    "title": title,
                    "source_file": file.filename,
                    "chunk_index": i
                }
            ))

        qdrant.upsert(collection_name=collection_name, points=points)
        return {"message": f"Uploaded and processed {len(chunks)} chunks from {file.filename}"}

    finally:
        os.remove(temp_path)

@app.get("/ask")
async def ask_question(q: str = Query(..., alias="question")):
    question_embedding = model.encode(q).tolist()

    search_result = qdrant.search(
        collection_name=collection_name,
        query_vector=question_embedding,
        limit=5
    )

    ids = [point.id for point in search_result]
    docs = {doc["_id"]: doc for doc in mongo_db.find({"_id": {"$in": ids}})}

    chunks_text = [docs.get(hit.id, {}).get("chunk_text", "") for hit in search_result]
    final_answer = generate_answer(q, chunks_text)

    return {
        "question": q,
        "generated_answer": final_answer,
        "answers": [
            {
                "score": float(hit.score),
                "title": docs.get(hit.id, {}).get("title", ""),
                "summary": docs.get(hit.id, {}).get("summary", ""),
                "text": docs.get(hit.id, {}).get("chunk_text", ""),
                "source_file": docs.get(hit.id, {}).get("source_file", ""),
                "chunk_index": docs.get(hit.id, {}).get("chunk_index", -1)
            }
            for hit in search_result
        ]
    }
