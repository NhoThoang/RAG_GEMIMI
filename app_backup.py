from fastapi import FastAPI, File, UploadFile, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Literal, Optional, List
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from uuid import uuid4
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import google.generativeai as genai
import torch
import redis
import os
import re
import json

# === CONFIG GOOGLE GEMINI ===
genai.configure(api_key="AIzaSyCJmPe82EYPXsMEVaIgt2oDftOtb6I3Et0")  # <== Bạn cần thay API key thật

# === DEVICE ===
device = 0 if torch.cuda.is_available() else -1
device_str = "cuda" if torch.cuda.is_available() else "cpu"
print("▶ Using device:", device_str)

app = FastAPI()

# === EMBEDDING & SUMMARIZER MODELS ===
model = SentenceTransformer("intfloat/multilingual-e5-base", device=device_str)

summarizer_tokenizer = AutoTokenizer.from_pretrained("pengold/t5-vietnamese-summarization")
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("pengold/t5-vietnamese-summarization")
summarizer = pipeline("text2text-generation", model=summarizer_model, tokenizer=summarizer_tokenizer, device=device)

# === DATABASE ===
mongo_client = MongoClient("mongodb://admin:admin123@localhost:27017")
mongo_db = mongo_client["books"].chunks

qdrant = QdrantClient("localhost", port=6333)
collection_name = "book_chunks"

if not qdrant.collection_exists(collection_name):
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

# === REDIS ===
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# === TEXT UTILITIES ===

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
        result = summarizer(text[:1024], max_new_tokens=100, do_sample=False)
        return result[0]['generated_text']
    except Exception:
        return text[:300]

# === PROMPT TEMPLATES ===

PROMPT_TEMPLATES = {
    "main_topic": "Tóm tắt nội dung chính của tài liệu sau:\n{context}",
    "year_of_event": "Dựa vào đoạn văn sau, cho biết năm diễn ra sự kiện liên quan:\n{context}\n\nCâu hỏi: {question}",
    "number_question": "Đếm và cho biết số lượng theo yêu cầu sau:\n{context}\n\nCâu hỏi: {question}",
    "who_is": "Cho biết thông tin về nhân vật hoặc sự kiện được hỏi:\n{context}\n\nCâu hỏi: {question}",
    "compare": "So sánh nội dung theo yêu cầu sau:\n{context}\n\nCâu hỏi: {question}",
    "why_reason": "Giải thích lý do theo câu hỏi sau:\n{context}\n\nCâu hỏi: {question}",
    "how_process": "Mô tả quá trình, cách làm hoặc diễn biến:\n{context}\n\nCâu hỏi: {question}",
    "default": "Trả lời câu hỏi sau dựa trên đoạn văn:\n{context}\n\nCâu hỏi: {question}"
}

def classify_question(question: str) -> str:
    q = question.lower()
    if "nội dung chính" in q or "tóm tắt" in q:
        return "main_topic"
    if "năm" in q:
        return "year_of_event"
    if "bao nhiêu" in q or "số lượng" in q:
        return "number_question"
    if "ai là" in q or q.startswith("ai "):
        return "who_is"
    if "khác" in q or "so sánh" in q:
        return "compare"
    if "tại sao" in q or "vì sao" in q:
        return "why_reason"
    if "như thế nào" in q or "làm sao" in q:
        return "how_process"
    return "default"

def normalize_question(text):
    return re.sub(r"\s+", " ", text.strip().lower())

def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
    intent = classify_question(question)
    prompt_template = PROMPT_TEMPLATES.get(intent, PROMPT_TEMPLATES["default"])
    prompt = prompt_template.format(context=context, question=question)
    print(f"[Prompt]: {prompt[:200]}...")

    try:
        response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Không thể trả lời câu hỏi này."

# === SCHEMA ===

class AskRequest(BaseModel):
    question: str
    type: Literal["new", "followup"]
    history: Optional[List[dict]] = None

# === ROUTES ===

@app.post("/upload-book")
async def upload_book(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return JSONResponse(content={"error": "Only PDF files allowed"}, status_code=400)

    temp_path = f"temp_{uuid4()}.pdf"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        reader = PdfReader(temp_path)
        chunks = []
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            for chunk in chunk_text(page_text):
                chunks.append({"text": chunk, "page": page_num + 1})

        points = []
        for i, item in enumerate(chunks):
            chunk_id = str(uuid4())
            summary = summarize_text(item["text"])
            vector = model.encode(summary).tolist()

            # Save to Mongo
            mongo_db.insert_one({
                "_id": chunk_id,
                "chunk_index": i,
                "summary": summary,
                "chunk_text": item["text"],
                "source_file": file.filename,
                "page": item["page"]
            })

            # Save to Qdrant
            points.append(PointStruct(
                id=chunk_id,
                vector=vector,
                payload={
                    "summary": summary,
                    "source_file": file.filename,
                    "chunk_index": i,
                    "page": item["page"]
                }
            ))

        qdrant.upsert(collection_name=collection_name, points=points)
        return {"message": f"Uploaded and processed {len(chunks)} chunks from {file.filename}"}
    finally:
        os.remove(temp_path)

@app.post("/ask")
async def ask_question(request: AskRequest = Body(...)):
    q = request.question.strip()
    norm_q = normalize_question(q)

    # NEW question: skip cache
    if request.type != "new":
        # CHECK CACHE
        cached = redis_client.get(f"q:{norm_q}")
        if cached:
            print("✅ Cache hit")
            cached_data = json.loads(cached)
            return {
                "question": q,
                "generated_answer": cached_data["answer"],
                "answers": cached_data["sources"]
            }

    # Search in vector DB
    question_embedding = model.encode(q).tolist()
    search_result = qdrant.search(
        collection_name=collection_name,
        query_vector=question_embedding,
        limit=5
    )

    ids = [point.id for point in search_result]
    docs = {doc["_id"]: doc for doc in mongo_db.find({"_id": {"$in": ids}})}
    chunks_text = [docs.get(hit.id, {}).get("chunk_text", "") for hit in search_result]

    # Build prompt
    if request.type == "followup" and request.history:
        history_text = "\n\n".join(
            f"Hỏi: {h['question']}\nĐáp: {h['answer']}" for h in request.history
        )
        prompt = f"""Lịch sử hội thoại:
{history_text}

Văn bản tham khảo:
{'\n\n'.join(chunks_text)}

Câu hỏi hiện tại: {q}
"""
        try:
            response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
            final_answer = response.text.strip()
        except Exception as e:
            print("Gemini error:", e)
            final_answer = "Không thể trả lời câu hỏi này."
    else:
        final_answer = generate_answer(q, chunks_text)

    source_info = [
        {
            "score": float(hit.score),
            "summary": docs.get(hit.id, {}).get("summary", ""),
            "text": docs.get(hit.id, {}).get("chunk_text", ""),
            "source_file": docs.get(hit.id, {}).get("source_file", ""),
            "chunk_index": docs.get(hit.id, {}).get("chunk_index", -1),
            "page": docs.get(hit.id, {}).get("page", -1)
        }
        for hit in search_result
    ]

    # Cache lại nếu là followup
    if request.type == "followup":
        redis_client.setex(
            f"q:{norm_q}",
            60 * 60 * 6,
            json.dumps({
                "answer": final_answer,
                "sources": source_info
            })
        )

    return {
        "question": q,
        "generated_answer": final_answer,
        "answers": source_info
    }
