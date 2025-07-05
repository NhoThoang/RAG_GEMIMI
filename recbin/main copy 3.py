from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from uuid import uuid4
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from vncorenlp import VnCoreNLP
import os

app = FastAPI()

# Load embedding model
model = SentenceTransformer("intfloat/multilingual-e5-base")

# Connect MongoDB & Qdrant
mongo_client = MongoClient("mongodb://localhost:27017")
mongo_db = mongo_client["books"]["chunks"]

qdrant = QdrantClient("localhost", port=6333)
collection_name = "book_chunks"

# Ensure Qdrant collection exists
if not qdrant.collection_exists(collection_name):
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

# Init VnCoreNLP for Vietnamese sentence splitting
vncorenlp = VnCoreNLP(
    r"D:\CODE2025\CUDA_PYTHON\Bench_mark\VnCoreNLP-1.2\VnCoreNLP-1.2.jar", 
    annotators="sent",
    max_heap_size='-Xmx2g'
)

def chunk_text(text, max_words=300):
    sentences_nested = vncorenlp.annotate(text)["sentences"]
    sentences = [" ".join(token["form"] for token in sent) for sent in sentences_nested]

    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        words = sent.split()
        if current_len + len(words) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_len = len(words)
        else:
            current_chunk.append(sent)
            current_len += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

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
            summary = chunk[:300]  # đơn giản: lấy 300 ký tự đầu làm tóm tắt
            vector = model.encode(summary).tolist()

            # Lưu vào MongoDB
            mongo_db.insert_one({
                "_id": chunk_id,
                "chunk_index": i,
                "summary": summary,
                "chunk_text": chunk,
                "source_file": file.filename
            })

            points.append(PointStruct(
                id=chunk_id,
                vector=vector,
                payload={
                    "summary": summary,
                    "source_file": file.filename,
                    "chunk_index": i
                }
            ))

        qdrant.upsert(collection_name=collection_name, points=points)
        return {"message": f"Uploaded and processed {len(chunks)} chunks from {file.filename}"}

    finally:
        os.remove(temp_path)
