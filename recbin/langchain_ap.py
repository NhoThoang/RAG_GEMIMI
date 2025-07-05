from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from uuid import uuid4
import os

# === Config Google Gemini ===
import google.generativeai as genai
genai.configure(api_key="YOUR_GEMINI_API_KEY")  # 👈 sửa lại key thật

# === FastAPI init ===
app = FastAPI()

# === Qdrant setup ===
qdrant_client = QdrantClient("localhost", port=6333)
COLLECTION_NAME = "book_chunks"

# === Embedding model ===
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

# === Chat model (Gemini) ===
chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-pro", google_api_key="YOUR_GEMINI_API_KEY")

# === Upload route ===
@app.post("/upload-book")
async def upload_book(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return JSONResponse(content={"error": "Only PDF files allowed"}, status_code=400)

    temp_path = f"temp_{uuid4()}.pdf"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        loader = PyPDFLoader(temp_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(pages)

        # Tạo vector store từ documents và embedding
        vectorstore = Qdrant.from_documents(
            documents=chunks,
            embedding=embedding_model,
            client=qdrant_client,
            collection_name=COLLECTION_NAME
        )

        return {"message": f"Uploaded and processed {len(chunks)} chunks from {file.filename}"}
    finally:
        os.remove(temp_path)

# === Ask route ===
@app.get("/ask")
async def ask_question(q: str = Query(..., alias="question")):
    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embedding_model
    )

    qa = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=vectorstore.as_retriever()
    )

    answer = qa.run(q)
    return {"question": q, "answer": answer}
