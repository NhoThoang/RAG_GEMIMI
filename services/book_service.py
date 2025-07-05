import os
from uuid import uuid4
from PyPDF2 import PdfReader
from qdrant_client.http.models import PointStruct

from models.embedding_model import EmbeddingModel
from models.summarizer_model import SummarizerModel
from database.mongo_client import MongoClient
from database.qdrant_client import QdrantClient
from utils.text_utils import TextUtils

class BookService:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.summarizer_model = SummarizerModel()
        self.mongo_client = MongoClient()
        self.qdrant_client = QdrantClient()
    
    async def upload_book(self, file_content: bytes, filename: str):
        """Upload and process book"""
        temp_path = f"temp_{uuid4()}.pdf"
        
        try:
            # Save temporary file
            with open(temp_path, "wb") as f:
                f.write(file_content)
            
            # Process PDF
            reader = PdfReader(temp_path)
            chunks = []
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                for chunk in TextUtils.chunk_text(page_text, self.embedding_model):
                    chunks.append({"text": chunk, "page": page_num + 1})
            
            # Process chunks
            points = []
            for i, item in enumerate(chunks):
                chunk_id = str(uuid4())
                summary = self.summarizer_model.summarize(item["text"])
                vector = self.embedding_model.encode(summary).tolist()
                
                # Save to MongoDB
                self.mongo_client.insert_one({
                    "_id": chunk_id,
                    "chunk_index": i,
                    "summary": summary,
                    "chunk_text": item["text"],
                    "source_file": filename,
                    "page": item["page"]
                })
                
                # Prepare for Qdrant
                points.append(PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload={
                        "summary": summary,
                        "source_file": filename,
                        "chunk_index": i,
                        "page": item["page"]
                    }
                ))
            
            # Save to Qdrant
            self.qdrant_client.upsert(points)
            
            return {"message": f"Uploaded and processed {len(chunks)} chunks from {filename}"}
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path) 